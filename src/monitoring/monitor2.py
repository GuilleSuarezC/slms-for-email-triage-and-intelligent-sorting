#!/usr/bin/env python3
"""
Resource Usage and Power Monitor

Launches a target program as a child process and continuously monitors its
resource usage (Raw CPU, Normalized CPU, RAM, GPU, Power) until it exits.

Uses `pyJoules` for standardized, cross-platform hardware energy reading
(Intel RAPL, AMD RAPL). Falls back to TDP estimation if unavailable.

Usage:
    python monitor.py [--interval 0.5] [--display] [--tdp 65.0] [--base 10.0] -- script.py args

    # Ejemplos con inference.py:
    python monitor.py -- python inference.py --model logreg
    python monitor.py -- python inference.py --model svm --save_predictions
    python monitor.py --interval 1.0 --tdp 95.0 -- python inference.py --model rf

    NOTA: el separador -- es obligatorio cuando el subcomando tiene sus propios
    flags (--model, --save_predictions, etc.) para que argparse no los confunda
    con flags del propio monitor.
"""

import sys
import os
import time
import csv
import json
import subprocess
import argparse
import psutil
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# Optional Library Loaders
# ---------------------------------------------------------

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_GPU = True
except (ImportError, Exception):
    HAS_GPU = False

try:
    from pyJoules.energy_meter import EnergyMeter
    from pyJoules.device.rapl_device import RaplDevice
    HAS_PYJOULES = True
except ImportError:
    HAS_PYJOULES = False


# ---------------------------------------------------------
# Power Monitoring Core
# ---------------------------------------------------------

class PowerMonitor:
    def __init__(self, tdp=65.0, base=0.0):
        self.num_cores = psutil.cpu_count(logical=True)
        self.tdp = tdp
        self.base = base

        self.meter = None
        self.use_pyjoules = False

        if HAS_PYJOULES:
            try:
                domains = RaplDevice().get_configured_domains()
                if domains:
                    self.meter = EnergyMeter(domains)
                    self.meter.start()
                    self.use_pyjoules = True
                    print("[INFO] pyJoules successfully attached to hardware RAPL domains.")
            except Exception as e:
                print(f"[WARNING] pyJoules found but hardware access denied/unavailable ({e}).")
                print("          (Note: Linux requires root/sudo to read RAPL hardware counters).")
        else:
            print("[INFO] pyJoules not found. Run 'pip install pyJoules' for native power tracking.")

        if not self.use_pyjoules:
            print(f"[INFO] Fallback power estimation: {self.base}W + (CPU% × ({self.tdp}W - {self.base}W)).")

    def get_gpu_metrics(self):
        """Returns GPU Utilization (%) and Power (Watts) via NVML."""
        if not HAS_GPU:
            return 0.0, 0.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_pct = float(util.gpu)
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            return gpu_pct, power_w
        except Exception:
            return 0.0, 0.0

    def get_cpu_power(self, cpu_normalized_percent):
        """Returns CPU Watts using pyJoules (Native) or Fallback Estimation."""
        share = min(1.0, cpu_normalized_percent / 100.0)

        if self.use_pyjoules:
            try:
                self.meter.record('tick')
                trace = self.meter.get_trace()
                sample = trace[-1]

                if len(trace) > 1000:
                    self.meter._trace.clear()

                duration = sample.duration
                if duration <= 0:
                    return 0.0

                cpu_energy_uj = 0
                for domain_name, energy_uj in sample.energy.items():
                    if 'package' in str(domain_name).lower():
                        cpu_energy_uj += energy_uj

                total_cpu_watts = (cpu_energy_uj / 1_000_000.0) / duration
                return max(0.0, total_cpu_watts * share)

            except Exception:
                pass

        return self.base + (share * (self.tdp - self.base))

    def get_total_power(self, cpu_normalized_percent):
        """Combined total process power (CPU + GPU)."""
        _, gpu_w = self.get_gpu_metrics()
        cpu_w    = self.get_cpu_power(cpu_normalized_percent)
        return cpu_w + gpu_w


# ---------------------------------------------------------
# System Measurement Loop
# ---------------------------------------------------------

def collect_metrics(root_proc, proc_cache, power_monitor, start_time):
    total_cpu_raw = 0.0
    total_ram_mb  = 0.0
    num_cores     = power_monitor.num_cores

    try:
        current_procs = [root_proc] + root_proc.children(recursive=True)
        current_pids  = {p.pid for p in current_procs}

        for p in current_procs:
            if p.pid not in proc_cache:
                proc_cache[p.pid] = p
                try:
                    p.cpu_percent(interval=None)
                except psutil.NoSuchProcess:
                    pass

        for pid in list(proc_cache.keys()):
            if pid not in current_pids:
                del proc_cache[pid]

        for pid, p in proc_cache.items():
            try:
                total_cpu_raw += p.cpu_percent(interval=None)
                total_ram_mb  += p.memory_info().rss / 1_048_576.0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except psutil.NoSuchProcess:
        pass

    total_cpu_normalized = total_cpu_raw / num_cores
    gpu_pct, _           = power_monitor.get_gpu_metrics()
    watts                = power_monitor.get_total_power(total_cpu_normalized)

    return {
        "timestamp":              time.time(),
        "elapsed_time":           time.time() - start_time,
        "cpu_percent_raw":        total_cpu_raw,
        "cpu_percent_normalized": total_cpu_normalized,
        "ram_mb":                 total_ram_mb,
        "gpu_percent":            gpu_pct,
        "watts":                  watts,
    }


# ---------------------------------------------------------
# Data Cleaning & Resampling  (FIX for duplicate-X problem)
# ---------------------------------------------------------

def resample_timeseries(data: list[dict], target_interval: float = 1.0) -> list[dict]:
    """
    Resamplea la serie temporal a intervalos regulares de ``target_interval``
    segundos para eliminar el problema de múltiples puntos en el mismo valor
    de X en los gráficos.

    El problema original:
        El bucle de monitorización muestrea con ``time.sleep(interval)`` pero
        el tiempo real entre muestras varía dependiendo del coste de
        ``collect_metrics()``. Cuando el proceso monitorizado es muy corto
        (< 5 s) o el intervalo es muy pequeño (< 0.1 s) pueden acumularse
        decenas de puntos con elapsed_time casi idéntico, lo que produce
        gráficas con líneas verticales y ejes X ilegibles.

    La solución:
        1. Ordenar las muestras por elapsed_time (por si llegaron
           ligeramente desordenadas por jitter del SO).
        2. Construir una rejilla uniforme de timestamps desde 0 hasta
           el último elapsed_time con paso ``target_interval``.
        3. Para cada celda de la rejilla, promediar todas las muestras
           originales que caen en ese intervalo. Si la celda está vacía
           (hueco en la monitorización) se interpola linealmente.

    Parameters
    ----------
    data            : lista de dicts devueltos por collect_metrics()
    target_interval : segundos entre puntos del resultado (default: 1.0 s)

    Returns
    -------
    Lista de dicts con las mismas claves que la entrada, con timestamps
    uniformemente espaciados.
    """
    if not data:
        return data

    # 1. Ordenar por tiempo transcurrido
    data = sorted(data, key=lambda d: d["elapsed_time"])

    t_start = data[0]["elapsed_time"]
    t_end   = data[-1]["elapsed_time"]
    duration = t_end - t_start

    # Si la duración total es menor que un intervalo, devolver tal cual
    # (no hay nada que resamplear: el proceso fue instantáneo)
    if duration < target_interval:
        return data

    # 2. Construir rejilla uniforme
    grid = np.arange(t_start, t_end + target_interval, target_interval)

    numeric_keys = [
        "cpu_percent_raw", "cpu_percent_normalized",
        "ram_mb", "gpu_percent", "watts",
    ]

    # Arrays de las series originales para interpolación
    orig_t    = np.array([d["elapsed_time"] for d in data])
    orig_vals = {k: np.array([d[k] for d in data]) for k in numeric_keys}

    resampled = []
    for t_grid in grid:
        # Muestras que caen dentro de la celda [t_grid, t_grid + target_interval)
        mask = (orig_t >= t_grid) & (orig_t < t_grid + target_interval)

        row: dict = {"elapsed_time": float(t_grid)}

        for k in numeric_keys:
            cell_vals = orig_vals[k][mask]
            if cell_vals.size > 0:
                # Promedio de las muestras en la celda
                row[k] = float(cell_vals.mean())
            else:
                # Celda vacía: interpolación lineal con los vecinos más cercanos
                row[k] = float(np.interp(t_grid, orig_t, orig_vals[k]))

        # timestamp absoluto: reconstruir a partir del primero
        row["timestamp"] = data[0]["timestamp"] + t_grid
        resampled.append(row)

    return resampled


def smooth_series(values: list[float], window: int = 3) -> list[float]:
    """
    Suavizado por media móvil centrada con ventana ``window``.

    Usado en los gráficos para reducir el ruido de alta frecuencia que
    dificulta la lectura de tendencias (especialmente en CPU y Watts).
    No modifica los datos almacenados en CSV/JSON, solo la visualización.

    Parameters
    ----------
    values : lista de floats
    window : tamaño de la ventana (número de puntos). 1 = sin suavizado.

    Returns
    -------
    Lista de floats suavizada, misma longitud que la entrada.
    """
    if window <= 1 or len(values) < window:
        return values

    arr    = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    # 'same' mantiene la longitud; 'edge' rellena los bordes con el primer/último valor
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    # Ajustar longitud exacta por diferencias de redondeo en el padding
    return smoothed[:len(values)].tolist()


# ---------------------------------------------------------
# Logging & Visualization
# ---------------------------------------------------------

def save_logs(data: list[dict], base_filename: str):
    """Guarda el CSV con todas las muestras originales y el JSON de resumen."""
    if not data:
        return

    os.makedirs("logs", exist_ok=True)
    csv_file  = f"logs/{base_filename}_metrics.csv"
    json_file = f"logs/{base_filename}_summary.json"

    keys = [
        "timestamp", "elapsed_time",
        "cpu_percent_raw", "cpu_percent_normalized",
        "ram_mb", "gpu_percent", "watts",
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    total_runtime = data[-1]["elapsed_time"]
    n             = len(data)
    avg_cpu_raw   = sum(d["cpu_percent_raw"]        for d in data) / n
    avg_cpu_norm  = sum(d["cpu_percent_normalized"] for d in data) / n
    peak_ram      = max(d["ram_mb"]                 for d in data)
    avg_gpu       = sum(d["gpu_percent"]            for d in data) / n

    energy_wh = 0.0
    for i in range(1, n):
        delta_t    = data[i]["timestamp"] - data[i - 1]["timestamp"]
        energy_wh += data[i]["watts"] * (delta_t / 3600.0)

    summary = {
        "total_runtime_seconds":          round(total_runtime, 2),
        "n_samples":                      n,
        "average_cpu_raw_percent":        round(avg_cpu_raw,  2),
        "average_cpu_normalized_percent": round(avg_cpu_norm, 2),
        "peak_ram_mb":                    round(peak_ram,     2),
        "average_gpu_percent":            round(avg_gpu,      2),
        "estimated_total_energy_wh":      round(energy_wh,    6),
    }

    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n[INFO] Logs saved: {csv_file}, {json_file}")
    for k, v in summary.items():
        print(f"       - {k}: {v}")


def generate_plots(
    data: list[dict],
    base_filename: str,
    display: bool,
    resample_interval: float = 1.0,
    smooth_window: int = 3,
):
    """
    Genera y guarda los gráficos de uso de recursos.

    Pasos de limpieza aplicados antes de graficar:
        1. resample_timeseries() → elimina múltiples puntos en el mismo X
           resamplando a una rejilla uniforme de ``resample_interval`` segundos.
        2. smooth_series()       → suavizado por media móvil para reducir
           el ruido de alta frecuencia (solo afecta a la visualización).

    Parameters
    ----------
    data              : muestras originales (sin modificar)
    base_filename     : nombre base para el fichero de salida
    display           : si True, llama a plt.show() (requiere entorno gráfico)
    resample_interval : segundos entre puntos del gráfico (default: 1.0 s)
    smooth_window     : ventana de suavizado en número de puntos (default: 3)
    """
    if not data:
        return

    os.makedirs("logs", exist_ok=True)

    # ── 1. Resamplear a rejilla uniforme ───────────────────────────────
    plot_data = resample_timeseries(data, target_interval=resample_interval)

    times     = [d["elapsed_time"]           for d in plot_data]
    cpus_norm = [d["cpu_percent_normalized"] for d in plot_data]
    rams      = [d["ram_mb"]                 for d in plot_data]
    gpus      = [d["gpu_percent"]            for d in plot_data]
    watts_raw = [d["watts"]                  for d in plot_data]

    # ── 2. Suavizar series ruidosas ────────────────────────────────────
    cpus_smooth  = smooth_series(cpus_norm,  window=smooth_window)
    watts_smooth = smooth_series(watts_raw,  window=smooth_window)
    # RAM y GPU son más estables; se grafican directamente sin suavizar

    # ── 3. Construir figura ────────────────────────────────────────────
    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(
        f"Resource Usage — {base_filename}\n"
        f"({len(data)} raw samples → resampled to {resample_interval}s grid, "
        f"smooth window={smooth_window})",
        fontsize=11,
    )

    duration = times[-1] if times else 1.0

    # Función auxiliar para anotar máximo y promedio en cada subplot
    def _annotate(ax, y_vals, unit=""):
        if not y_vals:
            return
        peak = max(y_vals)
        avg  = sum(y_vals) / len(y_vals)
        ax.axhline(avg,  color="grey",   linestyle=":",  linewidth=1.0, alpha=0.7)
        ax.axhline(peak, color="salmon", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(
            duration * 0.01, peak,
            f" peak {peak:.1f}{unit}", va="bottom",
            fontsize=7, color="salmon",
        )
        ax.text(
            duration * 0.01, avg,
            f" avg {avg:.1f}{unit}", va="top",
            fontsize=7, color="grey",
        )

    # ── Panel 0: CPU normalizado ───────────────────────────────────────
    axs[0].plot(times, cpus_norm,  color="tab:blue", alpha=0.25, linewidth=0.8,
                label="raw (resampled)")
    axs[0].plot(times, cpus_smooth, color="tab:blue", linewidth=1.5,
                label=f"smoothed (w={smooth_window})")
    axs[0].set_ylabel("CPU Normalized (%)")
    axs[0].set_ylim(bottom=0, top=max(105, max(cpus_norm) * 1.15) if cpus_norm else 110)
    axs[0].legend(fontsize=7, loc="upper right")
    axs[0].grid(True, linestyle="--", alpha=0.4)
    _annotate(axs[0], cpus_smooth, unit="%")

    # ── Panel 1: RAM ───────────────────────────────────────────────────
    axs[1].plot(times, rams, color="tab:orange", linewidth=1.5)
    axs[1].fill_between(times, rams, alpha=0.15, color="tab:orange")
    axs[1].set_ylabel("RAM Usage (MB)")
    axs[1].set_ylim(bottom=0)
    axs[1].grid(True, linestyle="--", alpha=0.4)
    _annotate(axs[1], rams, unit=" MB")

    # ── Panel 2: GPU ───────────────────────────────────────────────────
    axs[2].plot(times, gpus, color="tab:green", linewidth=1.5)
    axs[2].fill_between(times, gpus, alpha=0.15, color="tab:green")
    axs[2].set_ylabel("GPU Usage (%)")
    axs[2].set_ylim(bottom=0, top=105)
    axs[2].grid(True, linestyle="--", alpha=0.4)
    _annotate(axs[2], gpus, unit="%")

    # ── Panel 3: Potencia ──────────────────────────────────────────────
    axs[3].plot(times, watts_raw,    color="tab:red", alpha=0.25, linewidth=0.8,
                label="raw (resampled)")
    axs[3].plot(times, watts_smooth, color="tab:red", linewidth=1.5,
                label=f"smoothed (w={smooth_window})")
    axs[3].fill_between(times, watts_smooth, alpha=0.10, color="tab:red")
    axs[3].set_ylabel("Power (Watts)")
    axs[3].set_xlabel("Elapsed Time (s)")
    axs[3].set_ylim(bottom=0)
    axs[3].legend(fontsize=7, loc="upper right")
    axs[3].grid(True, linestyle="--", alpha=0.4)
    _annotate(axs[3], watts_smooth, unit=" W")

    # ── Eje X: ticks legibles independientemente de la duración ───────
    # Calcular un paso de tick que produzca entre 8 y 15 ticks en el eje
    for ax in axs:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12, integer=True))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    plt.tight_layout()

    plot_file = f"logs/{base_filename}_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")

    if display:
        plt.show()

    plt.close(fig)
    print(f"[INFO] Plot saved: {plot_file}")


# ---------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------

def run_target_process(
    command: list[str],
    interval: float,
    display: bool,
    tdp: float,
    base: float,
    resample_interval: float,
    smooth_window: int,
):
    print(f"\n[START] Command: {' '.join(command)}")
    power_monitor = PowerMonitor(tdp=tdp, base=base)
    data:       list[dict] = []
    proc_cache: dict       = {}

    try:
        process   = subprocess.Popen(command)
        root_proc = psutil.Process(process.pid)
    except Exception as e:
        print(f"\n[ERROR] Failed to start process: {e}")
        sys.exit(1)

    start_time      = time.time()
    last_print_time = start_time

    try:
        while process.poll() is None:
            metrics = collect_metrics(root_proc, proc_cache, power_monitor, start_time)
            data.append(metrics)

            now = time.time()
            if now - last_print_time >= 1.0:
                sys.stdout.write(
                    f"\r[LIVE] Time: {metrics['elapsed_time']:>6.1f}s | "
                    f"CPU(Norm): {metrics['cpu_percent_normalized']:>5.1f}% | "
                    f"CPU(Raw): {metrics['cpu_percent_raw']:>6.1f}% | "
                    f"RAM: {metrics['ram_mb']:>7.1f} MB | "
                    f"Pwr: {metrics['watts']:>5.1f} W"
                )
                sys.stdout.flush()
                last_print_time = now

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Terminating target process...")
        try:
            for child in root_proc.children(recursive=True):
                child.terminate()
            root_proc.terminate()
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass

    print("\n\n[FINISHED] Process exited.")

    base_filename = (
        f"{os.path.splitext(os.path.basename(command[0]))[0]}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    save_logs(data, base_filename)
    generate_plots(
        data, base_filename, display,
        resample_interval=resample_interval,
        smooth_window=smooth_window,
    )

    sys.exit(process.returncode if process.returncode is not None else 0)


# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------

def main():
    """
    Usa parse_known_args para que los flags del subcomando
    (--model, --save_predictions, etc.) no colisionen con los del monitor.

    El separador -- delimita ambos conjuntos:

        python monitor.py [flags monitor] -- comando [flags comando]

    Ejemplos:
        python monitor.py -- python inference.py --model logreg
        python monitor.py --interval 1.0 -- python inference.py --model svm --save_predictions
        python monitor.py --tdp 95.0 --base 15.0 -- python inference.py --model rf
        python monitor.py -- python spam_classifier_experiment.py
    """
    parser = argparse.ArgumentParser(
        description="Resource & Power Monitor — wraps any command as a child process.",
        epilog=(
            "Separa los flags del monitor del comando con --:\n"
            "  python monitor.py --interval 1.0 -- python inference.py --model logreg\n"
            "  python monitor.py -- python spam_classifier_experiment.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Intervalo de muestreo en segundos (default: 0.5)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Mostrar gráficas al finalizar (requiere entorno gráfico)",
    )
    parser.add_argument(
        "--tdp", type=float, default=65.0,
        help="TDP máximo del CPU en Watts para estimación fallback (default: 65.0)",
    )
    parser.add_argument(
        "--base", type=float, default=0.0,
        help="Consumo base del CPU en Watts (default: 0.0)",
    )
    parser.add_argument(
        "--resample", type=float, default=1.0, metavar="SECONDS",
        help=(
            "Intervalo de la rejilla uniforme para los gráficos en segundos "
            "(default: 1.0). Reduce este valor si el proceso dura menos de 10 s."
        ),
    )
    parser.add_argument(
        "--smooth", type=int, default=3, metavar="WINDOW",
        help=(
            "Ventana de suavizado (media móvil) para CPU y Watts en los gráficos "
            "(default: 3). Usa 1 para desactivar el suavizado."
        ),
    )

    args, remaining = parser.parse_known_args()
    command = [tok for tok in remaining if tok != "--"]

    if not command:
        parser.print_help()
        print(
            "\n[ERROR] No se especificó ningún comando.\n"
            "Uso:  python monitor.py [opciones] -- comando [opciones comando]\n"
            "Ej:   python monitor.py -- python inference.py --model logreg\n"
        )
        sys.exit(1)

    run_target_process(
        command=command,
        interval=args.interval,
        display=args.display,
        tdp=args.tdp,
        base=args.base,
        resample_interval=args.resample,
        smooth_window=args.smooth,
    )


if __name__ == "__main__":
    main()