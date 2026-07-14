#!/usr/bin/env python3
"""
Resource Usage and Power Monitor

Launches a target program as a child process and continuously monitors its 
resource usage (Raw CPU, Normalized CPU, RAM, GPU, Power) until it exits.

Upgraded to use `pyJoules` for standardized, cross-platform hardware energy reading
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
import matplotlib.pyplot as plt
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
        
        # 1. Try initializing pyJoules for native CPU power tracking
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
            print("[INFO] pyJoules library not found. Run 'pip install pyJoules' for native power tracking.")

        if not self.use_pyjoules:
            print(f"[INFO] Using Fallback Power Estimation: {self.base}W + (CPU% * ({self.tdp}W - {self.base}W)).")

    def get_gpu_metrics(self):
        """Returns GPU Utilization (%) and Power (Watts) via NVML"""
        if not HAS_GPU:
            return 0.0, 0.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
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
                if duration <= 0: return 0.0
                
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
        """Combined total process power."""
        _, gpu_w = self.get_gpu_metrics()
        cpu_w = self.get_cpu_power(cpu_normalized_percent)
        return cpu_w + gpu_w

# ---------------------------------------------------------
# System Measurement Loop
# ---------------------------------------------------------

def collect_metrics(root_proc, proc_cache, power_monitor, start_time):
    total_cpu_raw = 0.0
    total_ram_mb = 0.0
    num_cores = power_monitor.num_cores

    try:
        current_procs = [root_proc] + root_proc.children(recursive=True)
        current_pids = {p.pid for p in current_procs}
        
        for p in current_procs:
            if p.pid not in proc_cache:
                proc_cache[p.pid] = p
                try: p.cpu_percent(interval=None)
                except psutil.NoSuchProcess: pass

        for pid in list(proc_cache.keys()):
            if pid not in current_pids:
                del proc_cache[pid]

        for pid, p in proc_cache.items():
            try:
                total_cpu_raw += p.cpu_percent(interval=None)
                total_ram_mb += p.memory_info().rss / 1048576.0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except psutil.NoSuchProcess:
        pass

    total_cpu_normalized = total_cpu_raw / num_cores
    gpu_pct, _ = power_monitor.get_gpu_metrics()
    watts = power_monitor.get_total_power(total_cpu_normalized)
    
    return {
        "timestamp": time.time(),
        "elapsed_time": time.time() - start_time,
        "cpu_percent_raw": total_cpu_raw,
        "cpu_percent_normalized": total_cpu_normalized,
        "ram_mb": total_ram_mb,
        "gpu_percent": gpu_pct,
        "watts": watts
    }

# ---------------------------------------------------------
# Logging & Visualization
# ---------------------------------------------------------

def save_logs(data, base_filename):
    if not data: return

    csv_file = f"logs/{base_filename}_metrics.csv"
    json_file = f"logs/{base_filename}_summary.json"

    keys = ["timestamp", "elapsed_time", "cpu_percent_raw", "cpu_percent_normalized", "ram_mb", "gpu_percent", "watts"]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    total_runtime = data[-1]["elapsed_time"]
    avg_cpu_raw = sum(d["cpu_percent_raw"] for d in data) / len(data)
    avg_cpu_norm = sum(d["cpu_percent_normalized"] for d in data) / len(data)
    peak_ram = max(d["ram_mb"] for d in data)
    avg_gpu = sum(d["gpu_percent"] for d in data) / len(data)

    energy_wh = 0.0
    for i in range(1, len(data)):
        delta_t = data[i]["timestamp"] - data[i-1]["timestamp"]
        energy_wh += data[i]["watts"] * (delta_t / 3600.0)

    summary = {
        "total_runtime_seconds": round(total_runtime, 2),
        "average_cpu_raw_percent": round(avg_cpu_raw, 2),
        "average_cpu_normalized_percent": round(avg_cpu_norm, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "average_gpu_percent": round(avg_gpu, 2),
        "estimated_total_energy_wh": round(energy_wh, 6)
    }

    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n[INFO] Logs saved: {csv_file}, {json_file}")
    for k, v in summary.items():
        print(f"       - {k}: {v}")


def generate_plots(data, base_filename, display):
    if not data: return

    times = [d["elapsed_time"] for d in data]
    cpus_norm = [d["cpu_percent_normalized"] for d in data]
    rams = [d["ram_mb"] for d in data]
    gpus = [d["gpu_percent"] for d in data]
    watts = [d["watts"] for d in data]

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Resource Usage Over Time: {base_filename}")

    axs[0].plot(times, cpus_norm, color='tab:blue')
    axs[0].set_ylabel('CPU Normalized (%)')
    axs[0].set_ylim(bottom=0, top=max(105, max(cpus_norm) + 5))
    axs[0].grid(True, linestyle='--', alpha=0.7)

    axs[1].plot(times, rams, color='tab:orange')
    axs[1].set_ylabel('RAM Usage (MB)')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].plot(times, gpus, color='tab:green')
    axs[2].set_ylabel('GPU Usage (%)')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    axs[3].plot(times, watts, color='tab:red')
    axs[3].set_ylabel('Power (Watts)')
    axs[3].set_xlabel('Elapsed Time (s)')
    axs[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_file = f"logs/{base_filename}_plots.png"
    plt.savefig(plot_file)

    if display:
        plt.show()

# ---------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------

def run_target_process(command, interval, display, tdp, base):
    print(f"\n[START] Command: {' '.join(command)}")
    power_monitor = PowerMonitor(tdp=tdp, base=base)
    data = []
    proc_cache = {}

    try:
        process = subprocess.Popen(command)
        root_proc = psutil.Process(process.pid)
    except Exception as e:
        print(f"\n[ERROR] Failed to start process: {e}")
        sys.exit(1)

    start_time = time.time()
    last_print_time = start_time
    
    try:
        while process.poll() is None:
            metrics = collect_metrics(root_proc, proc_cache, power_monitor, start_time)
            data.append(metrics)
            
            now = time.time()
            if now - last_print_time >= 1.0:
                sys.stdout.write(
                    f"\r[LIVE] Time: {metrics['elapsed_time']:>5.1f}s | "
                    f"CPU(Norm): {metrics['cpu_percent_normalized']:>5.1f}% | "
                    f"CPU(Raw): {metrics['cpu_percent_raw']:>6.1f}% | "
                    f"RAM: {metrics['ram_mb']:>6.1f}MB | "
                    f"Pwr: {metrics['watts']:>5.1f}W"
                )
                sys.stdout.flush()
                last_print_time = now

            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n[WARNING] Terminating target process...")
        try:
            for child in root_proc.children(recursive=True): child.terminate()
            root_proc.terminate()
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass

    print("\n\n[FINISHED] Process exited.")
    base_filename = f"{os.path.basename(command[0])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_logs(data, base_filename)
    generate_plots(data, base_filename, display)
    sys.exit(process.returncode if process.returncode is not None else 0)


# ---------------------------------------------------------
# Argument Parsing  ← CORRECCIÓN PRINCIPAL
# ---------------------------------------------------------

def main():
    """
    Usa parse_known_args en lugar de argparse.REMAINDER para que los flags
    del subcomando (--model, --save_predictions, etc.) no colisionen con
    los flags del propio monitor (--interval, --tdp, --base, --display).

    El separador -- es la forma recomendada de delimitar ambos conjuntos:

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
        help="Intervalo de muestreo en segundos (default: 0.5)"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Mostrar gráficas al finalizar (requiere entorno gráfico)"
    )
    parser.add_argument(
        "--tdp", type=float, default=65.0,
        help="TDP máximo del CPU en Watts para estimación fallback (default: 65.0)"
    )
    parser.add_argument(
        "--base", type=float, default=0.0,
        help="Consumo base del CPU en Watts (default: 0.0)"
    )

    # parse_known_args devuelve (namespace, lista_de_extras).
    # Todo lo que venga tras -- (o cualquier token no reconocido) queda
    # en `remaining` intacto, sin que argparse intente interpretarlo.
    args, remaining = parser.parse_known_args()

    # Limpiar el separador -- si el usuario lo incluyó explícitamente
    # (parse_known_args lo deja en remaining como token vacío en algunos entornos)
    command = [tok for tok in remaining if tok != '--']

    if not command:
        parser.print_help()
        print(
            "\n[ERROR] No se especificó ningún comando para monitorizar.\n"
            "Uso:  python monitor.py [opciones monitor] -- comando [opciones comando]\n"
            "Ej:   python monitor.py -- python inference.py --model logreg\n"
        )
        sys.exit(1)

    run_target_process(command, args.interval, args.display, args.tdp, args.base)


if __name__ == "__main__":
    main()