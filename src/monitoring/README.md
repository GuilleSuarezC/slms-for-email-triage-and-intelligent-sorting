
# CPU Burner & Resource Monitor

A lightweight toolkit for **synthetic system load generation** and **resource monitoring**.

It contains two main tools:

- **`cpu_burner.py`** – Generates controllable CPU and RAM load with optional sine-wave fluctuations.
- **`monitor.py`** – Launches and monitors a process while recording CPU, RAM, GPU, and power usage.

Together they allow you to **stress-test systems**, **benchmark performance**, and **measure power consumption**.

---

# Features

### CPU & RAM Load Generator
- Multi-core CPU stress testing
- Realistic **sine-wave utilization patterns**
- Memory allocation with **real physical page commitment**
- Configurable runtime and intensity
- Adjustable RAM fluctuation window
- Flat or dynamic load modes

### Resource Monitor
- Tracks **CPU (raw & normalized)**
- Tracks **RAM usage**
- Tracks **GPU utilization and power (NVML)**
- Tracks **CPU power consumption**
  - Hardware energy via **pyJoules / RAPL**
  - Fallback **TDP-based estimation**
- Outputs:
  - CSV metrics log
  - JSON summary report
  - Performance plots

---

# Repository Structure

```

.
├── cpu_burner.py     # Synthetic CPU/RAM load generator
├── monitor.py        # Process resource and power monitor
└── README.md

````

---

# Installation

Python **3.8+** recommended.

Install required dependencies:

```bash
pip install psutil matplotlib
````

Optional features:

### GPU Monitoring

```bash
pip install nvidia-ml-py3
```

### Native CPU Power Monitoring

```bash
pip install pyJoules
```

Note:

* **pyJoules requires root privileges on Linux** to read RAPL counters.

---

# CPU Burner

`cpu_burner.py` generates synthetic CPU and memory load with **realistic fluctuation patterns**.

## Example Usage

Run a **30-second test** using 4 CPU cores and 2 GB RAM:

```bash
python cpu_burner.py --duration 30 --cores 4 --ram-mb 2000
```

Run with **dynamic load fluctuations**:

```bash
python cpu_burner.py --duration 30 --cores 4 --ram-mb 2000 --wave-period 10
```

Run **flat memory load (no wave)**:

```bash
python cpu_burner.py --duration 30 --cores 4 --ram-mb 2000 --no-wave
```

---

## CPU Load Model

CPU load is generated using a **sine-wave duty cycle**:

```
CPU intensity = (sin(t / period * 2π) + 1) / 2
```

This produces realistic load patterns:

```
Low → Medium → Peak → Medium → Low
```

instead of constant stress.

---

## RAM Allocation Strategy

The RAM worker:

1. **Pre-allocates full memory** using a large `bytearray`
2. **Touches every memory page** to force physical allocation
3. Keeps memory resident via periodic page writes

With waves enabled:

```
Active RAM fluctuates between:

min_ram_pct → 100%
```

Example:

```
--ram-mb 2000
--min-ram-pct 25
```

Active memory oscillates between:

```
500 MB → 2000 MB
```

---

## CPU Burner Arguments

| Argument        | Description                      | Default   |
| --------------- | -------------------------------- | --------- |
| `--duration`    | Runtime in seconds               | 30        |
| `--cores`       | Number of CPU worker processes   | All cores |
| `--ram-mb`      | RAM to allocate in MB            | 1024      |
| `--wave-period` | Duration of sine-wave cycle      | 10s       |
| `--no-wave`     | Disable RAM fluctuations         | False     |
| `--min-ram-pct` | Minimum RAM active during trough | 25        |

---

# Resource Monitor

`monitor.py` launches a program and records **system resource usage over time**.

Example:

```bash
python monitor.py python cpu_burner.py --duration 30
```

---

## Live Monitoring Output

During execution you will see:

```
[LIVE] Time: 12.0s | CPU(Norm): 95.1% | CPU(Raw): 760.0% | RAM: 2048MB | Pwr: 82.4W
```

Metrics include:

* CPU usage (raw)
* CPU usage normalized per core
* RAM consumption
* GPU utilization
* Estimated power consumption

---

# Power Monitoring

The monitor supports **two power measurement modes**.

### 1. Hardware Measurement (Preferred)

Using **pyJoules**:

* Intel RAPL
* AMD RAPL

This measures **real CPU energy usage**.

Requires:

```bash
sudo python monitor.py ...
```

---

### 2. Fallback Estimation

If RAPL is unavailable:

```
Power = Base + CPU% * (TDP - Base)
```

Example:

```bash
--tdp 65
--base 10
```

Meaning:

```
Idle = 10W
Max = 65W
```

---

# Monitor Arguments

| Argument     | Description                         | Default |
| ------------ | ----------------------------------- | ------- |
| `--interval` | Sampling interval (seconds)         | 0.5     |
| `--display`  | Show plots interactively            | False   |
| `--tdp`      | CPU TDP for fallback power estimate | 65W     |
| `--base`     | Base idle power                     | 0W      |

---

# Output Files

After execution the monitor generates:

### Metrics Log

```
scriptname_YYYYMMDD_HHMMSS_metrics.csv
```

Contains:

```
timestamp
elapsed_time
cpu_percent_raw
cpu_percent_normalized
ram_mb
gpu_percent
watts
```

---

### Summary Report

```
scriptname_YYYYMMDD_HHMMSS_summary.json
```

Example:

```json
{
  "total_runtime_seconds": 30.02,
  "average_cpu_raw_percent": 730.2,
  "average_cpu_normalized_percent": 91.3,
  "peak_ram_mb": 2048.0,
  "average_gpu_percent": 0.0,
  "estimated_total_energy_wh": 0.0165
}
```

---

### Performance Plots

```
scriptname_YYYYMMDD_HHMMSS_plots.png
```

Includes graphs for:

* CPU usage
* RAM usage
* GPU utilization
* Power consumption

---

# Example Workflow

Generate synthetic load:

```bash
python cpu_burner.py --duration 60 --cores 8 --ram-mb 4000
```

Monitor it:

```bash
python monitor.py python cpu_burner.py --duration 60 --cores 8 --ram-mb 4000
```

---

# Use Cases

This toolkit is useful for:

* Performance benchmarking
* Server stress testing
* Thermal testing
* Power consumption measurement
* Monitoring system behavior under fluctuating load
* Profiling applications

---

# License

MIT License


