#!/usr/bin/env python3
"""
CPU & RAM Burner with Fluctuations
Generates synthetic load across multiple cores and memory. Uses a sine-wave
duty cycle to simulate real-world fluctuating bursts.

Usage:
    python cpu_burner.py --duration 30 --cores 4 --ram-mb 2000 --wave-period 10
    python cpu_burner.py --duration 30 --cores 4 --ram-mb 2000 --no-wave
"""
import time
import argparse
import multiprocessing
import math
import random
import ctypes
import sys


def cpu_worker(duration, wave_period):
    """
    Fluctuates CPU usage based on a Sine Wave.
    """
    start_time = time.time()
    end_time = start_time + duration

    # Update frequency scales with the wave period to ensure smooth tracking
    window = min(0.02, wave_period / 10.0)

    while time.time() < end_time:
        elapsed = time.time() - start_time

        # Calculate sine wave position (0.0 to 1.0)
        intensity = (math.sin((elapsed / wave_period) * 2 * math.pi) + 1) / 2

        busy_time = window * intensity
        sleep_time = window - busy_time

        # Busy wait (The "Burn")
        t_end_busy = time.time() + busy_time
        while time.time() < t_end_busy:
            _ = math.sqrt(random.randint(1, 10000)) ** 0.5

        # Idle (The "Dip")
        if sleep_time > 0:
            time.sleep(sleep_time)


def _touch_pages(buf, length):
    """Write one byte per 4 KB page to force physical (RSS) allocation."""
    for i in range(0, length, 4096):
        buf[i] = (buf[i] + 1) & 0xFF   # dirty each page without zeroing


def ram_worker(duration, max_ram_mb, wave_period, use_wave, min_ram_fraction):
    """
    Allocates the requested RAM and keeps it resident in physical memory.

    Strategy
    --------
    1.  Pre-allocate the full ``max_ram_mb`` into a single contiguous
        ``bytearray`` and touch every page so the OS maps it into RSS
        immediately – no silent virtual-memory tricks.
    2.  When ``--wave`` is active the *active window* inside that buffer
        slides between ``min_ram_fraction`` and 100 % of the allocation,
        re-touching pages every tick so they stay hot.  Pages outside the
        window are still allocated but not touched, so the OS *may* page
        them out under pressure – which is the desired fluctuation effect.
    3.  When ``--no-wave`` is active the full buffer is re-touched every
        ``touch_interval`` seconds to fight OS reclamation.
    """
    PAGE = 4096
    total_bytes = max_ram_mb * 1048576
    min_bytes = int(total_bytes * min_ram_fraction)

    print(f"  [RAM] Pre-allocating {max_ram_mb} MB … ", end="", flush=True)
    try:
        buf = bytearray(total_bytes)
    except MemoryError:
        print(f"\n  [RAM] MemoryError: could not allocate {max_ram_mb} MB. "
              "Lower --ram-mb or free memory and retry.")
        return

    # Initial full touch – ensures every page is physically backed right away
    _touch_pages(buf, total_bytes)
    print("done", flush=True)

    start_time = time.time()
    end_time = start_time + duration

    # How often to re-touch pages to prevent the OS from reclaiming them.
    # Shorter than wave_period/20 so we track the wave smoothly.
    touch_interval = min(0.5, wave_period / 20.0) if use_wave else 1.0
    last_touch = start_time

    while time.time() < end_time:
        now = time.time()

        if use_wave:
            elapsed = now - start_time
            intensity = (math.sin((elapsed / wave_period) * 2 * math.pi) + 1) / 2
            # Active window: [min_bytes, total_bytes]
            active_bytes = min_bytes + int((total_bytes - min_bytes) * intensity)
            # Align to page boundary
            active_bytes = (active_bytes // PAGE) * PAGE
        else:
            active_bytes = total_bytes

        # Re-touch the active portion on every interval tick
        if now - last_touch >= touch_interval:
            _touch_pages(buf, active_bytes)
            last_touch = now

        time.sleep(touch_interval)

    # Explicit release – helps the GC on large allocations
    del buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU & RAM stress tool with sine-wave fluctuations."
    )
    parser.add_argument("--duration",    type=int,   default=30,
                        help="Total run time in seconds (default: 30)")
    parser.add_argument("--cores",       type=int,   default=multiprocessing.cpu_count(),
                        help="CPU worker processes (default: all cores)")
    parser.add_argument("--ram-mb",      type=int,   default=1024,
                        help="RAM to allocate in MB (default: 1024)")
    parser.add_argument("--wave-period", type=float, default=10.0,
                        help="Sine-wave cycle length in seconds (default: 10)")
    parser.add_argument("--no-wave",     action="store_true",
                        help="Hold RAM flat instead of fluctuating")
    parser.add_argument("--min-ram-pct", type=float, default=25.0,
                        help="Minimum RAM %% kept hot during wave troughs (default: 25)")

    args = parser.parse_args()

    if args.ram_mb <= 0:
        print("--ram-mb must be > 0")
        sys.exit(1)
    if not (0 <= args.min_ram_pct <= 100):
        print("--min-ram-pct must be between 0 and 100")
        sys.exit(1)

    use_wave = not args.no_wave
    mode_str = (f"sine wave (period {args.wave_period}s, "
                f"trough {args.min_ram_pct:.0f}% = "
                f"{int(args.ram_mb * args.min_ram_pct / 100)} MB)"
                if use_wave else "flat (no wave)")

    print(f"🔥 Starting Burner: {args.cores} core(s), {args.ram_mb} MB RAM")
    print(f"🌊 RAM mode : {mode_str}")
    print(f"⏱  Duration : {args.duration}s")

    processes = []

    for _ in range(args.cores):
        p = multiprocessing.Process(
            target=cpu_worker,
            args=(args.duration, args.wave_period),
        )
        p.start()
        processes.append(p)

    p_ram = multiprocessing.Process(
        target=ram_worker,
        args=(
            args.duration,
            args.ram_mb,
            args.wave_period,
            use_wave,
            args.min_ram_pct / 100.0,
        ),
    )
    p_ram.start()
    processes.append(p_ram)

    for p in processes:
        p.join()

    print("✅ Burner finished.")
    sys.exit(0)