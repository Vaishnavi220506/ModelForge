"""Run one certified model on a simulated 8-bit microcontroller (terminal demo).

Author: Vaishnavi
Usage: python mcu_demo.py <modelforge --certify output dir> <test.csv> [--samples N] [--mcu atmega2560]

Builds the integer-only model_q.c (and the float32 model_f.c, if it fits) for an
AVR with avr-gcc, runs both in simavr (cycle-accurate, 16 MHz) and compares the
device's predictions with the same C file compiled for this machine.
Uses the helpers of mcu_bench.py.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcu_bench import avr_build, host_codes, host_predictions, read_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("data")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--mcu", default="atmega2560")
    args = parser.parse_args()
    for tool in ("avr-gcc", "avr-size", "simavr", "gcc"):
        if not shutil.which(tool):
            sys.exit(f"needs {tool} (apt: gcc-avr avr-libc simavr)")
    directory = Path(args.directory)
    inputs = int(re.search(r"#define MF_INPUTS (\d+)", (directory / "model_q.h").read_text()).group(1))
    rows = read_rows(args.data, inputs, args.samples)
    codes = host_codes(directory, args.data, inputs, len(rows))
    q = avr_build(directory, ["model_q.c"], "model_q.h", "__flash", len(rows), inputs, rows, True, args.mcu, codes)
    host_q = host_predictions(directory, args.data, ["model_q.c"], inputs, len(rows))
    try:
        f = avr_build(directory, ["model_f.c"], "model_f.h", "__flash", len(rows), inputs, rows, False, args.mcu)
    except subprocess.CalledProcessError:
        f = None

    print(f"  device: {args.mcu}, 16 MHz, simulated by simavr; {len(rows)} test inputs")
    print(f"  {'code':22}{'flash B':>9}{'RAM B':>7}{'kcycles/inference':>19}{'float lib':>11}")
    mean = sum(q["cycles"]) / len(q["cycles"])
    print(f"  {'ForgeCert integer':22}{q['flash']:>9}{q['ram']:>7}{mean / 1000:>19.1f}"
          f"{'yes' if q['links_float'] else 'no':>11}")
    if f:
        fmean = sum(f["cycles"]) / len(f["cycles"])
        print(f"  {'float32 (soft-float)':22}{f['flash']:>9}{f['ram']:>7}{fmean / 1000:>19.1f}{'yes':>11}")
        print(f"  integer code is {fmean / mean:.1f}x faster than float32 on this device")
    else:
        print(f"  {'float32 (soft-float)':22}{'does not fit':>9}")
    print(f"  predictions on the MCU: {q['predictions']}")
    print(f"  predictions on host:    {host_q}")
    if q["predictions"] != host_q:
        sys.exit("  MISMATCH between the device and the host build")
    print("  bit-exact: the device decides exactly like the host build")


if __name__ == "__main__":
    main()
