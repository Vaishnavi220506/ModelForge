"""Run ModelForge's generated C on a cycle-accurate microcontroller simulator.

Author: Vaishnavi

For each model and code variant this script
  1. compiles the model with `modelforge --certify` (integer-only C99 + float32 C99),
  2. writes a harness that runs N test images through the model on an
     ATmega2560 (Arduino Mega: 8-bit AVR, 16 MHz, no FPU, 8 KB RAM), timing
     each inference with the 16-bit Timer1 (prescaler 1, overflow counted),
  3. builds it with avr-gcc -Os and runs it in simavr (cycle-accurate core),
  4. checks that every prediction made on the simulated MCU equals the host
     build of the same C file, and records flash, RAM and cycles per inference.
It also reports the code size for a Cortex-M0 (arm-none-eabi-gcc -Os).

Usage: python mcu_bench.py <modelforge build dir> [--samples N] [--out results.json]
Needs: avr-gcc, avr-libc, simavr, arm-none-eabi-gcc (apt: gcc-avr avr-libc simavr gcc-arm-none-eabi).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "review_2_core_implementation" / "models"

HARNESS = r"""
#include <avr/io.h>
#include <avr/interrupt.h>
#include <avr/sleep.h>
#include <stdint.h>
#include "%(header)s"

static volatile uint16_t overflows;
ISR(TIMER1_OVF_vect) { ++overflows; }
static void put(char c) { while (!(UCSR0A & (1 << UDRE0))) {} UDR0 = c; }
static void putu(uint32_t v) { char b[12]; int n = 0; do { b[n++] = '0' + v %% 10; v /= 10; } while (v); while (n) put(b[--n]); }
static uint32_t now(void) {
    uint16_t o, t; cli(); t = TCNT1; o = overflows; if ((TIFR1 & 1) && t < 32768) ++o; sei();
    return ((uint32_t)o << 16) | t;
}
static const %(rom)s float SAMPLES[%(n)d][%(inputs)d] = {%(samples)s};
static float x[%(inputs)d];
%(extra_decl)s
int main(void) {
    UCSR0B = (1 << TXEN0); UBRR0 = 0;
    TCCR1A = 0; TCCR1B = 1; TIMSK1 = 1; sei();
    for (int s = 0; s < %(n)d; ++s) {
        for (int i = 0; i < %(inputs)d; ++i) x[i] = SAMPLES[s][i];
        %(prepare)s
        uint32_t a = now();
        int best = %(call)s;
        uint32_t b = now();
        put('p'); putu((uint32_t)best); put(' '); put('c'); putu(b - a); put('\n');
    }
    cli(); sleep_cpu();
}
"""


def run(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def read_rows(path, inputs, count):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rows.append([float(v) for v in line.split(",")[:inputs]])
        if len(rows) == count:
            break
    return rows


def avr_build(directory, sources, header, rom, n, inputs, rows, integer):
    def literal(v):
        text = f"{v:.9g}"
        return (text if any(ch in text for ch in ".e") else text + ".0") + "f"
    samples = ",".join("{" + ",".join(literal(v) for v in row) + "}" for row in rows)
    if integer:
        extra = "static mf_input_t q[%d];" % inputs
        prepare = "mf_quantize_input(x, q);"
        call = "mf_infer_q(q, 0)"
    else:
        extra, prepare, call = "", "", "mf_predict_f(x, 0)"
    harness = directory / ("harness_q.c" if integer else "harness_f.c")
    harness.write_text(HARNESS % {"header": header, "rom": rom, "n": n, "inputs": inputs, "samples": samples,
                                  "extra_decl": extra, "prepare": prepare, "call": call})
    elf = directory / ("avr_q.elf" if integer else "avr_f.elf")
    # The model's tables go to flash via MF_ROM; the samples use the same qualifier.
    run(["avr-gcc", "-mmcu=atmega2560", "-Os", "-std=gnu99", "-ffunction-sections", "-fdata-sections",
         "-Wl,--gc-sections", f"-DMF_ROM={rom}", "-o", str(elf), str(harness)]
        + [str(directory / s) for s in sources])
    size = run(["avr-size", "-A", str(elf)]).stdout
    sections = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\.(\w+)\s+(\d+)", size, re.M)}
    samples_bytes = n * inputs * 4
    sim = run(["simavr", "-m", "atmega2560", "-f", "16000000", str(elf)], timeout=7200)
    out = sim.stdout + sim.stderr  # simavr echoes the UART on stderr
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    predictions = [int(m.group(1)) for m in re.finditer(r"p(\d+) c\d+", out)]
    cycles = [int(m.group(1)) for m in re.finditer(r"p\d+ c(\d+)", out)]
    return {"flash": sections.get("text", 0) + sections.get("data", 0) - samples_bytes,
            "ram": sections.get("data", 0) + sections.get("bss", 0),
            "cycles": cycles, "predictions": predictions}


def arm_size(directory, sources):
    if not shutil.which("arm-none-eabi-gcc"):
        return None
    obj = []
    for source in sources:
        target = directory / (Path(source).stem + ".m0.o")
        run(["arm-none-eabi-gcc", "-mcpu=cortex-m0", "-mthumb", "-Os", "-std=c99", "-c", "-o", str(target),
             str(directory / source)])
        obj.append(str(target))
    size = run(["arm-none-eabi-size", "-t"] + obj).stdout.splitlines()[-1].split()
    return {"text": int(size[0]), "data": int(size[1]), "bss": int(size[2])}


def host_predictions(directory, data, sources, inputs, count):
    exe = directory / "host_check"
    main = directory / "host_main.c"
    integer = "model_q.c" in sources
    main.write_text(
        '#include <stdio.h>\n#include <stdlib.h>\n#include "%s"\nstatic float x[%d];\nstatic char line[1<<20];\n'
        'int main(int c, char **v) { FILE *f = fopen(v[1], "r"); int n = 0;\n'
        ' while (n < %d && fgets(line, sizeof line, f)) { char *p = line; for (int i = 0; i < %d; ++i) { x[i] = strtof(p, &p); if (*p == \',\') ++p; }\n'
        '  printf("%%d\\n", %s); ++n; } return 0; }\n'
        % ("model_q.h" if integer else "model_f.h", inputs, count, inputs,
           "mf_predict(x, 0)" if integer else "mf_predict_f(x, 0)"))
    run(["gcc", "-std=c99", "-O2", "-o", str(exe), str(main)] + [str(directory / s) for s in sources])
    return [int(v) for v in run([str(exe), str(data)]).stdout.split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--out", default="mcu_results.json")
    args = parser.parse_args()
    build = Path(args.build)
    modelforge = build / "Release" / "modelforge"
    cases = [
        # model, data, calibration/domain, domain range, label, precision (None = tuned)
        ("image/mnist_cnn", "image/mnist_test.csv", "image/mnist_calib.csv", "0,1", 0.01),
        ("image/fashion_cnn", "image/fashion_test.csv", "image/fashion_calib.csv", "0,1", 0.01),
        ("image/mnist_mlp", "image/mnist_test.csv", "image/mnist_calib.csv", "0,1", 0.01),
        ("real/digits_mlp32x16", "real/digits_test.csv", "real/digits_train.csv", None, 0.1),
        ("real/wine_mlp32x16", "real/wine_test.csv", "real/wine_train.csv", None, 0.1),
    ]
    variants = [("forgecert", None), ("forgecert_latency", "latency"), ("uniform_int8", "int8")]
    results = []
    for model, data, domain, box, eps in cases:
        name = Path(model).name
        for variant, precision in variants:
            directory = build / "mcu" / f"{name}_{variant}"
            if directory.exists():
                shutil.rmtree(directory)
            command = [str(modelforge), str(MODELS / f"{model}.mforge"), "--certify", "--data", str(MODELS / data),
                       "--domain", str(MODELS / domain), "--eps", str(eps), "--max-regions", "60",
                       "--out", str(directory)]
            if box:
                command += ["--domain-range", box]
            if precision == "latency":
                command += ["--cost", "latency"]
            elif precision:
                command += ["--precision", precision]
            certify = subprocess.run(command, capture_output=True, text=True)
            log = certify.stdout
            summary = re.search(r"regions: (\d+) proved, (\d+) excluded.*?, (\d+) unproved", log)
            precision_text = re.search(r"Integer program: (.*)", log).group(1).strip()
            inputs = int(re.search(r"#define MF_INPUTS (\d+)", (directory / "model_q.h").read_text()).group(1))
            rows = read_rows(MODELS / data, inputs, args.samples)
            q_sources = ["model_q.c"]
            f_sources = ["model_f.c"]
            avr_q = avr_build(directory, q_sources, "model_q.h", "__flash", len(rows), inputs, rows, True)
            host_q = host_predictions(directory, MODELS / data, q_sources, inputs, len(rows))
            entry = {"model": name, "variant": variant, "precision": precision_text,
                     "proved": int(summary.group(1)), "excluded": int(summary.group(2)),
                     "unproved": int(summary.group(3)),
                     "avr_int": {k: avr_q[k] for k in ("flash", "ram")},
                     "avr_int_cycles_mean": sum(avr_q["cycles"]) / len(avr_q["cycles"]),
                     "estimated_cycles": float(re.search(r"estimated AVR cycles (\d+)", log).group(1)),
                     "avr_matches_host": avr_q["predictions"] == host_q,
                     "arm_m0_int": arm_size(directory, q_sources)}
            if variant == "forgecert":
                float_bytes = (directory / "model_f.c").stat().st_size
                avr_f = None
                for rom in ("__flash", "__memx"):  # __memx: tables beyond the first 64 KB of flash
                    try:
                        avr_f = avr_build(directory, f_sources, "model_f.h", rom, len(rows), inputs, rows, False)
                        break
                    except subprocess.CalledProcessError as failure:
                        reason = failure.stderr.strip().splitlines()[-1] if failure.stderr else "build failed"
                if avr_f is None:
                    # e.g. a float table larger than the 32 KB an AVR object may have
                    entry["avr_float"] = {"fits": False, "reason": reason}
                else:
                    host_f = host_predictions(directory, MODELS / data, f_sources, inputs, len(rows))
                    entry["avr_float"] = {k: avr_f[k] for k in ("flash", "ram")}
                    entry["avr_float"]["address_space"] = rom
                    entry["avr_float"]["fits"] = True
                    entry["avr_float_cycles_mean"] = sum(avr_f["cycles"]) / len(avr_f["cycles"])
                    entry["avr_float_matches_host"] = avr_f["predictions"] == host_f
                entry["arm_m0_float"] = arm_size(directory, f_sources)
                del float_bytes
            results.append(entry)
            print(json.dumps(entry))
            sys.stdout.flush()
    Path(args.out).write_text(json.dumps({"author": "Vaishnavi", "mcu": "ATmega2560 @ 16 MHz (simavr)",
                                          "samples": args.samples, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
