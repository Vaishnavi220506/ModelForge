# Paper drafts

Author: **Vaishnavi**

| File | Topic | Numbers come from |
| --- | --- | --- |
| `forgecert.tex` | **Main paper.** ForgeCert: proof-carrying integer code for neural networks via counterexample-guided precision tuning | `modelforge_certbench` → `../research/results/certbench.json`, summarised by `../research/summarize_certbench.py` |
| `guardian_apc.tex` | Earlier paper. Guardian-APC+, per-pass validation of graph rewrites (now one stage of the ForgeCert pipeline) | `modelforge_bench` → `../research/results/results.json` |

`references.bib` holds the citations for both papers.

Build on Overleaf (upload this folder) or locally:

```text
pdflatex forgecert
bibtex forgecert
pdflatex forgecert
pdflatex forgecert
```

To regenerate every ForgeCert number (about 20 minutes on one core):

```text
modelforge_certbench --out build/certbench
python ModelForge/research/summarize_certbench.py build/certbench/certbench.json
```

For Guardian-APC+ (seed-robustness rows: `--seed 1/2/3`, `--budget 32`):

```text
modelforge_bench
```

Before submitting, fill in the department, institution and email, and re-run
the benchmarks on the machine you report in the paper.
