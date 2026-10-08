# Paper draft: Guardian-APC

`guardian_apc.tex` is an IEEE two-column conference draft describing the
Guardian-APC validator and its evaluation. `references.bib` holds the citations.

Build it on Overleaf (upload this folder) or locally:

```text
pdflatex guardian_apc
bibtex guardian_apc
pdflatex guardian_apc
pdflatex guardian_apc
```

Every number in the paper comes from `modelforge_bench`. The snapshot used for
the draft is in `../research/results/results.json` (master seed 20260917,
96 models, budget 128). The seed-robustness and B = 32 rows come from:

```text
modelforge_bench --seed 1
modelforge_bench --seed 2
modelforge_bench --seed 3
modelforge_bench --budget 32
```

Fill in the author block before submitting, and re-run the benchmark on the
machine you report in the paper.
