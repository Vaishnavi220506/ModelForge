# Paper draft: Forge, then Temper

Author: **Vaishnavi**

`temper.tex` is an IEEE two-column conference draft: *Forge, then Temper:
Falsify-or-Certify Assurance for Every Transformation from Trained Network to
Edge Binary*. It supersedes the earlier Guardian-APC+ draft (in git history);
Guardian-APC+ is now Section 3, the falsifier half of Temper.
`references.bib` holds the citations.

Build it on Overleaf (upload this folder) or locally:

```text
pdflatex temper
bibtex temper
pdflatex temper
pdflatex temper
```

Every number comes from `modelforge_bench` (master seed 20260917). Snapshots:

| Table | File in `../research/results/` |
|---|---|
| Guardian-APC+ numbers (Section 3) | `results.json` |
| Table I, decisiveness | `temper_results.json` |
| Table II, compression brackets | `compression_results.json` |
| Table III, binary audit | `binary_results.json` |

Before submitting: add department, institution and email, re-run the benchmark
on the machine you report, and pick a venue. Good fits: the tool or research
track of ASE / ICSE / FSE / ISSTA (testing + verification of ML systems),
EMSOFT / LCTES (embedded deployment), or the MLSys workshops.
