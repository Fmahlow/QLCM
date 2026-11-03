# Manuscript: "Does the Circuit Matter?"

Paper on the v2 results (hybrid-teacher consistency distillation and circuit
controls). The historical v1 manuscript stays in `v1/paper/` and is not updated.

- `main.tex`, `references.bib`: manuscript.
- `analysis/make_assets.py`: regenerates every table (`generated/*.tex`), figure
  (`figures/*.pdf`) and statistic (`generated/stats.json`) from `v2/reports/*.json`.
  Nothing is trained or evaluated.
- `main.pdf`: last compiled version.

Build, from this directory:

```sh
python analysis/make_assets.py      # needs numpy, scipy, matplotlib
tectonic -X compile main.tex
```

Open items before submission (GPU runs and manual checks) are listed in `TODO.md`.
