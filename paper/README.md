# QLCM Manuscript

Primary file: `main.tex`.

Current status:

- Method and experimental protocol drafted.
- Results section intentionally contains placeholders because MNIST and
  Fashion-MNIST experiments have not yet been run.
- Bibliography is a first-pass source set and still needs full metadata/DOI
  cleanup before submission.

Build command:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
