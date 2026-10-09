# Thesis manuscript

Work-in-progress draft of the Master's thesis, written with the Sapienza `sapthesis` class.

## Compiling

On Overleaf: upload the `manuscript/` folder (or import the GitHub repository) and set `main.tex` as the main document. The compiler is pdfLaTeX, and the bibliography uses BibTeX with `references.bib`. `sapthesis` is part of TeX Live, so nothing extra is needed.

Locally, with a TeX Live installation:

    latexmk -pdf main.tex

## Layout

- `main.tex`: title page data, front matter, chapter order.
- `chapters/`: one file per chapter. `% ...` comments mark what each section will contain and where the numbers come from (`reports/`, `runs/`).
- `references.bib`: references. Entries marked `TODO` still need to be checked against the source.
