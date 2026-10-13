# Research proposal

The proposal is split into files the same way as the department's CIS thesis template, so each file can be copied into the template's Overleaf project as is.

| This folder | Template location | Contents |
|---|---|---|
| `thesis_metadata.tex` | `thesis_metadata.tex` | Titles (EN/HE), faculty, program, author, advisors, date. Copy the values into the template's fields. |
| `thesis_chapters.tex` | `thesis_chapters.tex` | Chapter order (`\thesischapter{EN}{label}{HE}{file}`), the same syntax as the template. |
| `Main_pages/abstract.tex` | `Main_pages/abstract.tex` | English abstract |
| `Main_pages/abstract_heb.tex` | `Main_pages/abstract_heb.tex` | Hebrew abstract |
| `content/*/*.tex` | `content/...` | Chapter text |
| `references.bib` | bibliography file | Cited with `\citep` / `\citet` (natbib) |

`main.tex` is a local stand-in for the template's own `main.tex`. It provides the cover pages, the `\thesischapter` macro, the bibliography, and the Hebrew back matter (abstract, table of contents, cover), so the proposal also compiles outside Overleaf. It also defines two notation macros used in the chapters, `\GT` and `\dist`. Add these to the template's preamble when moving the files.

## Build

```bash
xelatex -jobname=research_proposal main.tex
bibtex  research_proposal
xelatex -jobname=research_proposal main.tex
xelatex -jobname=research_proposal main.tex
```

The Hebrew font falls back from Arial Hebrew (macOS) to David CLM (Overleaf) and then to FreeSerif or Liberation Serif.

## Sources of the numbers

The preliminary results (Chapter 5) come from `results/cloud_runs/staged_run_2026-08-03_08-32-45/analysis/` and from `H2_results.xlsx`.
