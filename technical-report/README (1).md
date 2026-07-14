# Technical report template

Optional LaTeX template for FREUID Challenge 2026 reproducibility packages.

## Files

| File | Purpose |
| ---- | ------- |
| `freuid_technical_report.tex` | Main report skeleton |
| `references.bib` | Starter bibliography (challenge citation) |

## Build

```bash
latexmk -pdf freuid_technical_report.tex
```

## Customization

Edit the `\newcommand` lines near the top of the `.tex` file for team name, Kaggle team,
and contact email. Replace bracketed placeholders in each section.

Using this template is **not mandatory** — any readable PDF that covers method, data,
inference, and reproduction steps is acceptable.

Public download: [freuid2026.microblink.com/reproducibility.html](https://freuid2026.microblink.com/reproducibility.html)
