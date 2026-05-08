<div align="center">

# A Transparent Pipeline for Identifying Sexism in Social Media

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20090476.svg)](https://doi.org/10.5281/zenodo.20090476)
[![Applied Sciences 2024](https://img.shields.io/badge/Applied%20Sciences%202024-blue.svg)](https://doi.org/10.3390/app14198620)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)

*BERT + SHAP + ensemble learning for transparent sexism detection in social media.*

</div>

## Paper

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **Title**        | A Transparent Pipeline for Online Sexism Detection Based on the Combination of Explainable Artificial Intelligence, Feature Selection, and Ensemble Learning |
| **Authors**      | Hadi Mohammadi, Anastasia Giachanou, Robert A. Bagheri |
| **Affiliation**  | Utrecht University, The Netherlands |
| **Venue**        | Applied Sciences, 14(19), 8620 |
| **DOI (paper)**  | [10.3390/app14198620](https://doi.org/10.3390/app14198620) |
| **Code archive** | [10.5281/zenodo.20090476](https://doi.org/10.5281/zenodo.20090476) (this repository, snapshot v1.0-thesis) |

> This repository accompanies **Chapter 3** of the PhD thesis
> *Let Me Explain! Explainable NLP for Understanding Large Language Models* (Hadi Mohammadi, Utrecht University, 2026).

## Abstract

Sexism detection on social media is a high-stakes content-moderation task where transparency in model decisions is essential for moderators to verify and trust the system. This work develops a transparent pipeline that combines BERT-based classification with SHAP explanations and feature selection, allowing moderators to inspect the lexical and contextual cues that drove each prediction. Evaluated on benchmark datasets, the pipeline achieves competitive accuracy while exposing the words and phrases responsible for each decision.

## Citation

If you use this code or data, please cite **both** the paper and this code archive:

```bibtex
@article{mohammadi2024explainable,
  title         = {A Transparent Pipeline for Online Sexism Detection Based on the Combination of Explainable Artificial Intelligence, Feature Selection, and Ensemble Learning},
  author        = {Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Robert A.},
  year          = {2024},
  journal       = {Applied Sciences},
  doi           = {10.3390/app14198620}
}

@software{mohammadi_explainable_sexism_detection_2026,
  author    = {Mohammadi, Hadi and Giachanou, Anastasia and Bagheri, Robert A.},
  title     = {A Transparent Pipeline for Identifying Sexism in Social Media},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0-thesis},
  doi       = {10.5281/zenodo.20090476},
  url       = {https://doi.org/10.5281/zenodo.20090476}
}
```

---

## Key Contributions

- Transparent detection pipeline with interpretable predictions
- Analysis of linguistic patterns associated with sexist content
- Comparison of explanation methods (SHAP vs LIME) for text classification
- Human evaluation of model explanations

<div align="center">
<img src="figures/model.png" alt="Model Architecture" width="500"/>
<br><i>Ensemble model architecture combining ML and DL approaches</i>
</div>

## Repository Structure

```
Explainable-Sexism-Detection/
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── paper.pdf                       # Published Applied Sciences (2024) paper
├── data/
│   └── significant_tokens.csv      # SHAP token importances
└── figures/
    ├── methodology.{png,pdf}       # Pipeline overview
    ├── model.{png,pdf}             # Model architecture
    ├── backtranslation.pdf         # Back-translation augmentation diagram
    ├── histogram.pdf               # Token-importance histogram
    ├── importance.pdf              # Per-token importance plot
    ├── threshold.pdf               # Decision-threshold curve
    └── EDA1.png … EDA6.png         # Exploratory data analysis
```

## Quick Start

```bash
git clone https://github.com/mohammadi-hadi/Explainable-Sexism-Detection.git
cd Explainable-Sexism-Detection
open paper.pdf                      # macOS; or use any PDF viewer
```

The paper provides full methodology and references the figures in `figures/`. SHAP token importances aggregated over the test set are in `data/significant_tokens.csv`.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

- **Hadi Mohammadi** — Utrecht University
- Website: [mohammadi.cv](https://mohammadi.cv)
