# ML quality scoring

Scores each AI-generated road annotation on how likely it is to need human correction, and ranks them so a reviewer with a limited budget checks the riskiest ones first.

## Contents

| File | What it is |
|---|---|
| `ML_quality_scoring.ipynb` | The full pipeline, from profiling the data to the trained model and export |
| `feature_scores.csv` | One row per raw AI annotation (5,592 rows) |
| `cell_scores.csv` | One row per 50 m grid cell (227 rows), ranked |
| `chart_models.png` | Ranking performance compared against random selection |

## Approach

The raw U-Net output is compared against the ground-truth layer, and the manual QA stage is used as the answer key: if nothing similar to a raw feature survived QA, a human removed or heavily changed it. That covers 67.8% of raw features.

Each annotation gets a set of signals: overlap with ground truth (IoU), whether it matches anything at all, how fragmented a line is, and how large its endpoint gaps are compared with the rest of its class. The `score` column in the source data is not used, since it is constant within each class and so works as a class threshold rather than a confidence.

A gradient-boosted model then learns how to weight these signals from the QA outcomes. Training is leave-one-tile-out, so every tile is scored by a model that never saw it.

## Results

Lift over random selection at a 5% review budget:

| Approach | Lift |
|---|---|
| Rule-based weights | 1.35x |
| Logistic regression | 1.40x |
| Gradient boosting | **1.44x** |
| Best possible at this base rate | 1.48x |

To check the scoring holds up for new classes, one class at a time was held out and scored by a model that had never seen it. That still gives 1.23x, so a class such as traffic islands would be ranked usefully from the start and improve once it has QA history.

## Output columns

`feature_scores.csv`: `row_id` links to the `features` table in the database. `baseline_score` is the rule-based score, `model_score` the trained model's probability of needing correction, and `quality_score` is `100 × (1 − model_score)`. The `s_` columns are the individual signals.

`cell_scores.csv`: `model_score` is the mean over features in the cell, and `rank` orders cells by it. `expected_bad` (model score × feature count) is for ranking when the review budget is counted in cells rather than annotations. `correction_rate` is included instead of a yes/no flag, since almost every cell (96.9%) contains at least one corrected feature.

## Running it

Runs in Google Colab. It expects the client data zip and `standardised_features.csv` in `MyDrive/capstone/`. Run all cells from the top; the CSVs and chart are written back to the same folder.

## Limitations

Everything comes from one scene, so the results show the approach works across tiles and classes within that scene but cannot show how it generalises to a different area. The correction label is also a proxy: it treats any raw feature without a close QA match (IoU below 0.3) as corrected, so it does not distinguish a deleted feature from one that was redrawn.
