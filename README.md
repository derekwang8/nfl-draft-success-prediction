# NFL Draft Success Prediction

## Research question

Among drafted players with recorded career Approximate Value, how accurately can draft position and player characteristics distinguish late-round selections with at least one recorded Pro Bowl selection from those with none?

## Current data reservation

| Stage | Records |
| --- | ---: |
| Original source | 8,435 |
| Excluded: blank `carav` | 1,382 |
| Eligible after filtering | 7,053 |
| Development | 5,642 |
| Reporting — final phase only | 1,411 |

Use [PREREQUISITE.md](PREREQUISITE.md) for the revised submission text and [nfl_dataset_description.ipynb](nfl_dataset_description.ipynb) for the description and development-only calculations. The notebook is safe to Run All: it does not read the reporting CSV or original full CSV.

```bash
python -m pip install -r requirements.txt
python scripts/verify_reporting_integrity.py
```

The original dataset is copied locally to `Data/raw/nfl_draft.csv` and excluded from Git. Do not use this full file for subsequent model development; it overlaps the frozen reporting set. Collaborators can obtain the original from [Kaggle](https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015) if needed for provenance. `scripts/prepare_data.py` documents the one-time preprocessing and split, and refuses to overwrite an existing reservation. Do not rerun it during modeling.

## Revised preprocessing and split

The instructor clarification relayed on November 20, 2026 is implemented in this order:

1. Start with the original 8,435-row CSV.
2. Remove the 1,382 rows whose `carav` is blank or whitespace. Keep all 1,067 zero-valued `carav` records. Do not impute missing career AV.
3. Calculate the required description for the 7,053 eligible records before reserving the new reporting partition.
4. Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, reserving the first fold. Stratify on `pb >= 1`; group nonempty player IDs together, using a unique year/pick key when an ID is absent. No seed/fold was selected based on model performance.
5. Save **5,642 development records** and **1,411 reporting records**. This is 80/20 to whole-record rounding. No additional year, round, or demographic filters were applied before this split.

Both files preserve the original 34 columns. Membership is disjoint, covers every eligible record, and has no shared known player IDs. No model has been fitted in this revision. The prior reporting set is superseded by this instructor-directed replacement. Its Git history remains available, and a local backup is kept in `.local_backups/` (excluded from Git).

## Classification plan and scope

The binary target is `success = 1` for `pb >= 1` and `0` otherwise. A zero means no Pro Bowl selection is recorded, not that the player was necessarily bad. Late rounds are 4–7. Compare logistic regression and a random forest against simple baselines using development-only validation. Use probability calibration and precision/recall-oriented metrics alongside any accuracy measures; select model settings and thresholds before final evaluation.

Use draft-time predictors such as overall pick, age, position, college, draft year, and optionally drafting team. `carav` is used only for **eligibility**, not as a model input. Exclude all career outcomes, awards, last playing year, games played, AV measures, and player identifiers from predictors. Fit all learned preprocessing inside development training folds; keep known repeat players grouped during cross-validation.

Conclusions apply to **players with recorded career AV**, not all drafted players. Eligibility uses information recorded after the draft, so this is a retrospective, conditional study. Missing career AV is not evidence of a zero-value career. Career outcomes are an older snapshot and follow-up differs by draft class; filtering missing career AV does not resolve this limitation. The random split assesses generalization within the historical sample, not performance on future draft classes.

## Reporting-set policy

`Data/reporting_set.csv` is frozen for the final phase. Do not preview records, calculate its descriptive statistics or outcomes, train models on it, or use it to choose features, preprocessing, hyperparameters, or classification thresholds. The notebook loads only development data and a saved pre-reservation sample summary.

`Data/split_manifest.json` records the source checksum, filter, split settings, software versions, counts, and file checksums. The integrity script reads bytes only, without parsing records. Checksums detect modifications but do not prevent someone opening the file; team members must honor the holdout policy.

Full-file and eligible-sample summaries preceded reservation. No model was fitted in this revision, and the replacement reporting set remains unused for analysis from its creation onward. This is a documented replacement of the earlier split, not a claim that the original records were never previously examined.

## Filtered dataset description

The NFL Draft 1985–2015 dataset, published by Ulrik Thyge Pedersen and downloaded from Kaggle [1], originally contains 8,435 draft-selection records and 34 variables describing draft position, player characteristics, and NFL performance. Before the training/reporting split, 1,382 records with missing career Approximate Value (`carav`) were excluded, leaving 7,053 eligible records; observed zero values were retained. Among 7,050 records with age available, the mean recorded age was 22.63 years (SD = 0.84), the median was 23 years (interquartile range: 22–23), and ages ranged from 20 to 29 years; age was missing for 3 records (0.04%). The most common position groups were defensive backs (19.3%), linebackers (13.6%), and wide receivers (12.2%). College information was available for 4,455 records (63.2%), covering 300 distinct recorded college labels. Sex, race, and ethnicity are not recorded.

Age statistics exclude the three missing ages. Position percentages and college availability use all 7,053 eligible records. Position and college are sample characteristics, not demographic identities. The 34 raw columns include 3 identifying fields and 31 other variables, including outcomes; they are not 34 model predictors.

## Reference

[1] U. T. Pedersen, “NFL Draft 1985–2015,” Kaggle, data set. Accessed: Nov. 17, 2026. [Online]. Available: https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015
