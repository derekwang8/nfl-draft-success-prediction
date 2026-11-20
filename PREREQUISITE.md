# Revised project prerequisite

## Research question

Among drafted players with recorded career Approximate Value, how accurately can draft position and player characteristics distinguish late-round selections with at least one recorded Pro Bowl selection from those with none?

## Dataset description

The NFL Draft 1985–2015 dataset, published by Ulrik Thyge Pedersen and downloaded from Kaggle [1], originally contains 8,435 draft-selection records and 34 variables describing draft position, player characteristics, and NFL performance. Before the training/reporting split, 1,382 records with missing career Approximate Value (`carav`) were excluded, leaving 7,053 eligible records; observed zero values were retained. Among 7,050 records with age available, the mean recorded age was 22.63 years (SD = 0.84), the median was 23 years (interquartile range: 22–23), and ages ranged from 20 to 29 years; age was missing for 3 records (0.04%). The most common position groups were defensive backs (19.3%), linebackers (13.6%), and wide receivers (12.2%). College information was available for 4,455 records (63.2%), covering 300 distinct recorded college labels. Sex, race, and ethnicity are not recorded.

## GitHub repository

| GitHub Repository | Link |
| --- | --- |
| NFL Draft Success Prediction | [https://github.com/derekwang8/nfl-draft-success-prediction](https://github.com/derekwang8/nfl-draft-success-prediction) |

The repository contains a `Data` folder with the revised reporting set of 1,411 records (approximately 20% of the eligible dataset), reserved for final evaluation. The remaining 5,642 records are available for development and internal validation. Final results and the report will be uploaded to this repository.

## Preprocessing update

Records with blank career Approximate Value were excluded **before** the 80/20 split. Zero-valued observations were retained. The split is stratified by recorded Pro Bowl status and grouped by known player ID. The replacement reporting set will not be used for model development or selection.

## Reference

[1] U. T. Pedersen, “NFL Draft 1985–2015,” Kaggle, data set. Accessed: Nov. 17, 2026. [Online]. Available: https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015
