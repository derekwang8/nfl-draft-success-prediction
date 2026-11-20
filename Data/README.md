# Revised data partitions

The **blank-career-AV filter was applied before splitting**. Zero career AV values remain included.

- `development_set.csv`: 5,642 records for exploration, training, and internal validation.
- `reporting_set.csv`: 1,411 records, frozen until final evaluation. Do not preview or analyze it during development.
- `eligible_sample_summary.json`: descriptive statistics calculated on the 7,053 eligible records before reservation; not reporting-specific statistics.
- `split_manifest.json`: eligibility rule, counts, split settings, and SHA-256 checksums.
- `raw/nfl_draft.csv`: local, Git-ignored byte-for-byte copy of the original CSV; do not use it for modeling after reservation.

Both partition files preserve all 34 original columns, including the observed `carav` values. The only row exclusion before the split was blank `carav`. No imputation or model preprocessing has been fitted. This reservation supersedes the previous reporting set; the former file remains in Git history and a local ignored backup.
