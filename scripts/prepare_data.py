"""Filter blank career AV first, then reserve a fixed grouped, stratified 20%."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

SOURCE_URL = "https://www.kaggle.com/datasets/ulrikthygepedersen/nfl-draft-1985-2015"
EXPECTED_SOURCE_SHA = "d8ff231d3bd521ead2463bfdd89c2c156f794e84905f2e873379f5f3f754c110"


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, root):
    data = root / "Data"
    names = ("development_set.csv", "reporting_set.csv", "split_manifest.json", "eligible_sample_summary.json")
    if any((data/name).exists() for name in names):
        raise FileExistsError("Reservation already exists. Do not recreate or overwrite the frozen split.")
    if checksum(source) != EXPECTED_SOURCE_SHA:
        raise ValueError("Source differs from the original CSV used for this project.")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    if raw.shape != (8435, 34):
        raise ValueError("Unexpected source dimensions.")

    # Eligibility only: remove blank/whitespace career AV, not zero values.
    keep = raw["carav"].str.strip().ne("")
    eligible = raw.loc[keep].reset_index(drop=True)
    career_av = pd.to_numeric(eligible["carav"], errors="raise")
    assert len(eligible) == 7053
    assert int((~keep).sum()) == 1382
    assert int(career_av.eq(0).sum()) == 1067
    assert not eligible.duplicated(["year", "pick"]).any()

    # Required sample description is calculated BEFORE reserving the new holdout.
    age = pd.to_numeric(eligible["age"].replace("", np.nan), errors="raise")
    observed_age = age.dropna()
    position = eligible["position_standard"].value_counts()
    college = eligible["college_univ"].str.strip().replace("", np.nan)
    age_bands = pd.cut(observed_age, bins=[19, 21, 23, 29], labels=["20-21", "22-23", "24-29"])
    summary = {
        "scope": "Filtered eligible sample, summarized before reporting-set reservation",
        "original_rows": len(raw), "excluded_blank_carav": int((~keep).sum()),
        "eligible_rows": len(eligible), "columns": eligible.shape[1],
        "year_min": int(eligible.year.astype(int).min()), "year_max": int(eligible.year.astype(int).max()),
        "zero_carav_retained": int(career_av.eq(0).sum()),
        "age_observed": int(age.notna().sum()), "age_missing": int(age.isna().sum()),
        "age_missing_pct": float(100*age.isna().mean()),
        "age_mean": float(observed_age.mean()), "age_sample_sd": float(observed_age.std(ddof=1)),
        "age_median": float(observed_age.median()), "age_q1": float(observed_age.quantile(.25)),
        "age_q3": float(observed_age.quantile(.75)), "age_min": float(observed_age.min()),
        "age_max": float(observed_age.max()),
        "age_bands": {str(k): {"n": int(v), "pct_of_observed_ages": float(100*v/len(observed_age))}
                      for k,v in age_bands.value_counts(sort=False).items()},
        "positions": {str(k): {"n":int(v), "pct_of_eligible":float(100*v/len(eligible))}
                      for k,v in position.items()},
        "college_observed": int(college.notna().sum()), "college_observed_pct":float(100*college.notna().mean()),
        "distinct_college_labels": int(college.nunique()),
        "demographic_limit": "Sex, race, ethnicity, and nationality are not recorded.",
    }

    y = pd.to_numeric(eligible["pb"], errors="raise").ge(1).astype(int)
    groups = pd.Series([
        "player:"+pid if pid else "selection:"+year+":"+pick
        for pid,year,pick in zip(eligible.player_id, eligible.year, eligible.pick)
    ])
    # Fixed first fold and seed, carried forward from the previous protocol.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    development_idx, reporting_idx = next(splitter.split(np.zeros((len(eligible),1)), y, groups))
    assert abs(len(reporting_idx) - .2*len(eligible)) <= 1
    assert set(development_idx).isdisjoint(reporting_idx)
    assert set(development_idx).union(reporting_idx) == set(range(len(eligible)))
    assert set(groups.iloc[development_idx]).isdisjoint(groups.iloc[reporting_idx])
    assert eligible.iloc[development_idx].carav.str.strip().ne("").all()
    assert eligible.iloc[reporting_idx].carav.str.strip().ne("").all()

    data.mkdir(parents=True, exist_ok=True)
    (data/"raw").mkdir(exist_ok=True)
    shutil.copyfile(source, data/"raw/nfl_draft.csv")
    eligible.iloc[np.sort(development_idx)].to_csv(data/names[0], index=False, lineterminator="\n")
    eligible.iloc[np.sort(reporting_idx)].to_csv(data/names[1], index=False, lineterminator="\n")
    (data/names[3]).write_text(json.dumps(summary, indent=2)+"\n")
    manifest = {
        "protocol_version": 2, "created_date": "2026-11-20", "source_url": SOURCE_URL,
        "source_sha256": checksum(source), "source_rows":len(raw), "source_columns":raw.shape[1],
        "eligibility": "Keep only records with nonblank carav; zero values are retained. No year/round filtering before split.",
        "excluded_rows": int((~keep).sum()), "eligible_rows":len(eligible),
        "zero_carav_retained": int(career_av.eq(0).sum()),
        "target": "pb >= 1 (at least one recorded Pro Bowl selection), not lifetime success",
        "algorithm": "StratifiedGroupKFold; first fold reserved", "n_splits":5,
        "shuffle":True, "random_state":42,
        "group_key": "Nonblank player_id; otherwise unique year/pick",
        "development_rows":len(development_idx), "reporting_rows":len(reporting_idx),
        "reporting_fraction":len(reporting_idx)/len(eligible), "known_player_overlap":0,
        "files": {name:{"sha256":checksum(data/name),"bytes":(data/name).stat().st_size}
                  for name in (names[0],names[1],names[3])},
        "versions":{"python":platform.python_version(),"numpy":np.__version__,
                    "pandas":pd.__version__,"scikit_learn":sklearn.__version__},
        "prior_exposure":"Original and eligible-sample descriptions were calculated before reservation. No model was fitted in this preparation. This replacement is frozen from creation onward.",
        "replacement_reason":"Instructor clarification relayed by the requester: exclude blank career AV before train/test split.",
        "reporting_policy":"No reporting-set EDA, model fitting, tuning, or outcome evaluation until final phase. Byte-level integrity checks and upload are permitted.",
    }
    (data/names[2]).write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"source_rows":len(raw),"removed":int((~keep).sum()),"eligible":len(eligible),
                      "development":len(development_idx),"reporting":len(reporting_idx),
                      "zero_carav_retained":int(career_av.eq(0).sum())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    prepare(args.source, args.output_root)
