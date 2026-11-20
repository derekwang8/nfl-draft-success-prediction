"""Verify frozen bytes without parsing reporting records."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "Data/split_manifest.json").read_text())
expected = manifest["files"]["reporting_set.csv"]["sha256"]
actual = hashlib.sha256((root / "Data/reporting_set.csv").read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("FAIL: reporting_set.csv differs from its frozen checksum.")
print("PASS: reporting_set.csv is unchanged; no records were parsed or displayed.")
