"""
Check the raw SingleWordProductionDutch-iBIDS folder structure.

Expected structure:
    data/raw/SingleWordProductionDutch-iBIDS/sub-XX/ieeg/
        *_ieeg.nwb
        *_events.tsv
        *_channels.tsv
        *_electrodes.tsv
"""

import glob
from pathlib import Path
from config import RAW_DIR

print("=" * 80)
print("Raw iBIDS data check")
print("=" * 80)
print("Raw dataset directory:", RAW_DIR)

if not RAW_DIR.exists():
    raise SystemExit(f"ERROR: raw dataset directory does not exist: {RAW_DIR}")

subjects = sorted([p for p in RAW_DIR.iterdir() if p.is_dir() and p.name.startswith("sub-")])

print("\nDetected subjects:")
print([p.name for p in subjects])
print("Number of subjects:", len(subjects))

for subj_dir in subjects:
    ieeg_dir = subj_dir / "ieeg"
    print(f"\n{subj_dir.name}")

    if not ieeg_dir.exists():
        print("  ERROR: missing ieeg folder")
        continue

    patterns = {
        "NWB files": "*.nwb",
        "Events files": "*events.tsv",
        "Channels files": "*channels.tsv",
        "Electrodes files": "*electrodes.tsv",
    }

    for label, pat in patterns.items():
        files = sorted(glob.glob(str(ieeg_dir / pat)))
        print(f"  {label:16s}: {[Path(f).name for f in files]}")

print("\nRaw data check completed.")
