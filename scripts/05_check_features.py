"""
Check extracted feature and spectrogram files before training.

Usage:
    python scripts/05_check_features.py --feature-dir data/features_word
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data/features_word")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)

    print("=" * 70)
    print("Feature data check")
    print("=" * 70)
    print("Feature directory:", feature_dir.resolve())

    if not feature_dir.exists():
        raise SystemExit(f"ERROR: feature directory does not exist: {feature_dir}")

    feat_files = sorted(feature_dir.glob("*_feat.npy"))
    spec_files = sorted(feature_dir.glob("*_spec.npy"))

    print("Number of feature files:", len(feat_files))
    print("Number of spectrogram files:", len(spec_files))

    if not feat_files:
        raise SystemExit("ERROR: no *_feat.npy files found.")

    print("\nDetected subjects:")
    for feat_path in feat_files:
        subj = feat_path.name.replace("_feat.npy", "")
        spec_path = feature_dir / f"{subj}_spec.npy"

        if not spec_path.exists():
            print(f"{subj}: missing spectrogram file")
            continue

        X = np.load(feat_path, mmap_mode="r")
        y = np.load(spec_path, mmap_mode="r")
        ok = "OK" if len(X) == len(y) else "LENGTH MISMATCH"
        print(f"{subj}: X={X.shape}, y={y.shape} [{ok}]")

    print("\nFeature check completed.")


if __name__ == "__main__":
    main()
