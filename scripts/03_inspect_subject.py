"""
Inspect one subject NWB file, events.tsv, channels.tsv, and timestamps.

Usage:
    python scripts/03_inspect_subject.py --subject sub-01
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from config import RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-01", help="Subject ID, e.g., sub-01")
    args = parser.parse_args()

    ieeg_dir = RAW_DIR / args.subject / "ieeg"
    nwb_files = sorted(glob.glob(str(ieeg_dir / "*.nwb")))
    event_files = sorted(glob.glob(str(ieeg_dir / "*events.tsv")))
    channel_files = sorted(glob.glob(str(ieeg_dir / "*channels.tsv")))

    if not nwb_files:
        raise SystemExit(f"No NWB file found for {args.subject}: {ieeg_dir}")

    print("=" * 80)
    print("Subject inspection")
    print("=" * 80)
    print("Subject:", args.subject)
    print("NWB:", nwb_files[0])

    if event_files:
        events = pd.read_csv(event_files[0], sep="\t")
        print("\nEVENTS")
        print(events.head())
        print("Shape:", events.shape)
        print("Trial types:", events["trial_type"].value_counts(dropna=False).to_dict())

    if channel_files:
        channels = pd.read_csv(channel_files[0], sep="\t")
        print("\nCHANNELS")
        print(channels.head())
        print("Shape:", channels.shape)
        if "type" in channels.columns:
            print("Channel types:", channels["type"].value_counts(dropna=False).to_dict())

    with NWBHDF5IO(nwb_files[0], "r") as io:
        nwbfile = io.read()

        print("\nNWB identifier:", nwbfile.identifier)
        print("Session description:", nwbfile.session_description)
        print("Acquisition keys:", list(nwbfile.acquisition.keys()))

        for key in ["iEEG", "Audio", "Stimulus"]:
            if key not in nwbfile.acquisition:
                continue

            obj = nwbfile.acquisition[key]
            print(f"\n{key}")
            print("  data shape:", getattr(obj.data, "shape", None))
            print("  rate:", getattr(obj, "rate", None))
            print("  starting_time:", getattr(obj, "starting_time", None))
            ts = getattr(obj, "timestamps", None)
            if ts is not None:
                print("  timestamps shape:", getattr(ts, "shape", None))
                first = np.array(ts[:5])
                last = np.array(ts[-5:])
                print("  first timestamps:", first)
                print("  last timestamps:", last)
                if len(ts) > 1000:
                    dt = np.diff(np.array(ts[:1000]))
                    print("  estimated fs:", 1.0 / np.mean(dt))


if __name__ == "__main__":
    main()
