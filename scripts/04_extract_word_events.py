"""
Extract word-event iEEG features and log-mel targets from the iBIDS NWB files.

This script supports:
  --feature-set basic      mean, std, log broadband power per channel (final paper model)
  --feature-set bandpower  basic + theta/alpha/beta/low-gamma/high-gamma log power

Outputs:
  data/features_word/sub-XX_feat.npy
  data/features_word/sub-XX_spec.npy
  data/results_word/word_event_feature_extraction_summary.csv

Usage:
  python scripts/04_extract_word_events.py --feature-set basic
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.signal import butter, sosfilt, sosfiltfilt

from config import (
    RAW_DIR, PROJECT_ROOT,
    TARGET_FRAME_RATE, IEEG_FS, AUDIO_FS,
    IEEG_WINDOW_SEC, N_MELS, AUDIO_N_FFT, AUDIO_HOP,
    TRIM_START_SEC, TRIM_END_SEC, MIN_EVENT_SEC, EPS,
)

BANDS = [
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 70.0),
    ("high_gamma", 70.0, 150.0),
]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sr: float, n_fft: int, n_mels: int, fmin: float = 80.0, fmax: float = 8000.0) -> np.ndarray:
    """Create a simple triangular mel filterbank without external audio libraries."""
    n_freqs = n_fft // 2 + 1
    fmax = min(fmax, sr / 2)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        center = min(center, n_freqs - 1)
        right = min(right, n_freqs - 1)

        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def audio_logmel(audio_seg: np.ndarray, sr: float = AUDIO_FS, n_fft: int = AUDIO_N_FFT,
                 hop: int = AUDIO_HOP, n_mels: int = N_MELS) -> np.ndarray | None:
    """Compute log-mel spectrogram frames from one word-event audio segment."""
    audio_seg = audio_seg.astype(np.float32)
    if len(audio_seg) < n_fft + hop:
        return None

    audio_seg = audio_seg / (np.max(np.abs(audio_seg)) + EPS)
    n_frames = 1 + (len(audio_seg) - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    mel_fb = build_mel_filterbank(sr, n_fft, n_mels)

    spec = np.zeros((n_frames, n_mels), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = audio_seg[start:start + n_fft] * window
        power = np.abs(np.fft.rfft(frame, n=n_fft)).astype(np.float32) ** 2
        spec[i] = np.log(np.dot(mel_fb, power) + EPS)
    return spec


def bandpass_filter(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Bandpass-filter [samples, channels] data. Fallback to causal filtering for short events."""
    nyq = fs / 2.0
    low_norm = max(low / nyq, 1e-5)
    high_norm = min(high / nyq, 0.999)
    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, data, axis=0).astype(np.float32)
    except Exception:
        return sosfilt(sos, data, axis=0).astype(np.float32)


def frame_ieeg_features(ieeg_seg: np.ndarray, feature_set: str) -> np.ndarray | None:
    """Create 100 fps iEEG features for one word-event segment."""
    hop = int(round(IEEG_FS / TARGET_FRAME_RATE))
    win = int(round(IEEG_WINDOW_SEC * IEEG_FS))

    if ieeg_seg.shape[0] < win + hop:
        return None

    ieeg_seg = ieeg_seg.astype(np.float32)
    ieeg_seg = ieeg_seg - np.mean(ieeg_seg, axis=0, keepdims=True)

    n_frames = 1 + (ieeg_seg.shape[0] - win) // hop
    n_channels = ieeg_seg.shape[1]

    band_signals = []
    if feature_set == "bandpower":
        for _, low, high in BANDS:
            band_signals.append(bandpass_filter(ieeg_seg, IEEG_FS, low, high))

    n_features_per_channel = 3 + (len(BANDS) if feature_set == "bandpower" else 0)
    features = np.zeros((n_frames, n_channels * n_features_per_channel), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop
        seg = ieeg_seg[start:start + win]

        per_channel = [
            np.mean(seg, axis=0),
            np.std(seg, axis=0),
            np.log(np.mean(seg ** 2, axis=0) + EPS),
        ]

        for filtered in band_signals:
            band_seg = filtered[start:start + win]
            per_channel.append(np.log(np.mean(band_seg ** 2, axis=0) + EPS))

        # Interleave features by channel: ch1_feats, ch2_feats, ...
        features[i] = np.stack(per_channel, axis=0).T.reshape(-1)

    return features


def get_event_sample_bounds(row: pd.Series):
    """Return iEEG/audio sample bounds after trimming event boundaries."""
    start_sec = float(row["onset"]) + TRIM_START_SEC
    end_sec = float(row["onset"]) + float(row["duration"]) - TRIM_END_SEC

    if end_sec <= start_sec or (end_sec - start_sec) < MIN_EVENT_SEC:
        return None

    return (
        int(round(start_sec * IEEG_FS)),
        int(round(end_sec * IEEG_FS)),
        int(round(start_sec * AUDIO_FS)),
        int(round(end_sec * AUDIO_FS)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=["basic", "bandpower"], default="basic")
    args = parser.parse_args()

    if args.feature_set == "basic":
        feature_dir = PROJECT_ROOT / "data" / "features_word"
        result_dir = PROJECT_ROOT / "data" / "results_word"
    else:
        feature_dir = PROJECT_ROOT / "data" / "features_word_bandpower"
        result_dir = PROJECT_ROOT / "data" / "results_word_bandpower"

    feature_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p.name for p in RAW_DIR.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    print("=" * 80)
    print(f"Word-event feature extraction ({args.feature_set})")
    print("=" * 80)
    print("Subjects:", subjects)
    print("Output feature directory:", feature_dir)

    summary_rows = []

    for subj in subjects:
        print("\n" + "=" * 80)
        print("Processing", subj)
        print("=" * 80)

        ieeg_dir = RAW_DIR / subj / "ieeg"
        nwb_files = sorted(glob.glob(str(ieeg_dir / "*.nwb")))
        event_files = sorted(glob.glob(str(ieeg_dir / "*events.tsv")))

        if not nwb_files or not event_files:
            print("Missing NWB or events file:", subj)
            continue

        events = pd.read_csv(event_files[0], sep="\t")
        word_events = events[events["trial_type"] == "word"].reset_index(drop=True)

        with NWBHDF5IO(nwb_files[0], "r") as io:
            nwbfile = io.read()
            ieeg = np.asarray(nwbfile.acquisition["iEEG"].data, dtype=np.float32)
            audio = np.asarray(nwbfile.acquisition["Audio"].data, dtype=np.float32)

        print("Total events:", len(events))
        print("Word events:", len(word_events))
        print("Raw iEEG:", ieeg.shape)
        print("Raw audio:", audio.shape)

        X_all, y_all = [], []
        kept_events, skipped_events = 0, 0

        for _, row in word_events.iterrows():
            bounds = get_event_sample_bounds(row)
            if bounds is None:
                skipped_events += 1
                continue

            i0, i1, a0, a1 = bounds
            i0, i1 = max(0, i0), min(len(ieeg), i1)
            a0, a1 = max(0, a0), min(len(audio), a1)

            if i1 <= i0 or a1 <= a0:
                skipped_events += 1
                continue

            X = frame_ieeg_features(ieeg[i0:i1], args.feature_set)
            y = audio_logmel(audio[a0:a1])

            if X is None or y is None:
                skipped_events += 1
                continue

            n = min(len(X), len(y))
            if n < 5:
                skipped_events += 1
                continue

            X_all.append(X[:n])
            y_all.append(y[:n])
            kept_events += 1

        if not X_all:
            print("No valid events for", subj)
            continue

        X_sub = np.concatenate(X_all, axis=0).astype(np.float32)
        y_sub = np.concatenate(y_all, axis=0).astype(np.float32)

        np.save(feature_dir / f"{subj}_feat.npy", X_sub)
        np.save(feature_dir / f"{subj}_spec.npy", y_sub)

        print("Kept word events:", kept_events)
        print("Skipped events:", skipped_events)
        print("Saved X:", X_sub.shape)
        print("Saved y:", y_sub.shape)

        summary_rows.append({
            "subject": subj,
            "word_events_total": len(word_events),
            "word_events_kept": kept_events,
            "word_events_skipped": skipped_events,
            "frames": len(X_sub),
            "feature_dim": X_sub.shape[1],
            "spec_dim": y_sub.shape[1],
            "channels": ieeg.shape[1],
            "feature_set": args.feature_set,
        })

    summary = pd.DataFrame(summary_rows)
    out_csv = result_dir / f"word_event_feature_extraction_summary_{args.feature_set}.csv"
    summary.to_csv(out_csv, index=False)

    print("\nExtraction completed.")
    print("Summary saved to:", out_csv)
    print(summary)


if __name__ == "__main__":
    main()
