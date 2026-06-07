"""
Event-level and subject-level unseen detection for the final 200 ms lag/window model.

This script reads:
    data/results_word/window_sweep_lag200/window_200ms/evaluation_outputs.npz

and produces:
    data/results_word/event_level_lag200_window200/*.csv
    data/results_word/event_level_lag200_window200/figures/*.png

Usage:
    python scripts/07_event_level_evaluation.py
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    RAW_DIR, FEATURE_DIR,
    SEEN_SUBJECTS, UNSEEN_SUBJECTS,
    TARGET_FRAME_RATE, IEEG_FS, AUDIO_FS,
    IEEG_WINDOW_SEC, AUDIO_N_FFT, AUDIO_HOP,
    TRIM_START_SEC, TRIM_END_SEC, MIN_EVENT_SEC,
    FINAL_LAG_FRAMES, FINAL_WINDOW_FRAMES,
)


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, scores))


def get_event_sample_bounds(row: pd.Series):
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


def estimate_event_frame_count(row: pd.Series) -> int:
    """Reproduce frame-count logic used during word-event extraction."""
    bounds = get_event_sample_bounds(row)
    if bounds is None:
        return 0

    i0, i1, a0, a1 = bounds
    ieeg_len = max(0, i1 - i0)
    audio_len = max(0, a1 - a0)

    ieeg_hop = int(round(IEEG_FS / TARGET_FRAME_RATE))
    ieeg_win = int(round(IEEG_WINDOW_SEC * IEEG_FS))

    if ieeg_len < ieeg_win + ieeg_hop:
        return 0
    if audio_len < AUDIO_N_FFT + AUDIO_HOP:
        return 0

    x_frames = 1 + (ieeg_len - ieeg_win) // ieeg_hop
    y_frames = 1 + (audio_len - AUDIO_N_FFT) // AUDIO_HOP
    return int(max(0, min(x_frames, y_frames)))


def build_frame_metadata_for_subject(subj: str, feature_dir: Path) -> pd.DataFrame:
    """Build one metadata row per extracted frame for a subject."""
    event_files = sorted(glob.glob(str(RAW_DIR / subj / "ieeg" / "*events.tsv")))
    if not event_files:
        raise FileNotFoundError(f"No events.tsv found for {subj}")

    expected_frames = len(np.load(feature_dir / f"{subj}_spec.npy", mmap_mode="r"))
    events = pd.read_csv(event_files[0], sep="\t")
    word_events = events[events["trial_type"] == "word"].reset_index(drop=True)

    rows = []
    kept_event_id = 0

    for _, row in word_events.iterrows():
        n_frames = estimate_event_frame_count(row)
        if n_frames <= 0:
            continue

        kept_event_id += 1
        word_label = str(row.get("value", ""))

        for frame_idx in range(n_frames):
            rows.append({
                "subject": subj,
                "event_id": f"{subj}_event_{kept_event_id:03d}",
                "word": word_label,
                "frame_in_event": frame_idx,
            })

    meta = pd.DataFrame(rows)

    if len(meta) != expected_frames:
        print(f"WARNING: metadata length mismatch for {subj}: meta={len(meta)}, expected={expected_frames}")
        n = min(len(meta), expected_frames)
        meta = meta.iloc[:n].reset_index(drop=True)

    return meta.reset_index(drop=True)


def create_target_metadata_for_split(meta: pd.DataFrame, split_name: str, lag_frames: int, window_size: int) -> pd.DataFrame:
    """Align metadata with y[t] targets used by lagged-window creation."""
    n = len(meta)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    if split_name == "seen_test":
        split_meta = meta.iloc[n_train + n_val:].reset_index(drop=True)
    elif split_name == "unseen":
        split_meta = meta.reset_index(drop=True)
    else:
        raise ValueError("split_name must be 'seen_test' or 'unseen'")

    start_t = window_size + lag_frames
    if len(split_meta) <= start_t:
        return split_meta.iloc[0:0].copy()
    return split_meta.iloc[start_t:].reset_index(drop=True)


def compute_threshold_table(y_true: np.ndarray, scores: np.ndarray, percentiles=(95, 90, 85, 80, 75, 70, 65, 60)) -> pd.DataFrame:
    seen_scores = scores[y_true == 0]
    auroc = safe_auroc(y_true, scores)
    auprc = safe_auprc(y_true, scores)

    rows = []
    for p in percentiles:
        threshold = np.percentile(seen_scores, p)
        y_pred = (scores > threshold).astype(int)
        rows.append({
            "Threshold percentile": p,
            "Threshold value": float(threshold),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
            "AUROC": auroc,
            "AUPRC": auprc,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--eval-path", default="data/results_word/window_sweep_lag200/window_200ms/evaluation_outputs.npz")
    parser.add_argument("--output-dir", default="data/results_word/event_level_lag200_window200")
    parser.add_argument("--window-size", type=int, default=FINAL_WINDOW_FRAMES)
    parser.add_argument("--lag-frames", type=int, default=FINAL_LAG_FRAMES)
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    eval_data = np.load(eval_path, allow_pickle=True)
    seen_errors = eval_data["seen_errors"]
    unseen_errors = eval_data["unseen_errors"]

    print("=" * 80)
    print("Event-level evaluation")
    print("=" * 80)
    print("Evaluation file:", eval_path)
    print("Seen errors:", seen_errors.shape)
    print("Unseen errors:", unseen_errors.shape)

    seen_meta, unseen_meta = [], []

    for subj in SEEN_SUBJECTS:
        meta = build_frame_metadata_for_subject(subj, feature_dir)
        target_meta = create_target_metadata_for_split(meta, "seen_test", args.lag_frames, args.window_size)
        target_meta["group"] = "Seen"
        target_meta["label"] = 0
        seen_meta.append(target_meta)
        print(subj, "seen-test metadata:", len(target_meta))

    for subj in UNSEEN_SUBJECTS:
        meta = build_frame_metadata_for_subject(subj, feature_dir)
        target_meta = create_target_metadata_for_split(meta, "unseen", args.lag_frames, args.window_size)
        target_meta["group"] = "Unseen"
        target_meta["label"] = 1
        unseen_meta.append(target_meta)
        print(subj, "unseen metadata:", len(target_meta))

    seen_meta = pd.concat(seen_meta, ignore_index=True)
    unseen_meta = pd.concat(unseen_meta, ignore_index=True)

    # Safety alignment with saved error vectors.
    n_seen = min(len(seen_meta), len(seen_errors))
    n_unseen = min(len(unseen_meta), len(unseen_errors))
    seen_meta = seen_meta.iloc[:n_seen].reset_index(drop=True)
    unseen_meta = unseen_meta.iloc[:n_unseen].reset_index(drop=True)
    seen_errors = seen_errors[:n_seen]
    unseen_errors = unseen_errors[:n_unseen]

    seen_meta["frame_error"] = seen_errors
    unseen_meta["frame_error"] = unseen_errors

    frame_df = pd.concat([seen_meta, unseen_meta], ignore_index=True)
    frame_df.to_csv(output_dir / "frame_level_errors.csv", index=False)

    frame_y = frame_df["label"].values
    frame_scores = frame_df["frame_error"].values
    frame_thresh = compute_threshold_table(frame_y, frame_scores)
    frame_thresh.to_csv(output_dir / "frame_level_threshold_sensitivity.csv", index=False)

    event_df = (
        frame_df
        .groupby(["subject", "event_id", "word", "group", "label"], as_index=False)
        .agg(
            event_error_mean=("frame_error", "mean"),
            event_error_median=("frame_error", "median"),
            event_error_std=("frame_error", "std"),
            n_frames=("frame_error", "count"),
        )
    )
    event_df["event_error_std"] = event_df["event_error_std"].fillna(0)
    event_df.to_csv(output_dir / "event_level_errors.csv", index=False)

    event_y = event_df["label"].values
    event_scores = event_df["event_error_mean"].values
    event_thresh = compute_threshold_table(event_y, event_scores)
    event_thresh.to_csv(output_dir / "event_level_threshold_sensitivity.csv", index=False)

    subject_df = (
        frame_df
        .groupby(["subject", "group", "label"], as_index=False)
        .agg(
            subject_error_mean=("frame_error", "mean"),
            subject_error_median=("frame_error", "median"),
            subject_error_std=("frame_error", "std"),
            n_frames=("frame_error", "count"),
            n_events=("event_id", "nunique"),
        )
    )
    subject_df.to_csv(output_dir / "subject_level_errors.csv", index=False)

    subject_y = subject_df["label"].values
    subject_scores = subject_df["subject_error_mean"].values

    best_frame = frame_thresh.iloc[frame_thresh["F1-score"].idxmax()]
    best_event = event_thresh.iloc[event_thresh["F1-score"].idxmax()]

    summary = pd.DataFrame([
        {
            "Level": "Frame",
            "AUROC": safe_auroc(frame_y, frame_scores),
            "AUPRC": safe_auprc(frame_y, frame_scores),
            "Best F1": best_frame["F1-score"],
            "Best threshold percentile": best_frame["Threshold percentile"],
            "Best precision": best_frame["Precision"],
            "Best recall": best_frame["Recall"],
            "N samples": len(frame_df),
        },
        {
            "Level": "Event",
            "AUROC": safe_auroc(event_y, event_scores),
            "AUPRC": safe_auprc(event_y, event_scores),
            "Best F1": best_event["F1-score"],
            "Best threshold percentile": best_event["Threshold percentile"],
            "Best precision": best_event["Precision"],
            "Best recall": best_event["Recall"],
            "N samples": len(event_df),
        },
        {
            "Level": "Subject",
            "AUROC": safe_auroc(subject_y, subject_scores),
            "AUPRC": safe_auprc(subject_y, subject_scores),
            "Best F1": np.nan,
            "Best threshold percentile": np.nan,
            "Best precision": np.nan,
            "Best recall": np.nan,
            "N samples": len(subject_df),
        },
    ])
    summary.to_csv(output_dir / "event_level_detection_summary.csv", index=False)

    print("\nDetection summary:")
    print(summary)

    # Figure used in the paper: event-level error histogram.
    seen_event = event_df[event_df["label"] == 0]["event_error_mean"].values
    unseen_event = event_df[event_df["label"] == 1]["event_error_mean"].values

    plt.figure(figsize=(8, 5))
    plt.hist(seen_event, bins=40, alpha=0.6, label="Seen events")
    plt.hist(unseen_event, bins=40, alpha=0.6, label="Unseen events")
    plt.xlabel("Mean reconstruction error per word event")
    plt.ylabel("Number of events")
    plt.title("Event-level Reconstruction Error Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "event_error_histogram_lag200_window200.png", dpi=300)
    plt.close()

    # Additional optional figures for internal analysis.
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(subject_df)), subject_df["subject_error_mean"])
    plt.xticks(np.arange(len(subject_df)), subject_df["subject"], rotation=45)
    plt.ylabel("Mean reconstruction error")
    plt.title("Subject-level Reconstruction Error")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "subject_error_lag200_window200.png", dpi=300)
    plt.close()

    print("\nSaved outputs to:", output_dir)
    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
