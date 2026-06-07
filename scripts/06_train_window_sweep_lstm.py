"""
Train and evaluate the word-event LSTM decoder with a 200 ms fixed lag.

This script reproduces the main model-selection experiment from the paper:
  - seen subjects: sub-01 ... sub-08
  - unseen subjects: sub-09, sub-10
  - fixed lag: 200 ms = 20 frames
  - context windows: 50, 100, 200 ms = 5, 10, 20 frames
  - target normalization using training data only
  - reconstruction error used as unseen detection score

Usage:
  python -u scripts/06_train_window_sweep_lstm.py \
      --feature-dir data/features_word \
      --result-dir data/results_word/window_sweep_lag200 \
      --model-dir models/lstm_word_ynorm_window_sweep_lag200
"""

import os

# In the HPC environment used for this paper, TensorFlow was most stable when
# imported before packages that may import JAX/XLA. Keep this near the top.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

print("Step 0: importing TensorFlow", flush=True)
import tensorflow as tf  # noqa: E402

print("TensorFlow:", tf.__version__, flush=True)
print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)

from tensorflow.keras import backend as K  # noqa: E402
from tensorflow.keras import layers, models  # noqa: E402
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

print("Step 1: importing standard packages", flush=True)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import SEEN_SUBJECTS, UNSEEN_SUBJECTS, FINAL_LAG_FRAMES  # noqa: E402


def set_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("Could not set memory growth:", exc, flush=True)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def mean_spectral_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    vals = [safe_corr(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return float(np.nanmean(vals))


def create_temporal_windows_with_lag(X: np.ndarray, y: np.ndarray, window_size: int, lag_frames: int):
    """Predict y[t] using X[t-lag-window : t-lag]."""
    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]
    start_t = window_size + lag_frames

    X_win, y_win = [], []
    for t in range(start_t, n):
        X_win.append(X[t - lag_frames - window_size:t - lag_frames])
        y_win.append(y[t])

    return np.asarray(X_win, dtype=np.float32), np.asarray(y_win, dtype=np.float32)


def normalize_windows(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    n_samples, t_steps, n_feat = X.shape
    X2d = X.reshape(-1, n_feat)
    X2d = (X2d - mean) / (std + 1e-8)
    return X2d.reshape(n_samples, t_steps, n_feat).astype(np.float32)


def normalize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean) / (std + 1e-8)).astype(np.float32)


def denormalize_y(y_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (y_norm * (std + 1e-8) + mean).astype(np.float32)


def load_subject(feature_dir: Path, subj: str, min_dim: int):
    X = np.load(feature_dir / f"{subj}_feat.npy").astype(np.float32)
    y = np.load(feature_dir / f"{subj}_spec.npy").astype(np.float32)
    n = min(len(X), len(y))
    return X[:n, :min_dim], y[:n]


def build_model(input_shape, output_dim: int) -> tf.keras.Model:
    """Final lightweight unidirectional LSTM decoder."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.0),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(output_dim, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


def compute_threshold_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    seen_scores = scores[y_true == 0]
    rows = []

    auroc = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))

    for p in [95, 90, 85, 80, 75, 70, 65, 60]:
        threshold = np.percentile(seen_scores, p)
        y_pred = (scores > threshold).astype(int)

        rows.append({
            "threshold_percentile": p,
            "threshold_value": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auroc": auroc,
            "auprc": auprc,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data/features_word")
    parser.add_argument("--result-dir", default="data/results_word/window_sweep_lag200")
    parser.add_argument("--model-dir", default="models/lstm_word_ynorm_window_sweep_lag200")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[5, 10, 20],
                        help="Context sizes in frames. 5=50 ms, 10=100 ms, 20=200 ms at 100 fps.")
    parser.add_argument("--lag-frames", type=int, default=FINAL_LAG_FRAMES)
    parser.add_argument("--max-train-samples", type=int, default=60000)
    parser.add_argument("--max-val-samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    set_gpu_memory_growth()

    feature_dir = Path(args.feature_dir)
    result_root = Path(args.result_dir)
    model_root = Path(args.model_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    all_subjects = SEEN_SUBJECTS + UNSEEN_SUBJECTS
    min_dim = min(np.load(feature_dir / f"{s}_feat.npy", mmap_mode="r").shape[1] for s in all_subjects)

    print("=" * 80, flush=True)
    print("Window sweep: word-event y-normalized LSTM with fixed lag", flush=True)
    print("=" * 80, flush=True)
    print("Feature directory:", feature_dir, flush=True)
    print("Seen subjects:", SEEN_SUBJECTS, flush=True)
    print("Unseen subjects:", UNSEEN_SUBJECTS, flush=True)
    print("Common feature dimension:", min_dim, flush=True)
    print("Lag frames:", args.lag_frames, flush=True)
    print("Window sizes:", args.window_sizes, flush=True)

    summary_rows = []

    for window_size in args.window_sizes:
        window_ms = int(window_size * 10)
        print("\n" + "=" * 80, flush=True)
        print(f"Running window experiment: {window_size} frames = {window_ms} ms", flush=True)
        print("=" * 80, flush=True)

        K.clear_session()
        gc.collect()

        result_dir = result_root / f"window_{window_ms}ms"
        model_dir = model_root / f"window_{window_ms}ms"
        result_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_seen_test_list, y_seen_test_list = [], []
        seen_test_subject_ids = []

        for subj in SEEN_SUBJECTS:
            X_sub, y_sub = load_subject(feature_dir, subj, min_dim)
            n = min(len(X_sub), len(y_sub))
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)

            splits = {
                "train": (X_sub[:n_train], y_sub[:n_train]),
                "val": (X_sub[n_train:n_train + n_val], y_sub[n_train:n_train + n_val]),
                "test": (X_sub[n_train + n_val:], y_sub[n_train + n_val:]),
            }

            X_train_win, y_train_win = create_temporal_windows_with_lag(*splits["train"], window_size, args.lag_frames)
            X_val_win, y_val_win = create_temporal_windows_with_lag(*splits["val"], window_size, args.lag_frames)
            X_test_win, y_test_win = create_temporal_windows_with_lag(*splits["test"], window_size, args.lag_frames)

            X_train_list.append(X_train_win); y_train_list.append(y_train_win)
            X_val_list.append(X_val_win); y_val_list.append(y_val_win)
            X_seen_test_list.append(X_test_win); y_seen_test_list.append(y_test_win)
            seen_test_subject_ids.extend([subj] * len(X_test_win))

            print(f"{subj}: train {X_train_win.shape}, val {X_val_win.shape}, test {X_test_win.shape}", flush=True)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        X_seen_test = np.concatenate(X_seen_test_list, axis=0)
        y_seen_test = np.concatenate(y_seen_test_list, axis=0)
        seen_test_subject_ids = np.asarray(seen_test_subject_ids)

        X_unseen_list, y_unseen_list = [], []
        unseen_subject_ids = []

        for subj in UNSEEN_SUBJECTS:
            X_sub, y_sub = load_subject(feature_dir, subj, min_dim)
            X_unseen_win, y_unseen_win = create_temporal_windows_with_lag(X_sub, y_sub, window_size, args.lag_frames)
            X_unseen_list.append(X_unseen_win)
            y_unseen_list.append(y_unseen_win)
            unseen_subject_ids.extend([subj] * len(X_unseen_win))
            print(f"{subj}: unseen {X_unseen_win.shape}", flush=True)

        X_unseen = np.concatenate(X_unseen_list, axis=0)
        y_unseen = np.concatenate(y_unseen_list, axis=0)
        unseen_subject_ids = np.asarray(unseen_subject_ids)

        n_features = X_train.shape[2]
        x_mean = X_train.reshape(-1, n_features).mean(axis=0)
        x_std = X_train.reshape(-1, n_features).std(axis=0)

        X_train_lstm = normalize_windows(X_train, x_mean, x_std)
        X_val_lstm = normalize_windows(X_val, x_mean, x_std)
        X_seen_test_lstm = normalize_windows(X_seen_test, x_mean, x_std)
        X_unseen_lstm = normalize_windows(X_unseen, x_mean, x_std)

        y_mean = y_train.mean(axis=0, keepdims=True)
        y_std = y_train.std(axis=0, keepdims=True)
        y_train_norm = normalize_y(y_train, y_mean, y_std)
        y_val_norm = normalize_y(y_val, y_mean, y_std)

        y_seen_test_orig = y_seen_test.astype(np.float32)
        y_unseen_orig = y_unseen.astype(np.float32)

        if len(X_train_lstm) > args.max_train_samples:
            idx = np.random.choice(len(X_train_lstm), args.max_train_samples, replace=False)
            X_train_lstm = X_train_lstm[idx]
            y_train_norm = y_train_norm[idx]

        if len(X_val_lstm) > args.max_val_samples:
            idx = np.random.choice(len(X_val_lstm), args.max_val_samples, replace=False)
            X_val_lstm = X_val_lstm[idx]
            y_val_norm = y_val_norm[idx]

        model = build_model((X_train_lstm.shape[1], X_train_lstm.shape[2]), y_train_norm.shape[1])
        best_model_path = model_dir / "best_model.keras"

        callbacks = [
            ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, mode="min", verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, mode="min", verbose=1),
            CSVLogger(result_dir / "training_log.csv"),
        ]

        t0 = time.time()
        history = model.fit(
            X_train_lstm,
            y_train_norm,
            validation_data=(X_val_lstm, y_val_norm),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )
        elapsed = time.time() - t0

        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        best_val_loss = float(np.min(history.history["val_loss"]))
        np.save(result_dir / "training_history.npy", history.history)

        pred_seen_norm = model.predict(X_seen_test_lstm, batch_size=512, verbose=1)
        pred_unseen_norm = model.predict(X_unseen_lstm, batch_size=512, verbose=1)

        pred_seen = denormalize_y(pred_seen_norm, y_mean, y_std)
        pred_unseen = denormalize_y(pred_unseen_norm, y_mean, y_std)

        seen_mse = float(mean_squared_error(y_seen_test_orig, pred_seen))
        unseen_mse = float(mean_squared_error(y_unseen_orig, pred_unseen))
        seen_corr = mean_spectral_corr(y_seen_test_orig, pred_seen)
        unseen_corr = mean_spectral_corr(y_unseen_orig, pred_unseen)

        seen_errors = np.mean((y_seen_test_orig - pred_seen) ** 2, axis=1)
        unseen_errors = np.mean((y_unseen_orig - pred_unseen) ** 2, axis=1)

        y_true = np.concatenate([np.zeros(len(seen_errors)), np.ones(len(unseen_errors))])
        scores = np.concatenate([seen_errors, unseen_errors])

        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))

        threshold_df = compute_threshold_table(y_true, scores)
        threshold_df.to_csv(result_dir / "threshold_sensitivity.csv", index=False)

        best_f1_row = threshold_df.iloc[threshold_df["f1"].idxmax()]

        # Save all information needed for event-level evaluation.
        np.savez_compressed(
            result_dir / "evaluation_outputs.npz",
            y_seen_true=y_seen_test_orig,
            y_seen_pred=pred_seen,
            y_unseen_true=y_unseen_orig,
            y_unseen_pred=pred_unseen,
            seen_errors=seen_errors,
            unseen_errors=unseen_errors,
            seen_test_subject_ids=seen_test_subject_ids,
            unseen_subject_ids=unseen_subject_ids,
            y_mean=y_mean,
            y_std=y_std,
            x_mean=x_mean,
            x_std=x_std,
            window_size=window_size,
            window_ms=window_ms,
            lag_frames=args.lag_frames,
            min_dim=min_dim,
        )

        metrics = {
            "window_size": window_size,
            "window_ms": window_ms,
            "lag_frames": args.lag_frames,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_seconds": elapsed,
            "seen_mse": seen_mse,
            "unseen_mse": unseen_mse,
            "seen_corr": seen_corr,
            "unseen_corr": unseen_corr,
            "corr_gap": seen_corr - unseen_corr,
            "auroc": auroc,
            "auprc": auprc,
            "best_threshold_percentile": float(best_f1_row["threshold_percentile"]),
            "best_f1": float(best_f1_row["f1"]),
            "best_precision": float(best_f1_row["precision"]),
            "best_recall": float(best_f1_row["recall"]),
        }

        (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary_rows.append(metrics)

        print("Window metrics:", json.dumps(metrics, indent=2), flush=True)

        del model
        K.clear_session()
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = result_root / "window_sweep_lag200_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nAll experiments completed.")
    print("Summary saved to:", summary_csv)
    print(summary_df)


if __name__ == "__main__":
    main()
