"""
Project-wide configuration for the SPECOM iEEG-to-speech experiments.

Run scripts from the project root, for example:
    python scripts/02_check_raw_data.py
"""

from pathlib import Path

PROJECT_ROOT = Path.cwd()

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "SingleWordProductionDutch-iBIDS"

# Final paper pipeline uses word-event basic features.
FEATURE_DIR = PROJECT_ROOT / "data" / "features_word"
RESULT_DIR = PROJECT_ROOT / "data" / "results_word"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

SEEN_SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04",
    "sub-05", "sub-06", "sub-07", "sub-08",
]
UNSEEN_SUBJECTS = ["sub-09", "sub-10"]

# Dataset/signal parameters used in the paper.
TARGET_FRAME_RATE = 100.0
IEEG_FS = 1024.0
AUDIO_FS = 48000.0

IEEG_WINDOW_SEC = 0.05
N_MELS = 23
AUDIO_N_FFT = 1024
AUDIO_HOP = int(AUDIO_FS / TARGET_FRAME_RATE)

TRIM_START_SEC = 0.10
TRIM_END_SEC = 0.10
MIN_EVENT_SEC = 0.40

EPS = 1e-8

# Final model from the paper.
FINAL_LAG_FRAMES = 20       # 200 ms at 100 fps
FINAL_WINDOW_FRAMES = 20    # 200 ms at 100 fps

def ensure_dirs() -> None:
    for p in [FEATURE_DIR, RESULT_DIR, MODEL_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)
