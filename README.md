# SPECOM 2026 iEEG-to-Speech LSTM Decoder

This repository contains the reproducibility code for:

**A Lightweight LSTM-Based iEEG-to-Speech Decoder with Event-Level Unseen Neural Pattern Detection**

The code implements the paper pipeline:

1. Check the Python/GPU environment.
2. Check the raw iBIDS dataset structure.
3. Extract word-event iEEG features and 23-dimensional log-mel targets.
4. Train a lightweight unidirectional LSTM decoder with a 200 ms neural-to-speech lag.
5. Run a 50/100/200 ms context-window sweep.
6. Compute frame-level, event-level, and subject-level unseen neural pattern detection.

## Expected folder structure

Place the raw iBIDS dataset here:

```text
data/raw/SingleWordProductionDutch-iBIDS/
    sub-01/ieeg/...
    sub-02/ieeg/...
    ...
    sub-10/ieeg/...
```

The scripts assume they are run from the project root.

## Environment

The main packages are listed in `requirements.txt`.

On the HPC system used for the paper, the following module was used:

```bash
module purge
module load AI_env/v1
```

Then run:

```bash
python scripts/01_check_environment.py
```

## Step-by-step reproduction

### 1. Check raw data

```bash
python scripts/02_check_raw_data.py
```

### 2. Inspect one subject

```bash
python scripts/03_inspect_subject.py --subject sub-01
```

### 3. Extract word-event features

Final paper model uses **basic iEEG features**: mean, standard deviation, and log broadband power per channel.

```bash
python -u scripts/04_extract_word_events.py --feature-set basic
```

This creates:

```text
data/features_word/sub-XX_feat.npy
data/features_word/sub-XX_spec.npy
data/results_word/word_event_feature_extraction_summary_basic.csv
```

Optional extended bandpower features can be extracted with:

```bash
python -u scripts/04_extract_word_events.py --feature-set bandpower
```

### 4. Check extracted features

```bash
python scripts/05_check_features.py --feature-dir data/features_word
```

### 5. Train window-sweep LSTM models

```bash
python -u scripts/06_train_window_sweep_lstm.py
```

This trains the 50 ms, 100 ms, and 200 ms context models with fixed 200 ms lag.

Key output:

```text
data/results_word/window_sweep_lag200/window_sweep_lag200_summary.csv
data/results_word/window_sweep_lag200/window_200ms/evaluation_outputs.npz
models/lstm_word_ynorm_window_sweep_lag200/window_200ms/best_model.keras
```

### 6. Event-level unseen detection

```bash
python -u scripts/07_event_level_evaluation.py
```

Key output:

```text
data/results_word/event_level_lag200_window200/event_level_detection_summary.csv
data/results_word/event_level_lag200_window200/event_level_threshold_sensitivity.csv
data/results_word/event_level_lag200_window200/figures/event_error_histogram_lag200_window200.png
```

## SLURM jobs

Example jobs are provided in `jobs/`.

```bash
sed -i 's/\r$//' jobs/train_window_sweep_lstm.sh
sbatch jobs/train_window_sweep_lstm.sh
```

Monitor output:

```bash
tail -f logs/specom_lstm_<JOBID>.log
```

Run event-level evaluation:

```bash
sed -i 's/\r$//' jobs/event_level_evaluation.sh
sbatch jobs/event_level_evaluation.sh
```

## Notes

- The final paper configuration is:
  - word-event extraction
  - basic iEEG features
  - 200 ms neural-to-speech lag
  - 200 ms iEEG context window
  - lightweight unidirectional LSTM decoder
  - event-level reconstruction error for unseen detection
- Seen subjects: `sub-01` to `sub-08`
- Unseen subjects: `sub-09`, `sub-10`
- No unseen-subject data are used for training, validation, normalization, model selection, or threshold estimation.
