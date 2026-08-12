# NTU RGB+D 120 Pilot — Full-Run Runbook

This runbook is for the point at which a GPU machine is available. The expensive stage is limited to encoder training and embedding extraction; frozen B0–B3 personalization experiments run on the extracted embeddings.

## 0. Environment

From repository root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

The NTU skeleton root must contain both the original `S001–S017` archive and the `S018–S032` extension, totaling 114,480 `.skeleton` files.

## 1. Dataset preflight

```bash
python scripts/validate_ntu120_manifest.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120
```

Do not continue unless this reports `PRECHECK PASSED`.

The inner subject split is intentionally stricter than a single train/validation split:

- `encoder_train`: representation training;
- `encoder_val`: encoder checkpoint selection only;
- `detector_calib`: one-class threshold calibration only;
- `retention_val`: held-out global-normal retention evaluation only;
- `deployment_test`: official cross-subject test subjects used for personalization simulation.

## 2. Train representation encoder

```bash
python scripts/train_encoder.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --epochs 20 \
  --batch-size 128 \
  --out outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt
```

The trainer now saves three provenance-linked artifacts:

- `simple_skeleton_encoder.pt`: checkpoint bundle with `state_dict` plus training metadata;
- `simple_skeleton_encoder.history.csv`: per-epoch train/validation history;
- `simple_skeleton_encoder.run.json`: Git commit, manifest SHA-256, checkpoint SHA-256, seed, hyperparameters, software/platform metadata, best epoch, and best validation accuracy.

Keep these files together. The encoder is a supervised global-normal action representation baseline, not the proposed continual-personalization method.

## 3. Session-0 sanity check

```bash
python scripts/run_session0.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --checkpoint outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt \
  --out outputs/ntu120_pilot_v0.1/session0/session0_scores.csv
```

Inspect candidate-personal-normal FPR, protected-anomaly recall/FNR, retained global-normal FPR, and score margin before proceeding.

## 4. Extract embeddings once

```bash
python scripts/extract_embeddings.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --checkpoint outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt \
  --out outputs/ntu120_pilot_v0.1/embeddings/embeddings.npz \
  --metadata-out outputs/ntu120_pilot_v0.1/embeddings/metadata.csv \
  --batch-size 256
```

This additionally writes `embeddings.provenance.json`, which records the Git commit, manifest hash, checkpoint hash and embedded checkpoint metadata, embedding/metadata hashes, sequence length, sample count, and embedding dimension. This makes it possible to verify that a baseline result was produced from the intended manifest and checkpoint rather than an accidentally stale artifact.

After this point, B0–B3 do not require skeleton parsing or GPU inference.

## 5. Inspect feasible feedback budgets

Before final baseline runs, inspect the script's printed `personal_samples_per_subject` distribution. NTU120 has only about nine A42 clips per subject on average, so large per-session budgets may be nominal rather than realized.

Recommended first scientific pass:

```bash
python scripts/run_personalization_baselines.py \
  --embeddings outputs/ntu120_pilot_v0.1/embeddings/embeddings.npz \
  --metadata outputs/ntu120_pilot_v0.1/embeddings/metadata.csv \
  --methods B0 B1 B2 B3 \
  --sessions 5 \
  --budgets 1 2 \
  --order-seeds 101 202 303 \
  --out outputs/ntu120_pilot_v0.1/baselines/frozen_baselines.csv
```

Only add larger saturation budgets if `feedback_used` is reported and the interpretation is explicit.

## 6. Aggregate across subjects

```bash
python scripts/summarize_personalization_results.py \
  --input outputs/ntu120_pilot_v0.1/baselines/frozen_baselines.csv \
  --out outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv \
  --subject-out outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_subject_summary.csv \
  --paired-out outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_paired_vs_b0.csv \
  --bootstrap-reps 5000 \
  --bootstrap-seed 1337
```

Subjects, not clips or seeds, are the inferential unit. Use subject-paired final-session deltas against B0 rather than inferring method differences from non-overlap of marginal confidence intervals.

## 7. Generate core figures

```bash
python scripts/plot_personalization_results.py \
  --summary outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv \
  --out-dir outputs/ntu120_pilot_v0.1/figures
```

Expected figures: `personal_fpr_by_session.png`, `safe_recall_by_session.png`, `global_fpr_by_session.png`, and `personal_gain_vs_safety_drop.png`.

## 8. Unit tests

```bash
pytest -q
```

All unit tests must pass before interpreting experimental results.

## 9. Interpretation limits

The current A42 personal-normal metric is a residual-pool evaluation over unconfirmed A42 clips, not a true future-only longitudinal estimate. NTU120 has too few A42 samples per subject to support a clean feedback-stream / future-test split with useful power. Therefore the pilot can establish a controlled mechanism signal, but not a real longitudinal deployment claim.

Likewise, the fixed `retention_val` cohort is independent of threshold calibration, but uncertainty in `global_fpr` primarily reflects variation across target personalization subjects and not resampling of the retention cohort itself. Treat this as a controlled retention probe rather than population-wide uncertainty.

## 10. Decision gate before B4/B5

Continue to neural/gradient adaptation only when frozen experiments establish a meaningful problem: nontrivial pre-adaptation personal-normal FPR; measurable `PersonalGain`; reproducible `SafetyDrop` or margin degradation; paired subject-level evidence; and failure of threshold-only calibration to explain away the entire problem.

If these conditions fail, stop or reformulate before spending additional GPU time.
