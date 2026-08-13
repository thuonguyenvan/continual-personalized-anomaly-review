# NTU RGB+D 120 Pilot — Full-Run Runbook

This runbook is for the point at which a GPU machine is available. The expensive stage is limited to encoder training and embedding extraction; frozen detector/personalization experiments run on extracted embeddings.

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

## 2. Simple representation baseline

```bash
python scripts/train_encoder.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --epochs 20 \
  --batch-size 128 \
  --out outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt
```

The trainer saves the checkpoint, per-epoch history, and run metadata. The encoder is a supervised global-normal action representation baseline, not the proposed continual-personalization method.

## 3. Simple-encoder Session-0 sanity check

```bash
python scripts/run_session0.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --checkpoint outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt \
  --out outputs/ntu120_pilot_v0.1/session0/session0_scores.csv
```

Inspect candidate-personal-normal FPR, protected-anomaly recall/FNR, retained global-normal FPR, and score margin before proceeding.

## 4. Extract simple-encoder embeddings

```bash
python scripts/extract_embeddings.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --checkpoint outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt \
  --out outputs/ntu120_pilot_v0.1/embeddings/embeddings.npz \
  --metadata-out outputs/ntu120_pilot_v0.1/embeddings/metadata.csv \
  --batch-size 256
```

This additionally writes `embeddings.provenance.json`.

## 5. Detector audit before personalization

Use the frozen embeddings to distinguish a weak one-class geometry from a weak representation:

```bash
python scripts/audit_session0_detectors.py \
  --embeddings outputs/ntu120_pilot_v0.1/embeddings/embeddings.npz \
  --metadata outputs/ntu120_pilot_v0.1/embeddings/metadata.csv \
  --out outputs/ntu120_pilot_v0.1/session0/detector_audit.csv
```

The detector variants are diagnostic only. Do not choose k/K/nu/gamma by looking at A43 protected-anomaly recall.

## 6. ST-GCN representation audit

If the simple encoder produces a nontrivial A42 false-positive signal but all reasonable one-class geometries still have weak A43 recall, train a stronger graph-based representation before interpreting safety-plasticity.

The in-repo ST-GCN encoder is a compact ST-GCN-style baseline using the NTU 25-joint graph, spatial graph convolution, temporal convolution, residual blocks, and global pooling. It is not claimed to reproduce a published implementation bit-for-bit.

Train it using exactly the same `encoder_train` / `encoder_val` subject partitions:

```bash
python scripts/train_stgcn_encoder.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 2 \
  --out outputs/ntu120_pilot_v0.1/checkpoints/stgcn_encoder.pt
```

Then extract embeddings:

```bash
python scripts/extract_stgcn_embeddings.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --checkpoint outputs/ntu120_pilot_v0.1/checkpoints/stgcn_encoder.pt \
  --out outputs/ntu120_pilot_v0.1/embeddings_stgcn/embeddings.npz \
  --metadata-out outputs/ntu120_pilot_v0.1/embeddings_stgcn/metadata.csv \
  --provenance-out outputs/ntu120_pilot_v0.1/embeddings_stgcn/embeddings.provenance.json \
  --batch-size 128 \
  --num-workers 2
```

Audit the same detector families without changing the protected test protocol:

```bash
python scripts/audit_session0_detectors.py \
  --embeddings outputs/ntu120_pilot_v0.1/embeddings_stgcn/embeddings.npz \
  --metadata outputs/ntu120_pilot_v0.1/embeddings_stgcn/metadata.csv \
  --out outputs/ntu120_pilot_v0.1/session0/detector_audit_stgcn.csv
```

Interpret this as a representation audit, not an A43-driven model-selection sweep. A43 remains evaluation-only. If multiple representation families are tried, report them all rather than silently cherry-picking the best protected-anomaly recall.

## 7. Personalization baselines

Only after a representation/detector combination has a meaningful Session-0 safety signal should the continual-personalization baselines be interpreted scientifically.

```bash
python scripts/run_personalization_baselines.py \
  --embeddings /path/to/chosen/frozen/embeddings.npz \
  --metadata /path/to/chosen/frozen/metadata.csv \
  --methods B0 B1 B2 B3 \
  --sessions 5 \
  --budgets 1 2 \
  --order-seeds 101 202 303 \
  --out outputs/ntu120_pilot_v0.1/baselines/frozen_baselines.csv
```

Before final runs, inspect `personal_samples_per_subject` and `feedback_used`. Large per-session feedback budgets are often nominal because NTU120 has few A42 clips per subject.

## 8. Aggregate across subjects

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

## 9. Generate core figures

```bash
python scripts/plot_personalization_results.py \
  --summary outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv \
  --out-dir outputs/ntu120_pilot_v0.1/figures
```

Expected figures: `personal_fpr_by_session.png`, `safe_recall_by_session.png`, `global_fpr_by_session.png`, and `personal_gain_vs_safety_drop.png`.

## 10. Unit tests

```bash
pytest -q
```

All unit tests must pass before interpreting experimental results.

## 11. Interpretation limits

The current A42 personal-normal metric is a residual-pool evaluation over unconfirmed A42 clips, not a true future-only longitudinal estimate. NTU120 has too few A42 samples per subject to support a clean feedback-stream / future-test split with useful power. Therefore the pilot can establish a controlled mechanism signal, but not a real longitudinal deployment claim.

Likewise, the fixed `retention_val` cohort is independent of threshold calibration, but uncertainty in `global_fpr` primarily reflects variation across target personalization subjects and not resampling of the retention cohort itself. Treat this as a controlled retention probe rather than population-wide uncertainty.

## 12. Decision gate before B4/B5

Continue to neural/gradient adaptation only when frozen experiments establish a meaningful problem: nontrivial pre-adaptation personal-normal FPR; meaningful pre-adaptation protected-anomaly sensitivity; measurable `PersonalGain`; reproducible `SafetyDrop` or margin degradation; paired subject-level evidence; and failure of threshold-only calibration to explain away the entire problem.

If these conditions fail, stop or reformulate before spending additional GPU time.
