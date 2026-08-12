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

## 2. Train representation encoder

Run a short smoke test first on any new machine, then the full run.

```bash
python scripts/train_encoder.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --epochs 20 \
  --batch-size 128 \
  --out outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt
```

Record the complete training log and best dev-val accuracy. The encoder is a supervised global-normal action representation baseline, not the proposed continual-personalization method.

## 3. Session-0 sanity check

```bash
python scripts/run_session0.py \
  --manifest data/ntu120_manifest.csv \
  --root /path/to/ntu120 \
  --checkpoint outputs/ntu120_pilot_v0.1/checkpoints/simple_skeleton_encoder.pt \
  --out outputs/ntu120_pilot_v0.1/session0/session0_scores.csv
```

Inspect candidate-personal-normal FPR, protected-anomaly recall/FNR, global-normal FPR, and score margin before proceeding.

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

After this point, B0–B3 do not require skeleton parsing or GPU inference.

## 5. Inspect feasible feedback budgets

Before final baseline runs, inspect the script's printed `personal_samples_per_subject` distribution. NTU120 has only about nine A42 clips per subject on average, so `K=5/10` per session may be nominal rather than realized.

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

Only add `K=5/10` as saturation settings if `feedback_used` is reported and the interpretation is explicit.

## 6. Aggregate across subjects

```bash
python scripts/summarize_personalization_results.py \
  --input outputs/ntu120_pilot_v0.1/baselines/frozen_baselines.csv \
  --out outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv \
  --subject-out outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_subject_summary.csv
```

The summarizer averages stochastic orderings within subject first, then reports subject-level mean/std/95% CI so subjects are the statistical unit rather than individual clips.

## 7. Generate core figures

```bash
python scripts/plot_personalization_results.py \
  --summary outputs/ntu120_pilot_v0.1/summaries/frozen_baselines_summary.csv \
  --out-dir outputs/ntu120_pilot_v0.1/figures
```

Expected figures:

- `personal_fpr_by_session.png`
- `safe_recall_by_session.png`
- `global_fpr_by_session.png`
- `personal_gain_vs_safety_drop.png`

## 8. Unit tests

```bash
pytest -q
```

All unit tests must pass before interpreting experimental results.

## 9. Decision gate before B4/B5

Do not implement a complex proposed method merely because the pipeline runs. Continue to neural/gradient adaptation only when the frozen experiments establish a meaningful problem:

- personal-normal FPR is nontrivial before adaptation;
- personalization yields measurable `PersonalGain`;
- at least some naive adaptation creates reproducible `SafetyDrop` or margin degradation;
- threshold-only calibration does not explain away the entire problem.

If these conditions fail, stop or reformulate the research question before spending additional GPU time.
