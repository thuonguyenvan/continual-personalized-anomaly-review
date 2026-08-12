# B4/B5 Neural Adaptation Protocol — Design Before GPU

This document specifies the neural continual-adaptation baselines that should be implemented only if the frozen B0–B3 stage reveals a meaningful personalization–safety trade-off.

## Purpose

The aim is not to maximize anomaly-detection performance. B4/B5 are stress tests for the central hypothesis:

> Updating a learned normal representation using sparse human-confirmed personal normals may reduce personal false alarms but can distort the representation of global normality and safety-critical anomalies.

The experiment must therefore isolate *what changes because of representation adaptation* rather than confounding it with threshold calibration, extra labels, or test-set access.

## Data firewall

The following partitions retain their current roles:

- `encoder_train`: initial representation training only.
- `encoder_val`: initial checkpoint selection only.
- `detector_calib`: anomaly-threshold calibration only.
- `retention_val`: held-out global-normal retention evaluation only.
- `deployment_test`: target-subject continual-personalization simulation.

Neither A43 protected anomalies nor future/unconfirmed A42 clips may be used as adaptation data. `retention_val` and `detector_calib` must never be used in gradient updates.

## Shared initialization

For every target subject, seed, and feedback budget:

1. Load the same frozen initial encoder checkpoint used for B0–B3.
2. Fit the initial global normal prototype from `encoder_train` embeddings.
3. Calibrate the initial anomaly threshold on `detector_calib` global normals.
4. Use the same deterministic pseudo-session ordering and caregiver-feedback selection rule as the frozen baselines.
5. Clone the initial model independently for every method/subject/seed/budget run. No personalization state may leak between runs.

## B4 — Naive neural personalization

B4 deliberately contains no preservation mechanism.

### Trainable parameters

First pass: update only the projection portion of the encoder (`frame_proj`) and keep the GRU + LayerNorm frozen. This is preferred over full-model fine-tuning for the first neural stress test because it is cheaper and limits catastrophic drift while still allowing the embedding geometry to move.

If B4 shows no measurable representation effect, a secondary ablation may update the complete encoder. Full-encoder adaptation is not the default.

### Adaptation objective

For each feedback-confirmed personal-normal clip `x`, optimize its embedding toward the *fixed initial global-normal center* `c0`:

`L_personal = || f_theta(x) - c0 ||_2^2`

`c0` is computed once from the initial encoder on `encoder_train` global-normal samples and is not recomputed during B4. This avoids moving both representation and target simultaneously.

A small parameter-drift penalty relative to the initial checkpoint may be logged as an ablation, but it must not be part of B4 by default because B4 is intended to be the unconstrained/naive neural baseline.

### Per-session update

At session `t`:

1. Score arriving A42 clips with the current model.
2. Only clips that are currently false alarms are eligible for simulated caregiver confirmation.
3. Select up to the feedback budget using the same deterministic random rule as B0–B3.
4. Add selected clips to the cumulative confirmed-personal memory.
5. Fine-tune for a fixed number of very small update epochs/steps on confirmed personal normals only.
6. Refit/recompute the global prototype under the *updated encoder* using `encoder_train` global-normal data.
7. Recalibrate the threshold on `detector_calib` global normals under the updated encoder.
8. Evaluate personal residual-pool FPR, A43 recall/FNR/margin, and retention FPR.

Steps 6–7 are required for fairness: otherwise B4 would be penalized simply because the embedding coordinate system changed while the detector remained stale.

## B5 — Replay-preserving neural personalization

B5 is identical to B4 except every update minibatch contains confirmed personal normals plus a fixed replay sample from `encoder_train` global normals.

### Replay construction

Before deployment, sample a deterministic replay buffer from `encoder_train` only. The default buffer should be class-balanced across the 24 global-normal actions when feasible.

The buffer must be fixed before seeing a target subject. It cannot depend on A42/A43 or target-subject scores.

Recommended first-pass replay sizes:

- 64 global-normal clips;
- 256 global-normal clips as a sensitivity setting.

### Objective

`L = L_personal + lambda_replay * L_replay`

where:

- `L_personal = ||f_theta(x_personal) - c0||^2`
- `L_replay = ||f_theta(x_replay) - f_theta0(x_replay)||^2`

`theta0` is the frozen initial encoder. The replay term therefore performs representation distillation on trusted global normals rather than requiring action labels during personalization.

This is intentionally a simple preservation baseline, not a proposed safety method.

## Protected anomaly firewall

A43 samples are evaluation-only throughout B4/B5. They must not be:

- used in the loss;
- used for early stopping;
- used to choose learning rate, lambda, replay size, update epochs, or trainable layers;
- used to choose the best checkpoint within a personalization session.

Hyperparameters should be fixed from pilot engineering considerations or tuned only on development partitions that do not include protected anomalies. If a hyperparameter is changed after looking at A43 results, that run becomes exploratory and must not be reported as confirmatory evidence.

## Hyperparameters for first implementation

These values are engineering defaults, not yet validated scientific choices:

- optimizer: AdamW;
- personalization LR: `1e-4`;
- weight decay: `1e-4`;
- update epochs per session: `5`;
- batch size: all available confirmed samples up to a cap, because feedback sets are tiny;
- replay buffer size: `64` for B5;
- `lambda_replay = 1.0`;
- gradient clipping: `1.0`;
- trainable block: `frame_proj` only.

The first GPU run should treat these as fixed defaults. Broad tuning before establishing the failure mode risks optimizing the experiment toward the desired conclusion.

## Metrics added for neural adaptation

In addition to the frozen-stage metrics, record after every session:

- L2 parameter drift from the initial encoder;
- mean cosine shift of `retention_val` embeddings relative to session 0;
- mean cosine shift of A43 embeddings relative to session 0 (evaluation only);
- prototype drift from the initial global prototype;
- update wall-clock time;
- number of gradient steps;
- replay samples consumed.

These diagnostics help distinguish threshold effects from actual representation drift.

## Fairness requirements against B0–B3

B4/B5 must reuse:

- exactly the same target subjects;
- pseudo-session order seeds;
- feedback budgets;
- false-alarm-only caregiver simulation;
- session-level metric definitions;
- initial encoder checkpoint;
- detector-calibration quantile.

No method may receive a different set of confirmed labels merely because its earlier adaptation changed which clips become false alarms without recording that difference. Therefore `feedback_available`, `feedback_used`, and the exact selected path IDs must be logged per method/session.

This means the methods model different *interactive trajectories*. A secondary controlled ablation may replay a fixed oracle-confirmed feedback set across methods, but it must be reported separately because it answers a different question.

## Stop/go gate before implementation or full run

Do not spend GPU budget on B4/B5 unless B0–B3 show at least one of the following:

1. A42 personal-normal FPR is clearly nontrivial at session 0 and B1–B3 cannot trivially solve it;
2. boundary/prototype personalization changes the A43 margin or recall in a reproducible direction;
3. the frozen results leave a plausible reason to test whether representation adaptation creates a stronger safety–plasticity conflict.

If threshold-only calibration resolves nearly all false alarms without safety loss, B4/B5 become low priority and the research question should be reconsidered first.

## What B4/B5 cannot prove

Even if B4 exhibits a large safety drop and B5 reduces it, that does not by itself establish novelty or clinical safety. It only establishes that, under this controlled NTU proxy protocol, learned representation personalization can create a measurable preservation problem that replay partially mitigates.

A proposed method would still need a principled safety constraint, stronger baselines, external validation, and a more realistic elderly/longitudinal setting.
