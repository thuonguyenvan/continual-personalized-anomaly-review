# Implementation Audit — 2026-08-12

This note records issues found while hardening the NTU RGB+D 120 pilot before full GPU runs. It is intentionally skeptical: passing code is not equivalent to validating the research hypothesis.

## Resolved engineering mismatches

1. `manifest_dataset.py` imported a nonexistent `preprocess_skeleton_sequence`; the actual preprocessing entry point is `preprocess_skeleton`.
2. Manifest schema and loader schema disagreed (`subject` vs `subject_id`, `inner_split` vs `split`, etc.).
3. `run_session0.py` imported nonexistent `PrototypeDistanceDetector`; the implemented class is `PrototypeDetector`.
4. Session-0 originally allowed a random encoder. This was removed; a trained checkpoint is now required.
5. `PYTHONPATH=.` is currently required when invoking scripts directly. Proper packaging remains a later engineering cleanup item.

## Data integrity status

The local preflight reported:

- 114,480 rows;
- 106 subjects;
- 120 actions;
- 0 duplicate paths;
- 63,360 outer-train / 51,120 outer-test samples;
- 51,840 dev-train / 11,520 dev-val / 51,120 deployment-test samples;
- 22,836 global-normal samples;
- 948 candidate-personal-normal (A42) samples;
- 948 protected-anomaly (A43) samples.

This validates indexing/split integrity only. It does not validate the scientific semantics of the action-role assignment.

## Major protocol issue: feedback budget granularity

There are only 948 A42 samples across 106 subjects, i.e. roughly nine candidate-personal-normal clips per subject on average. With five simulated sessions, a nominal feedback budget of `K=5` or `K=10` per session can easily exceed the number of available false alarms/samples in a session.

Therefore:

- every result must report `feedback_available`, `feedback_used`, and `confirmed_cumulative`;
- nominal `K` must never be interpreted as actual human annotation count;
- the first NTU-specific run should inspect per-subject counts before deciding final budgets;
- smaller budgets such as `K=1` and possibly `K=2` are likely more interpretable on NTU120;
- `K=5/10` may remain only as saturation/sensitivity settings if the realized counts are explicitly reported.

Do not silently retain `K=1,5,10` in the final paper without this check.

## Major scientific caveats

### A42 is an experimental semantic proxy

`A42 = staggering` is treated as a candidate personal normal only to simulate a benign behavior that could be accepted after trusted feedback. NTU does not establish that staggering is clinically safe or normal for an elderly person. The paper must clearly distinguish the dataset label from the experimental semantic assignment.

### A43 is a protected anomaly proxy

`A43 = falling down` is used as the protected safety-critical anomaly and is never adaptation data. This is defensible as a controlled mechanism test, but NTU contains staged actions rather than real elderly falls.

### Sessions are pseudo-longitudinal

NTU120 is not a longitudinal deployment dataset. Session orderings are deterministic pseudo-sessions generated from available subject clips. Conclusions must be limited to a simulated continual-personalization protocol, not real temporal drift.

### Supervised normal-action representation

The current `SimpleSkeletonEncoder` is trained with action-class supervision using only global-normal action classes. The anomaly detector itself remains one-class, but the complete representation-learning pipeline is not an end-to-end one-class training procedure. This is acceptable for a mechanism pilot, but should not be described as a pure one-class neural training pipeline.

## Frozen-baseline design decisions

- B0: no adaptation.
- B1: threshold-only adaptation; population threshold is retained as a lower bound and personal-normal feedback can raise it. This deliberately tests whether a trivial calibration solution removes the need for continual representation/boundary adaptation.
- B2: global + personal dual prototype; threshold recalibrated on retained global-normal validation embeddings.
- B3: One-Class SVM refit with global-normal embeddings plus confirmed personal normals; global-normal validation data recalibrate the threshold.

Only model-raised A42 false alarms are eligible for simulated caregiver confirmation. Confirmed samples are removed from subsequent personal-normal evaluation to avoid evaluating directly on the feedback items.

## Go / stop interpretation

The current hypothesis is supported only if, across subjects and seeds:

1. pre-personalization personal-normal FPR is materially nontrivial;
2. at least one adaptation baseline reduces that FPR;
3. adaptation measurably reduces A43 recall or anomaly margin, or otherwise creates a reproducible safety-plasticity trade-off;
4. the effect is not explained entirely by one pathological baseline or by nominal feedback budgets that are never realized.

If threshold-only adaptation solves the problem without safety degradation, the proposed continual-learning contribution should be weakened or reformulated. If A42 is rarely flagged, the NTU mechanism test does not establish the motivating failure mode.
