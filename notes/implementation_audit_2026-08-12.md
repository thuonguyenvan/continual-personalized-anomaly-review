# Implementation Audit — 2026-08-12

This note records issues found while hardening the NTU RGB+D 120 pilot before full GPU runs. Passing code is not equivalent to validating the research hypothesis.

## Resolved engineering mismatches

1. `manifest_dataset.py` imported a nonexistent `preprocess_skeleton_sequence`; the actual preprocessing entry point is `preprocess_skeleton`.
2. Manifest schema and loader schema disagreed (`subject` vs `subject_id`, `inner_split` vs `split`, etc.).
3. `run_session0.py` imported nonexistent `PrototypeDistanceDetector`; the implemented class is `PrototypeDetector`.
4. Session-0 originally allowed a random encoder. This was removed; a trained checkpoint is now required.
5. `PYTHONPATH=.` is currently required when invoking scripts directly. Proper packaging remains a later engineering cleanup item.

## Resolved methodological leakage risk: reused validation subjects

The first implementation reused the same `dev_val` subjects for three distinct purposes:

- selecting the encoder checkpoint;
- calibrating the anomaly threshold;
- measuring global-normal retention FPR.

This is not deployment-test leakage, but it makes retention performance optimistic by evaluating on the same subjects used for threshold calibration and also couples representation model selection to detector calibration.

The manifest protocol has therefore been changed. The 53 official NTU120 cross-subject training subjects are now deterministically partitioned by subject into:

- `encoder_train`: 37 subjects;
- `encoder_val`: 6 subjects, used only for encoder checkpoint selection;
- `detector_calib`: 5 subjects, used only for anomaly-threshold calibration;
- `retention_val`: 5 subjects, used only for global-normal retention evaluation.

The 53 official cross-subject test subjects remain `deployment_test` and are never used for representation training, checkpoint selection, or initial threshold calibration.

After pulling this change, the old manifest must be regenerated before any full experiment. Results produced with the old `dev_train/dev_val` manifest must not be mixed with results from the hardened protocol.

## Data integrity status

The earlier local preflight reported:

- 114,480 rows;
- 106 subjects;
- 120 actions;
- 0 duplicate paths;
- 63,360 outer-train / 51,120 outer-test samples;
- 22,836 global-normal samples;
- 948 candidate-personal-normal (A42) samples;
- 948 protected-anomaly (A43) samples.

The exact inner-split sample counts must be re-reported after regenerating the hardened manifest. Data-integrity checks validate indexing/split integrity only; they do not validate the scientific semantics of action-role assignment.

## Major protocol issue: feedback budget granularity

There are only 948 A42 samples across 106 subjects, i.e. roughly nine candidate-personal-normal clips per subject on average. With five simulated sessions, nominal budgets such as `K=5` or `K=10` can exceed available false alarms in a session.

Therefore:

- every result must report `feedback_available`, `feedback_used`, and `confirmed_cumulative`;
- nominal `K` must never be interpreted as actual human annotation count;
- the default frozen-baseline budgets are now `K=1,2,5`;
- `K=5` is best interpreted as a saturation/sensitivity setting unless realized feedback counts support it.

## Major scientific caveats

### A42 is an experimental semantic proxy

`A42 = staggering` is treated as a candidate personal normal only to simulate a benign behavior that could be accepted after trusted feedback. NTU does not establish that staggering is clinically safe or normal for an elderly person. The paper must clearly distinguish the dataset label from the experimental semantic assignment.

### A43 is a protected anomaly proxy

`A43 = falling down` is used as the protected safety-critical anomaly and is never adaptation data. This is defensible as a controlled mechanism test, but NTU contains staged actions rather than real elderly falls.

### Sessions are pseudo-longitudinal

NTU120 is not a longitudinal deployment dataset. Session orderings are deterministic pseudo-sessions generated from available subject clips. Conclusions must be limited to a simulated continual-personalization protocol, not real temporal drift.

### Personal-normal evaluation is pool-based

After feedback, confirmed samples are removed from the personal-normal evaluation pool, but the remaining pool can contain both already-arrived unconfirmed clips and future pseudo-session clips. Therefore `personal_fpr` is currently a residual-pool metric, not a strict future-only longitudinal generalization metric. The final paper must either state this explicitly or add a separate future-only/held-out evaluation design if data volume permits.

### Supervised normal-action representation

The current `SimpleSkeletonEncoder` is trained with action-class supervision using only global-normal action classes. The anomaly detector itself remains one-class, but the complete representation-learning pipeline is not an end-to-end one-class training procedure. This is acceptable for a mechanism pilot, but should not be described as a pure one-class neural training pipeline.

## Frozen-baseline design decisions

- B0: no adaptation.
- B1: threshold-only adaptation; the population-calibrated threshold is retained as a lower bound and confirmed personal-normal feedback can raise it.
- B2: global + personal dual prototype; threshold is recalibrated only on `detector_calib`, while global retention is measured only on `retention_val`.
- B3: One-Class SVM refit with `encoder_train` global-normal embeddings plus confirmed personal normals; threshold is recalibrated on `detector_calib`, while global retention is measured on `retention_val`.

Only model-raised A42 false alarms are eligible for simulated caregiver confirmation. Confirmed samples are removed from subsequent personal-normal evaluation to avoid evaluating directly on feedback items.

## Go / stop interpretation

The current hypothesis is supported only if, across subjects and seeds:

1. pre-personalization personal-normal FPR is materially nontrivial;
2. at least one adaptation baseline reduces that FPR;
3. adaptation measurably reduces A43 recall or anomaly margin, or otherwise creates a reproducible safety-plasticity trade-off;
4. the effect is not explained entirely by one pathological baseline or by nominal feedback budgets that are never realized.

If threshold-only adaptation solves the problem without safety degradation, the proposed continual-learning contribution should be weakened or reformulated. If A42 is rarely flagged, the NTU mechanism test does not establish the motivating failure mode.
