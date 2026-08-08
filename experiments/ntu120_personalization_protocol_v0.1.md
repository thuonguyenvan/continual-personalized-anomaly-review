# NTU RGB+D 120 Personalization Protocol v0.1

## Purpose

This protocol is a **gap-validation experiment**, not yet a final benchmark proposal. Its purpose is to test whether the central research failure actually exists:

> Continual adaptation to feedback-confirmed personal-normal behavior reduces user-specific false alarms but may also reduce sensitivity to safety-critical anomalies by over-expanding the learned normal region.

The experiment deliberately starts with 3D skeleton sequences to minimize appearance/domain confounds and make the adaptation dynamics easier to interpret.

## Dataset facts used by the protocol

NTU RGB+D 120 contains 114,480 samples from 106 subjects, with RGB, depth, IR and 3D skeleton modalities. The official taxonomy contains 82 daily actions, 12 medical-condition actions, and 26 mutual actions. The official cross-subject protocol assigns 53 subjects to training and 53 to testing.

Official cross-subject training IDs:

`1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103`

Official cross-subject testing IDs:

`3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40, 41, 42, 43, 44, 48, 51, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 75, 76, 77, 79, 87, 88, 90, 96, 99, 101, 102, 104, 105, 106`

No test-subject sample may be used for global representation learning, detector fitting, hyperparameter tuning, or threshold selection.

## Important limitation

NTU RGB+D 120 is **not an elderly or Parkinson's dataset**. Therefore this protocol must never be presented as evidence about clinical Parkinson behavior. It is a controlled public benchmark used to test the learning mechanism and the safety/plasticity trade-off.

The elderly relevance must later be validated on ETRI-Activity3D and/or private elderly data.

---

# 1. Two complementary personalization stress tests

A single NTU protocol can accidentally conflate two different phenomena. We therefore separate them.

## Track A — Subject-style personalization

Goal: test whether the same population-normal actions are scored differently for an unseen subject because of subject-specific execution style.

### Global normal

Use selected single-person daily actions from official training subjects only.

### Personal normal

For each held-out target subject, use the **same semantic normal action set**, but treat early false positives from that subject as feedback-confirmed personal-normal samples.

### Interpretation

This track tests genuine subject shift without changing the semantics of what counts as normal.

### Limitation

If the global encoder generalizes well, FPR may already be low. A weak effect here does not invalidate Track B.

---

## Track B — Evolving-normality personalization

Goal: simulate the supervisor's target scenario more directly: a pattern that is outside the initial global normality later becomes accepted as normal for a particular user.

### Global normal

Train only on common daily actions from official training subjects.

### Candidate personal-normal behaviors

A small set of initially excluded behaviors is introduced only after deployment. These behaviors are treated as **normal for the target user after feedback**.

Primary candidate:

- `A42 — staggering`

This is especially useful because it is close to the motivating scenario of gait impairment: it is globally unusual but can be interpreted as a persistent user-specific condition after trusted feedback.

Secondary benign-new-normal candidates for robustness experiments:

- `A41 — sneeze/cough`
- `A103 — yawn`
- `A104 — stretch oneself`
- `A105 — blow nose`

These secondary actions are not intended to model Parkinson's disease. They provide additional cases where a behavior excluded from global normal training can later be incorporated as personal normal.

### Protected safety anomaly

Primary protected anomaly:

- `A43 — falling down`

A43 is **never** used for personalization updates and is always evaluated as a protected dangerous anomaly.

Optional secondary protected conditions can be studied later, but A43 should remain the primary endpoint because its skeleton semantics are visually meaningful and directly relevant to elderly monitoring.

### Why A42 and A43 are important together

A42 and A43 create a useful stress test:

> Can the detector absorb a gait-instability-like pattern as personal normal without moving the normal region so far that falling becomes less anomalous?

This is not a clinical claim; it is a controlled representation-learning test.

---

# 2. Action-role policy

## Global-normal pool

For the first pilot, use only **single-person daily actions**, excluding mutual actions and the 12 medical-condition actions.

To reduce object-centric and fine-finger-motion confounds in a skeleton-only pilot, begin with a compact locomotion/posture/ADL subset rather than all 82 daily classes.

Recommended pilot normal set:

- A1 drink water
- A2 eat meal
- A6 pick up
- A8 sit down
- A9 stand up
- A11 reading
- A12 writing
- A23 hand waving
- A25 reach into pocket
- A28 phone call
- A29 play with phone/tablet
- A33 check time
- A34 rub two hands
- A35 nod head/bow
- A36 shake head
- A37 wipe face
- A40 cross hands in front
- A80 squat down
- A96 cross arms
- A97 arm circles
- A98 arm swings
- A99 run on the spot
- A100 butt kicks
- A101 cross toe touch

This set is intentionally provisional. It should be frozen before the first reported pilot and later stress-tested with a broader normal pool.

## Excluded from pilot

- all mutual/two-person actions;
- A43 falling down from every normal/adaptation set;
- candidate personal-normal actions from initial global training in Track B;
- actions whose skeleton signal is too weak or object-dependent for the first mechanistic experiment, unless later added in robustness studies.

---

# 3. Subject split and nested validation

Use the official NTU120 cross-subject split as the outer split.

## Global development side

Official 53 training subjects are used for:

- encoder training or representation learning;
- fitting the global one-class detector;
- selecting hyperparameters;
- setting the initial anomaly threshold;
- ablation development.

Within these 53 subjects, create a deterministic inner split:

- approximately 80% development-train subjects;
- approximately 20% development-validation subjects.

The exact subject IDs must be generated once with a fixed seed and saved to a manifest before model development.

## Deployment side

The 53 official test subjects remain untouched until evaluation.

Each deployment subject is evaluated independently. Do not pool personal feedback across deployment subjects in the primary experiment.

This preserves the intended question:

> How does a population model personalize to one new person?

---

# 4. Session construction

The dataset is not naturally longitudinal, so sessions must be simulated carefully and transparently.

For each target subject:

## Session 0 — pre-adaptation

Evaluate the untouched global detector on:

- target-subject global-normal samples;
- candidate personal-normal samples;
- protected A43 falling samples.

Record baseline score distributions and operating metrics.

## Sessions 1..K — continual personalization

Recommended pilot: `K = 5` sessions.

At each session:

1. reveal a small batch of candidate personal-normal samples;
2. compute anomaly scores with the current model;
3. select samples that would have triggered an alert;
4. simulate caregiver feedback confirming a subset as normal;
5. update the method using **only information available up to that session**;
6. evaluate on held-out samples not used for that update.

### Feedback budgets

Primary budgets:

- 1 confirmed sample/session;
- 5 confirmed samples/session;
- 10 confirmed samples/session.

This allows feedback-efficiency curves rather than a single arbitrary setting.

## Ordering

Use at least three deterministic random session orderings per subject, or multiple fixed seeds, because NTU does not provide a true usage chronology.

Never claim these synthetic sessions reproduce real-world time. They approximate sequential deployment for mechanism validation.

---

# 5. No leakage rules

The following are prohibited:

1. selecting the threshold using official test subjects;
2. using future-session samples during an earlier update;
3. using A43 or any protected anomaly for ordinary adaptation;
4. selecting hyperparameters based on A43 performance on test subjects;
5. using all samples of a target subject to build a prototype before Session 0;
6. mixing data from different deployment subjects in the primary personalization model;
7. evaluating on the exact feedback samples used for adaptation and calling this personalization improvement.

For every session, maintain separate:

- adaptation set;
- held-out personal-normal evaluation set;
- global-normal retention set;
- protected-anomaly evaluation set.

---

# 6. Representation pipeline

## Stage 1 pilot

Prefer a frozen skeleton representation so that the first experiment isolates anomaly-boundary adaptation.

Candidate choices:

1. pretrained/fitted lightweight ST-GCN-family encoder;
2. simple temporal GCN encoder trained on global-normal/action data;
3. representation from a reproducible strong skeleton backbone, frozen before deployment.

The first gap test should not depend on inventing a new backbone.

## Normalization

At minimum:

- root-center skeletons;
- normalize body scale;
- preserve temporal motion;
- use a fixed temporal length or mask/padding policy;
- document person-selection logic for multi-body files.

The preprocessing pipeline must be identical before and after personalization.

---

# 7. Baselines

Run baselines in increasing complexity.

## B0 — No adaptation

Frozen encoder + frozen anomaly detector + fixed global threshold.

## B1 — Threshold-only personalization

Keep representation and normal model fixed; calibrate only the decision threshold from confirmed feedback.

This is a critical baseline. If it solves the problem, a continual-learning method may be unnecessary.

## B2 — Prototype update

Maintain a population-normal prototype and update/add a personal prototype from confirmed personal-normal embeddings.

Evaluate both:

- replacement/merged prototype;
- dual global + personal prototype scoring.

## B3 — Memory-bank / kNN update

Add confirmed personal-normal embeddings to a bounded memory and score by distance to trusted normal memory.

## B4 — Naive one-class update

Update the one-class detector using confirmed personal-normal feedback without an explicit safety constraint.

## B5 — Replay-preserving update

Update using personal-normal samples plus representative global-normal memory.

Only after these baselines should a new safety-constrained method be designed.

---

# 8. Metrics

## Personalization

Per subject and per session:

- personal-normal false-positive rate (FPR);
- false alarms per evaluated sample/session;
- mean/median anomaly score on personal-normal samples;
- fraction of confirmed patterns successfully accepted after adaptation.

## Safety

Primary:

- A43 falling recall/sensitivity;
- A43 false-negative rate;
- mean/quantile anomaly score on A43;
- score margin between A43 and personal-normal distributions.

A safety result should be reported at fixed operating points, not AUROC alone.

## Stability

- FPR on held-out population/global-normal actions;
- change in global-normal anomaly score;
- retention after every session;
- forgetting across previous personalization sessions.

## Overall detection

Secondary:

- AUROC;
- AUPRC;
- F1 at predeclared threshold(s).

## Feedback efficiency

- performance vs cumulative confirmed samples;
- area under the personalization-benefit-vs-feedback curve where useful.

---

# 9. Primary hypotheses

## H1 — Deployment shift

An unseen subject and/or newly introduced personal-normal behavior produces a materially higher false-positive rate than global normal validation data.

## H2 — Adaptation benefit

Using confirmed personal-normal feedback reduces target-subject false positives.

## H3 — Safety/plasticity tension

At least one naive adaptive baseline reduces personal-normal FPR while also decreasing A43 anomaly separation or recall.

H3 is the most important hypothesis for the candidate research direction.

## H4 — Threshold insufficiency

Threshold-only personalization cannot fully recover the personalization benefit without unacceptable safety or global-normal trade-offs.

If H4 fails strongly, the need for a learned continual-personalization method weakens.

---

# 10. Decision gates

## GO — safe continual personalization is supported

Proceed toward a new method if the experiments repeatedly show:

1. meaningful personal-normal false alarms before adaptation;
2. adaptation provides a useful reduction in false alarms;
3. naive adaptation causes measurable degradation in protected-anomaly sensitivity/separation or global-normal stability;
4. threshold calibration alone does not solve the trade-off.

Then the research target becomes:

> safety-constrained continual normality personalization.

## CONDITIONAL GO

If adaptation is useful but no safety degradation appears, pivot toward:

- feedback-efficient personalization;
- memory-efficient personalization;
- global/personal representation preservation;
- personalized calibration under evolving normality.

Do not manufacture a safety method without an observed failure.

## STOP / REFORMULATE

Reconsider the research premise if:

- subject/personal-normal FPR is negligible;
- threshold-only personalization matches learned adaptation;
- continued adaptation offers no meaningful benefit;
- the observed trade-off exists only under contrived settings.

---

# 11. Robustness studies after the pilot

Only after the pilot establishes the phenomenon:

1. broaden the global-normal action pool;
2. vary candidate personal-normal classes;
3. vary feedback budgets;
4. vary anomaly detector families;
5. vary skeleton encoders;
6. repeat with ETRI-Activity3D for real elderly subject variation;
7. validate on private elderly normal/fall data if available;
8. later move from server-side update to on-device constraints.

---

# 12. Claims this protocol does NOT support

Even with positive results, NTU120 alone cannot support claims that:

- the method is clinically validated for Parkinson's disease;
- A42 is a genuine Parkinson-specific behavior;
- the benchmark reproduces real caregiver feedback distributions;
- synthetic sessions represent true longitudinal disease evolution;
- the method is ready for safety-critical deployment.

The appropriate claim is narrower:

> NTU120 provides a controlled subject-wise skeleton benchmark for testing continual expansion of personalized normality while monitoring preservation of a protected fall-like anomaly.

---

# 13. Immediate implementation checklist

- [ ] Obtain official NTU RGB+D 120 skeleton files under the dataset license.
- [ ] Build metadata parser for subject, action, setup, camera, repetition.
- [ ] Save official CSub train/test manifests.
- [ ] Freeze pilot action-role manifest.
- [ ] Create inner development subject split with a fixed seed.
- [ ] Implement skeleton preprocessing.
- [ ] Train/freeze one baseline encoder.
- [ ] Implement global prototype/distance detector.
- [ ] Run Session 0 score-distribution audit before any continual update.
- [ ] Implement B0/B1/B2 first.
- [ ] Add B3/B4/B5 only if the phenomenon is visible.
- [ ] Save all per-subject/per-session scores, not only aggregate metrics.
