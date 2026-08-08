# Gap-Validation Experiment

## Goal

Before proposing a new method, test whether the hypothesized safety–plasticity conflict actually appears under a realistic personalization loop.

Working hypothesis:

> Naïve continual adaptation to human-confirmed false alarms reduces subject-specific false positives, but may also expand the learned normal region toward safety-critical anomalies and reduce dangerous-event sensitivity.

The experiment is intentionally simple. Its purpose is to validate the research problem, not to maximize anomaly-detection accuracy.

---

## Research questions

### E1. Does subject shift create persistent false alarms?

Given a detector trained on population-level normal data, does a held-out subject with idiosyncratic benign behavior exhibit a materially higher false-positive rate?

### E2. Does naïve personalization reduce those false alarms?

After adding sparse feedback-confirmed false alarms as new normal samples, how much does subject-specific FPR decrease?

### E3. Does this adaptation damage safety-critical anomaly detection?

After each personalization update, measure whether recall, FNR, and anomaly-score margins on critical anomalies deteriorate.

### E4. Is the effect cumulative across repeated sessions?

Run several sequential update sessions rather than a single fine-tuning step. Determine whether adaptation drift accumulates.

### E5. Which update mechanism is most vulnerable?

Compare lightweight boundary/prototype updates against gradient-based fine-tuning and replay-based updates.

---

## Minimal deployment simulation

For each target subject `u`:

1. Build a global training set using normal data from subjects excluding `u`.
2. Train the initial detector `M0`.
3. Split subject `u` chronologically or pseudo-chronologically into sessions `S1 ... ST`.
4. In every session, expose benign user-specific samples and safety-critical abnormal samples only for evaluation unless explicitly released as feedback.
5. Run `Mt` and collect samples predicted abnormal.
6. Reveal a small feedback budget from false alarms that are known normal.
7. Update the detector to `M(t+1)`.
8. Re-evaluate on:
   - current/future personal-normal samples;
   - retained global-normal validation data;
   - safety-critical abnormal data.
9. Repeat over sessions.

The core protocol must be subject-independent during initial training and subject-specific only after deployment.

---

## Feedback simulation

The first experiment should use an oracle to simulate caregiver feedback so that algorithmic effects are isolated from annotation noise.

Feedback budgets to evaluate:

- `K = 1` confirmed false alarm per session;
- `K = 5`;
- `K = 10`;
- optionally percentage-based budgets if session sizes vary strongly.

Selection policies:

1. random false alarm;
2. highest-confidence anomaly among false alarms;
3. cluster representative / medoid;
4. diversity-aware selection.

For the first pass, random selection is sufficient. More sophisticated querying belongs to a later research direction unless the feedback budget itself becomes the dominant limitation.

---

## Baselines

### B0 — No adaptation

Freeze the global model for all sessions. This establishes the personalization need.

### B1 — Threshold recalibration only

Adjust only the anomaly threshold using confirmed personal-normal samples. This tests whether a full continual-learning method is unnecessary.

### B2 — Prototype / centroid update

Keep the encoder frozen and incrementally update one or more normal prototypes.

### B3 — One-Class SVM refit/update

Use fixed embeddings and retrain/refit the one-class decision boundary using retained global normal plus confirmed personal normal.

### B4 — Naïve gradient fine-tuning

Fine-tune the detector/head on confirmed personal-normal samples without explicit preservation constraints.

### B5 — Replay-preserving fine-tuning

Fine-tune with confirmed personal-normal samples plus a small replay subset of global-normal samples.

### B6 — Oracle offline retraining

Retrain using all normal data available up to the current session. This is not deployable but provides an upper/reference bound.

A representative closest-method baseline from the final top-tier corpus should be added once reproduction feasibility is verified.

---

## Model hierarchy

Start with the simplest representation that can expose the phenomenon.

### Stage A — frozen representation

- skeleton or video encoder frozen;
- embeddings extracted once;
- anomaly detector operates on embeddings.

Candidate detectors:

- distance-to-centroid / prototype;
- One-Class SVM;
- Deep-SVDD-style head.

This isolates normality-boundary adaptation from backbone representation learning.

### Stage B — partial neural adaptation

Only if Stage A shows a meaningful trade-off:

- update projection/head only;
- then selected encoder blocks if necessary.

Do not begin with full video-backbone fine-tuning.

---

## Data requirements

The ideal benchmark must support:

1. subject identity;
2. normal activity sequences;
3. abnormal/dangerous activity sequences;
4. enough inter-subject variation to create personal-normal shift;
5. temporal ordering, or at least a defensible session simulation.

Priority modalities:

1. skeleton / pose sequence — preferred for clean first experiments and later edge deployment;
2. RGB video — secondary validation once the protocol is established.

If no single public dataset contains both strong subject-specific benign variation and critical anomalies, construct the first benchmark carefully from compatible public sources only if leakage/domain mismatch can be controlled. Private elderly data should be external validation, not the sole evidence base.

---

## Metrics

### Personalization

For each session `t`:

- personal-normal false-positive rate `FPR_personal(t)`;
- false alarms per clip/hour/session if duration metadata exists;
- mean anomaly score on personal-normal data.

### Safety

For safety-critical abnormal set `A_safe`:

- recall / sensitivity;
- false-negative rate;
- AUPRC for anomaly detection;
- mean anomaly score;
- minimum or percentile anomaly margin relative to personalized-normal samples.

Define a safety-drop statistic:

`SafetyDrop(t) = Recall_safe(M0) - Recall_safe(Mt)`

and a personalization-gain statistic:

`PersonalGain(t) = FPR_personal(M0) - FPR_personal(Mt)`

The central empirical object is the trade-off curve between `PersonalGain` and `SafetyDrop`.

### Stability

- global-normal retention FPR;
- forgetting after each session;
- embedding/prototype/boundary drift when measurable.

### Feedback efficiency

- cumulative human confirmations;
- FPR reduction per feedback item.

---

## Main hypothesis test

A naïve method supports the proposed research gap only if the following pattern occurs across multiple subjects/seeds:

1. `PersonalGain > 0` by a meaningful margin;
2. `SafetyDrop > 0` or safety anomaly margins shrink materially;
3. the deterioration grows or persists across sequential updates;
4. replay or conservative updates partially reduce the safety loss, suggesting the conflict is algorithmically controllable rather than a pure dataset artifact.

If naïve adaptation reduces FPR without meaningful safety degradation, the current candidate gap must be weakened or reformulated.

---

## Required ablations for the validation stage

- feedback budget `K`;
- number of sessions;
- frozen vs partially trainable representation;
- with vs without global-normal replay;
- threshold-only vs representation/boundary update;
- subject-wise results rather than aggregate-only reporting.

Run at least 3 random seeds for stochastic methods. Prefer confidence intervals across subjects.

---

## Stop / go criteria

### GO — proceed to method design

Proceed if a reproducible personalization–safety trade-off is observed for several baseline families and multiple subjects.

Then the method-design question becomes:

> How can personal normality be expanded while explicitly constraining dangerous-anomaly absorption and preserving global normality?

### CONDITIONAL GO

If degradation appears only under gradient fine-tuning but not simple prototype/threshold adaptation, narrow the contribution toward the regime where learned representations must adapt.

### STOP / REFRAME

Reconsider the current direction if:

- subject-specific false alarms are weak or rare;
- threshold recalibration solves most of the issue;
- naïve updates improve personalization without safety loss;
- the trade-off appears only under unrealistic synthetic settings.

---

## Immediate implementation order

1. Select one public subject-wise skeleton/behavior dataset.
2. Define population-normal / personal-normal / safety-abnormal splits.
3. Extract frozen embeddings.
4. Implement B0–B3 first.
5. Simulate 3–5 personalization sessions.
6. Plot `FPR_personal` and safety recall after each session.
7. Only then add neural fine-tuning/replay baselines.

The experiment should be treated as a research-direction falsification test, not as confirmation of a predetermined hypothesis.
