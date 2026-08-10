# Closest Competitors — Deep Read v1

## Purpose

This note does not attempt a broad survey. It asks a stricter question:

> Which high-quality papers come closest to solving the target problem, and exactly what remains unsolved relative to subject-specific continual normality personalization from sparse trusted feedback with explicit safety preservation?

The current target is:

> A population-level one-class/anomaly detector is deployed to a new individual. User-specific benign behavior initially triggers alarms. Sparse trusted feedback confirms selected false alarms as normal. The model updates repeatedly over time to absorb personal normality while preserving sensitivity to protected safety-critical anomalies.

---

## 1. When Model Meets New Normals — AAAI 2024

### Problem solved

The paper explicitly identifies the **new normal problem** in unsupervised time-series anomaly detection: the distribution of normality can shift between training and deployment, so static detectors may incorrectly reject valid new normal patterns.

### Mechanism

Test-time adaptation uses trend estimation plus self-supervised learning to adapt to newly observed normalities at inference time.

### Why it is close

This paper removes a major possible novelty claim: **adapting anomaly detection to new normality after deployment is already a published top-tier problem.**

### What it does not provide relative to our target

- no subject-specific/user-specific personalization objective;
- no trusted caregiver/expert confirmation of false alarms as the main supervision signal;
- no explicit protected set of dangerous anomalies that must remain outside the normal region;
- not centered on repeated per-user personalization sessions in human behavior/video/skeleton monitoring.

### Novelty consequence

We cannot claim novelty from “learning new normals.” Any contribution must be more specific than this.

---

## 2. CANDI — AAAI 2026

### Problem solved

CANDI studies multivariate time-series anomaly detection under deployment distribution shift and adapts the pretrained detector at test time.

### Mechanism

The method performs **False Positive Mining (FPM)** using anomaly scores and latent similarity to curate likely false-positive adaptation samples. It then uses a dedicated normality-adaptation module while attempting to preserve pretrained knowledge.

### Why it is the most dangerous competitor

Its high-level loop is very close to the motivating system:

`deployment -> likely false positive -> treat as adaptation evidence -> update normality`

Therefore, merely proposing automatic false-positive selection followed by normality adaptation would be insufficient.

### Remaining differences

- mined false positives are not explicitly trusted human-confirmed personal-normal feedback;
- the target is MTSAD distribution shift, not subject-specific human behavior;
- no explicit safety-critical anomaly protection objective of the form “this dangerous region must remain anomalous while personal normality expands”;
- no caregiver-feedback budget or per-user personalization protocol.

### Novelty consequence

Our method must beat a stronger claim than “use false alarms to learn normality.” The defensible axis is **trusted subject-specific feedback + safety-constrained expansion + repeated personalization**.

---

## 3. One-for-More — CVPR 2025

### Problem solved

Direct continual anomaly detection. It addresses catastrophic forgetting as anomaly-detection patterns arrive incrementally.

### Mechanism

A diffusion-based detector uses gradient projection to protect learned knowledge, plus memory-efficient iterative SVD and anomaly-masked conditioning.

### Why it matters

It invalidates any claim that continual anomaly detection is itself new.

### Difference from our target

- industrial/image anomaly benchmarks rather than user behavior;
- no personalization to one individual;
- no human feedback;
- stability is framed as catastrophic-forgetting protection rather than preventing trusted-normal adaptation from absorbing specific dangerous anomalies.

### What to borrow

- continual evaluation discipline;
- knowledge-preservation framing;
- per-stage reporting;
- stability/plasticity analysis.

---

## 4. DFM — CVPR 2025

### Problem solved

Differentiable feature matching for anomaly detection, with strong results including continual anomaly detection settings.

### Why it matters

It shows that continual AD is not represented by only one specialized method; strong general anomaly frameworks are now also evaluated in continual settings.

### Difference from our target

DFM does not center its contribution on evolving personal normality, sparse human feedback, or safety-constrained personalization.

### Novelty consequence

A future method should not rely on a weak/static anomaly backbone and claim gains only because continual evaluation is new. Continual AD baselines need to be competitive.

---

## 5. MemStream — The Web Conference 2022

### Problem solved

Streaming anomaly detection under concept drift. The method maintains a memory of recent normal trends and adapts online.

### Safety-relevant mechanism

MemStream explicitly considers **memory poisoning**, making it particularly relevant to the risk of anomalous samples entering adaptive normal memory.

### Why it is dangerous to our safety claim

We cannot claim that “preventing contamination of an adaptive normal memory” is new in general.

### Remaining difference

- no trusted human-confirmed personalization;
- not user/subject specific;
- poisoning robustness is not the same as maintaining a semantic protected safety-anomaly margin during personalization;
- no human behavior modality.

### What to borrow

A bounded trusted-normal memory baseline should be included in experiments.

---

## 6. Taming False Positives in OOD Detection with Human Feedback — AISTATS 2024

### Problem solved

Uses expert feedback to update an OOD decision threshold online and provides theoretical control of the false-positive rate while limiting human feedback.

### Why it is extremely close conceptually

It combines:

- deployment-time false positives;
- human/expert feedback;
- online adjustment;
- explicit safety/reliability control.

### Critical difference

The method adapts **the threshold**, rather than continually learning a personalized normal representation/model from recurring subject-specific behavior.

### Novelty consequence

This makes **threshold-only personalization a mandatory baseline**. If threshold calibration solves the practical false-alarm problem, a complex continual-learning method is unjustified.

It also means the phrase “safe human-feedback adaptation for false positives” is too broad to claim as novel.

---

## 7. Contamination-Resilient Anomaly Detection — ICML 2024

### Problem solved

Learns a normal-data distribution when the available dataset is contaminated and only small partially observed normal/anomaly sets are available.

### Why it matters

The paper provides theoretical and empirical evidence that contamination-safe anomaly learning is already a serious top-tier topic.

### Difference from our target

- principally a training-data contamination setting rather than sequential personal-normal expansion;
- not per-user personalization;
- not repeated deployment feedback;
- not specifically protecting safety-critical behavior semantics while accepting adjacent personal-normal behavior.

### Novelty consequence

We should formulate safety as a **continual boundary/representation constraint under personalized normality expansion**, not generic contamination robustness.

---

## 8. Self-Trained Deep Ordinal Regression for End-to-End VAD — CVPR 2020

### Problem solved

Unsupervised video anomaly detection with self-training; the authors also demonstrate a human-in-the-loop anomaly detection setting.

### Why it matters

Human participation in video anomaly detection is not new at main-track CVPR level.

### Difference from our target

- not a longitudinal subject-personalization problem;
- human involvement is not framed as repeated caregiver-confirmed new-normal expansion;
- no explicit global-normal vs personal-normal vs protected-dangerous knowledge structure.

### Novelty consequence

“Human-in-the-loop VAD” cannot be a novelty statement by itself.

---

## 9. CoTTA — CVPR 2022

### Problem solved

Continual test-time adaptation under non-stationary target streams. It explicitly addresses noisy pseudo-label error accumulation and catastrophic forgetting.

### Mechanism

Weight/augmentation averaged teacher predictions plus stochastic restoration toward source model weights.

### Relevance

Provides a mature stability/plasticity baseline and a clear example that naïve continual adaptation can drift.

### Difference from our target

Classification/segmentation under domain shift rather than one-class normality expansion; no trusted human feedback or subject-specific normal semantics.

---

## 10. RoTTA / PETAL — CVPR 2023

### Problem solved

Both study realistic/lifelong TTA under temporally evolving streams.

- **RoTTA** uses a memory bank with timeliness and uncertainty plus teacher-student stabilization.
- **PETAL** uses probabilistic lifelong TTA, source regularization, uncertainty and data-driven parameter restoration.

### Why they matter

These papers show that repeated-session continual deployment, uncertainty-aware memory and anti-drift mechanisms are already well developed outside anomaly detection.

### Difference from our target

Neither formulates the critical event as:

> trusted feedback says an initially anomalous pattern is personal normal, but nearby/related dangerous behavior must remain anomalous.

---

# Capability matrix interpretation

After this deep read, essentially every **individual ingredient** has a strong precedent:

- continual anomaly detection — yes;
- new-normal adaptation — yes;
- false-positive mining — yes;
- human feedback to reduce false positives — yes;
- streaming memory under concept drift — yes;
- contamination/memory-poisoning robustness — yes;
- stability/plasticity in repeated adaptation — yes;
- human-in-the-loop VAD — yes.

Therefore the research case cannot be built on ingredient novelty.

The remaining defensible problem hypothesis is narrower:

> **Subject-specific, human-confirmed continual expansion of normality under an explicit constraint that protected safety-critical anomalies remain separated across repeated personalization sessions.**

This is still a hypothesis, not a first-of-its-kind claim.

---

# Research design implications

## Mandatory baselines

At minimum, the eventual experiment should include:

1. no adaptation;
2. threshold-only adaptation inspired by human-feedback OOD control;
3. streaming trusted-normal memory / kNN or prototype memory;
4. naïve normality adaptation;
5. source/global-normal preserving continual adaptation;
6. a contamination-resistant baseline where implementable;
7. a safety-constrained candidate method only if naïve adaptation exhibits the predicted failure.

## Mandatory ablations

Separate the effect of:

- human confirmation vs automatic false-positive mining;
- personal memory vs global memory;
- safety anchors/constraints;
- feedback budget;
- repeated sessions;
- threshold change vs representation/model change.

## Strongest falsification test

The candidate direction should be abandoned or reformulated if:

- threshold-only adaptation is sufficient;
- personal-normal adaptation does not harm protected-anomaly separation;
- the effect appears only under an artificially chosen action pair;
- the method needs access to protected anomaly labels at deployment in a way inconsistent with the intended system.

---

# Current verdict

The literature no longer supports a broad claim such as:

> "We introduce continual learning with human feedback for anomaly detection."

A potentially defensible research question is instead:

> **How can a population-level anomaly detector repeatedly personalize its normal representation to one user using sparse trusted false-alarm feedback, while guaranteeing or empirically preserving a protected safety-anomaly margin and global-normal knowledge?**

The immediate next evidence needed is experimental: determine whether the personalization/safety trade-off is observable under a controlled subject-wise protocol before designing a new method.
