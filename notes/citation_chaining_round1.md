# Citation Chaining Round 1

## Purpose

This round performs targeted backward/forward-style discovery around the closest novelty threats identified so far: evolving/new normality, false-positive-driven adaptation, human-feedback threshold adaptation, streaming anomaly detection, and continual anomaly detection.

Only peer-reviewed main-track/top-tier or strong Q1/A* works are admitted to the core evidence. Workshop/preprint material remains excluded from novelty claims.

## New high-priority papers discovered

### TUNE — Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time (AAAI 2026)

**Why it matters:** TUNE directly studies *unseen but normal* samples that are falsely detected as anomalies after deployment. It introduces test-time adaptation specifically to correct false alarms caused by normality shift.

**Threat to our novelty:** Very high at the problem-definition level. It further weakens any claim that adapting to unseen normal patterns or correcting post-deployment false alarms is novel.

**Still missing relative to our target:**
- subject-specific personalization;
- trusted human confirmation of false alarms;
- repeated personalization sessions;
- explicit preservation of a protected dangerous-anomaly region;
- video/skeleton elderly behavior setting.

### Zero-Shot Anomaly Detection via Batch Normalization / Adaptive Centered Representations (NeurIPS 2023)

**Why it matters:** Explicitly addresses drift in the normal distribution and generalization to a "new normal" without target training data.

**Threat:** High against broad new-normal claims. A method cannot claim novelty merely from allowing a normality distribution to shift.

**Difference:** Zero-shot/generalization setting rather than continual subject-specific adaptation from feedback.

### Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective (NeurIPS 2023)

**Why it matters:** Establishes top-tier work on making anomaly detectors robust to domain/covariate shifts through invariant representations.

**Threat:** Rules out broad claims around "robust anomaly detection under distribution shift".

**Difference:** Seeks invariance rather than personalized continual expansion of normality.

### ATTA — Anomaly-aware Test-Time Adaptation for OOD Detection in Segmentation (NeurIPS 2023)

**Why it matters:** Jointly handles domain shift and semantic/OOD detection, explicitly adapting while trying to preserve anomaly discrimination.

**Threat:** Important safety-adaptation precedent. It shows that adaptation while preserving OOD/anomaly detectability is already a recognized top-tier problem.

**Difference:** Segmentation/OOD setting, no user personalization, no sparse trusted feedback, no evolving personal normality across sessions.

### Monitoring Risks in Test-Time Adaptation (NeurIPS 2025)

**Why it matters:** Adds statistically rigorous risk monitoring to continually adapting models, raising alarms when deployment performance violates predefined criteria.

**Threat:** Weakens any generic claim that "safe adaptation requires monitoring risk".

**Difference:** Risk monitoring is generic and not a mechanism for safe normality expansion in one-class anomaly detection.

## Existing closest competitors reinforced

### When Model Meets New Normals (AAAI 2024)

Directly formulates the evolving/new-normal problem in unsupervised time-series anomaly detection and adapts the detector at test time.

### CANDI (AAAI 2026)

Selectively mines potential false positives for adaptation while preserving pretrained knowledge. This remains one of the strongest direct algorithmic competitors to a naive feedback-driven normality-update idea.

### Taming False Positives in OOD Detection with Human Feedback (AISTATS 2024)

Uses expert feedback to safely adapt thresholds online and provides FPR guarantees. Human feedback + online false-positive reduction is therefore not itself a novelty.

### One-for-More (CVPR 2025)

Confirms continual anomaly detection and catastrophic-forgetting mitigation are established top-tier research problems.

## Updated novelty boundary

After this citation-chaining round, the defensible candidate gap is narrower:

> **Subject-specific, human-confirmed continual normality personalization across repeated sessions, with an explicit constraint that prevents safety-critical anomalies from being absorbed into the personalized normal region.**

The following claims should now be considered invalid or too broad:

- adapting anomaly detectors to new normals is novel;
- correcting false alarms from unseen normals is novel;
- using false positives as adaptation candidates is novel;
- human feedback for online false-positive reduction is novel;
- preserving anomaly/OOD detection during test-time adaptation is novel in a generic sense;
- risk monitoring for adaptive models is novel.

## What would still make a first paper scientifically meaningful

The contribution must demonstrate a failure that existing methods do not explicitly solve:

1. a population one-class model is deployed to a new subject;
2. trusted false alarms become personal-normal supervision over multiple sessions;
3. naive or existing new-normal adaptation reduces false alarms;
4. but this adaptation can shrink the anomaly-score margin or increase false negatives for protected safety-critical events;
5. a proposed method enforces subject-specific plasticity while preserving a safety separation constraint;
6. results are reported per subject and per session, rather than only in pooled stationary evaluation.

## Literature saturation status

The broad landscape is approaching saturation. Additional literature search should now be triggered mainly by:

- a newly discovered direct competitor;
- a new method idea whose components require novelty checking;
- reviewer-style verification before submission.

The main next step should therefore remain empirical gap validation rather than indefinite expansion of the review corpus.
