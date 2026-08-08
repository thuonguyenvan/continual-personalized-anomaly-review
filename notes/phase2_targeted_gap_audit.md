# Phase 2 — Targeted Gap Audit

## Goal

Phase 2 stress-tests the provisional hypothesis from Phase 1:

> **Safe continual expansion of user-specific normality from sparse human feedback.**

The audit focuses on neighboring top-tier/Q1 literature in human-in-the-loop anomaly detection, concept drift, online/continual adaptation, personalized activity modeling, and contamination-safe adaptation.

## Important evidence rule

Two evidence layers are kept separate:

1. **Core review corpus:** only peer-reviewed A/A*/top-tier conferences and Q1/high-quality journals, following the repository quality policy.
2. **Novelty sentinels:** a lower-tier/workshop paper may be recorded only if it is unusually close to the target problem and therefore could invalidate a novelty claim. Such papers are **not** used as primary methodological evidence or benchmark-quality support.

This distinction is essential: excluding a weak venue from the survey corpus does not make its idea nonexistent for novelty purposes.

---

## A. Human-in-the-loop anomaly detection already exists at top tier

### Self-Trained Deep Ordinal Regression for End-to-End Video Anomaly Detection — CVPR 2020

This paper explicitly demonstrates **human-in-the-loop video anomaly detection** and motivates the setting by the rarity of anomalies and high cost of false negatives.

### Consequence for our topic

We cannot claim that using human feedback in video anomaly detection is itself novel.

What remains different in our intended setting is the *semantics and temporal role* of feedback:

- feedback is primarily confirmation that a flagged event is actually **personal normal**;
- these confirmations accumulate across deployment sessions;
- they are used to continually modify a user's normality model;
- adaptation must preserve sensitivity to truly dangerous anomalies.

Therefore the candidate novelty is not `human feedback + anomaly detection`, but potentially **feedback-driven longitudinal normality personalization**.

---

## B. Concept drift and non-stationary online learning are mature top-tier topics

### DAWIDD — ICML 2020

DAWIDD studies non-parametric concept-drift detection and shows how distributional change can be detected without relying on a fixed predictive model.

### DriftSurf — ICML 2021

DriftSurf explicitly alternates stable and reactive states to adapt to concept drift while controlling false drift alarms.

### Continual Prototype Evolution — ICCV 2021

This work learns prototypes continually from non-stationary, online, imbalanced streams and explicitly connects continual learning with concept drift.

### Incremental Learning in Online Scenario — CVPR 2020

This work considers both catastrophic forgetting and changing distributions of previously observed classes.

### Consequence for our topic

We cannot claim novelty merely from:

- adapting to changing distributions;
- updating prototypes online;
- combining continual learning with concept drift;
- detecting a distribution shift after deployment.

The research question must specify **which distribution changes, what supervision is available, and what must be preserved**.

Our distinctive target is a semantically asymmetric drift:

> a pattern initially outside population-level normality is subsequently confirmed as benign for a specific person and should become personal normal, while dangerous anomalies must remain outside the accepted normal region.

This differs from generic concept drift because not every new/recurrent distribution should be absorbed.

---

## C. Online video anomaly detection is established, including strong Q1 work

### Online anomaly detection in surveillance videos with asymptotic bound on false alarm rate — Pattern Recognition, 2021

This work studies truly online video anomaly detection and provides a principled false-alarm treatment. It is valuable because **false-alarm control** is directly relevant to the intended elderly-monitoring use case.

### Consequence

Reducing false alarms in online VAD is not a sufficient contribution by itself. The important difference must be that our system uses longitudinal user-specific evidence/feedback to *change what counts as normal* rather than only calibrating an online decision rule.

---

## D. A near-exact historical precursor exists outside the strict core corpus

### Continual Learning for Anomaly Detection in Surveillance Videos — CVPR Workshop 2020

This work is excluded from the **core evidence corpus** under our venue policy, but it is a crucial **novelty sentinel** because it is unusually close conceptually:

- continual video anomaly detection;
- sequential updates;
- human expert labels false alarms;
- false-alarm features are added to memory so similar future events are not flagged.

### Why this matters

This paper means the following broad claim is unsafe:

> "We are the first to continually learn normal patterns from human-confirmed false alarms in video anomaly detection."

That claim is likely false or at least indefensible.

This is the most important correction produced by Phase 2.

### What it does *not* settle

The precursor does not, by itself, establish that the following have been solved:

- **subject-specific personalization** of normality;
- explicit separation of population-normal and personal-normal knowledge;
- safe/conservative expansion that protects critical anomaly recall;
- contamination-aware acceptance of feedback-derived normal patterns;
- sparse-feedback query selection;
- longitudinal subject-wise evaluation with repeated personalization sessions;
- skeleton/elderly-health-specific behavior drift.

Therefore the problem remains potentially publishable, but the contribution must move substantially beyond simple false-alarm memory insertion.

---

## E. Refined gap hypotheses after Phase 2

### G1 — Safety-constrained continual normality personalization

**Status: strongest candidate.**

The central technical tension becomes:

> How can the model expand personal normality using confirmed benign samples without moving the acceptance region toward safety-critical anomalies?

A serious method should explicitly model this constraint rather than relying on naive memory insertion or unrestricted fine-tuning.

Potential mechanisms to investigate later:

- frozen or slowly changing population-normal anchors;
- personal-normal prototypes learned from feedback;
- safety-critical abnormal anchors when available;
- margin constraints between new personal-normal clusters and dangerous anomalies;
- teacher/student preservation of anomaly scores;
- trust/uncertainty gates before absorbing a sample;
- delayed cluster-level acceptance rather than sample-level acceptance;
- rollback or update rejection when safety validation degrades.

**Novelty confidence:** medium, not yet high. A dedicated search for conservative/safe anomaly adaptation and contamination-resistant continual anomaly learning is still required.

---

### G2 — Feedback-efficient continual personalization

**Status: promising but likely adjacent to active anomaly detection.**

The research problem is not merely collecting feedback, but minimizing caregiver burden:

> Given a limited feedback budget, which alarms should be presented for confirmation so that personal normality improves fastest without compromising safety?

Potential signals:

- uncertainty;
- recurrence;
- cluster representativeness;
- diversity;
- expected reduction in false alarms;
- estimated safety risk.

**Novelty confidence:** low-to-medium until active anomaly-detection literature is fully venue-audited.

---

### G3 — Global–personal normality decomposition

**Status: promising structural contribution.**

Instead of a single mutable normal region, explicitly separate:

- `N_global`: stable population-level normality;
- `N_personal`: user-specific benign patterns learned after deployment;
- optionally `A_safety`: safety-critical anomaly anchors or protected regions.

The detector can then personalize without overwriting global knowledge.

**Novelty confidence:** medium. Needs targeted comparison with personalized HAR, domain adaptation, and mixture/prototype anomaly models.

---

### G4 — Deployment-realistic subject-wise continual anomaly protocol

**Status: potentially valuable benchmark contribution, especially when paired with G1.**

A benchmark should evaluate the actual loop:

`population normal training -> unseen subject -> false alarms -> sparse feedback -> update -> continued stream -> repeated updates`

and report both adaptation benefit and safety cost after every session.

Minimum metrics:

- personal-normal false-positive rate;
- critical-anomaly recall/sensitivity;
- AUROC/AUPRC;
- false alarms per unit time where possible;
- global-normal retention;
- adaptation/forgetting across sessions;
- feedback count;
- contamination rate or unsafe-normalization rate.

A protocol alone is unlikely to be enough for a top vision venue unless it reveals a major evaluation flaw or is accompanied by a meaningful dataset/method.

---

## F. Updated novelty boundaries

After Phase 2, the following claims should be considered **already occupied** or too broad:

- continual anomaly detection;
- online anomaly detection;
- concept-drift-aware continual learning;
- human-in-the-loop anomaly detection;
- using false alarms to improve a video anomaly detector;
- prototype evolution under non-stationarity;
- reducing online VAD false alarms.

The working claim must become narrower and technically stronger:

> **Safety-constrained, subject-specific continual normality personalization from sparse feedback.**

Even this remains a hypothesis until the remaining safety/personalization literature is audited.

---

## G. Recommended research formulation now

### Problem formulation

Given:

- an initial anomaly detector trained predominantly or exclusively on population-level normal behavior;
- a deployment stream from a previously unseen individual;
- sparse feedback indicating that selected model alarms are benign for that individual;
- optionally a small protected set of safety-critical anomalies for evaluation or constrained adaptation;

learn a sequence of personalized detectors such that:

1. false positives on recurring user-specific benign behavior decrease;
2. population-level normal knowledge is retained;
3. sensitivity to safety-critical anomalies does not significantly degrade;
4. adaptation remains robust to incorrect/noisy feedback or contaminated candidate-normal samples;
5. the number of required feedback interactions is limited.

### Core scientific tension

This is better described as a three-way balance:

- **plasticity** — absorb new personal normality;
- **stability** — preserve previous normality;
- **safety** — prevent dangerous anomalies from becoming normalized.

The third axis is what potentially distinguishes this problem from ordinary continual learning.

---

## H. Next required audit before algorithm design

Phase 3 should be narrower than Phase 2 and search only for possible killers of the G1 claim:

1. conservative/safe anomaly adaptation;
2. contamination-resistant anomaly detection with online updates;
3. continual anomaly learning with protected anomaly memory;
4. personalized anomaly detection with subject-specific normal regions;
5. human-feedback systems that update anomaly boundaries over repeated sessions;
6. safe/constraint-based continual learning applicable to anomaly boundaries.

Only after this audit should we freeze the paper's central novelty claim and begin method design.
