# Phase 1 — Closest-Work Audit

## Purpose

This phase tests whether the target problem is already directly solved in high-quality literature. The working target is:

> Continually personalize a one-class anomaly detector for an individual user using sparse human-confirmed false alarms, reducing false positives while preserving sensitivity to genuinely dangerous anomalies.

The first stage permits server-side model updates. On-device updating is intentionally excluded from the core scope for now.

## Strict evidence policy

Only peer-reviewed work from recognized top venues/journals is admitted to the core evidence base. The present audit uses CVPR, ICCV, NeurIPS, ICML, and TPAMI papers. ArXiv-only papers, MDPI papers, workshops, and low-rank venues are not used to establish novelty claims.

## Initial findings

### 1. Continual anomaly detection clearly exists as a top-tier problem

The strongest direct reference found so far is **One-for-More (CVPR 2025)**. It explicitly studies continual anomaly detection and catastrophic forgetting, but its setting is continual visual anomaly detection on MVTec/VisA rather than user-specific evolving normality, video/skeleton behavior, or sparse human feedback.

**Implication:** the research claim cannot be "introducing continual learning to anomaly detection." That claim would be false.

### 2. Online anomaly detection also exists, but with a different adaptation target

**Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts (ICCV 2025)** studies online anomaly detection under long-tailed streams. It is highly relevant to streaming/online adaptation, but it does not model a personalized normal region whose semantics change through caregiver feedback.

**Implication:** the novelty must be tied to the *type of distributional evolution and supervision*, not merely online updating.

### 3. Continuous learning from video is established

**Learning from One Continuous Video Stream (CVPR 2024)** demonstrates rigorous online learning from a single continuous video stream. Video continual/class-incremental learning is also represented by **Park et al. (ICCV 2021)**, **vCLIMB (CVPR 2022)**, and **Space-time Prompting (ICCV 2023)**.

**Implication:** sequential video learning is not new. These works are useful for protocols, forgetting metrics, memory design, and adaptation schedules.

### 4. Continual learning from skeleton streams is established

**Else-Net (ICCV 2021)** studies continual action recognition from skeleton sequences. **Data-Free Class-Incremental Hand Gesture Recognition (ICCV 2023)** studies privacy-aware skeleton class-incremental learning.

**Implication:** using skeleton data plus continual learning alone is insufficient novelty.

### 5. One-class anomaly detection is mature, but mostly static

**OneFlow (TPAMI 2022)**, **THOC (NeurIPS 2020)**, and the foundational **Deep SVDD (ICML 2018)** provide strong one-class formulations. They learn compact representations/normal regions but do not directly address longitudinal user personalization with feedback-confirmed false alarms.

**Implication:** the research should not focus merely on inventing another static OCC loss unless it directly solves the evolving-normality problem.

### 6. Open-world/open-vocabulary VAD is related but is not the same problem

**Open-Vocabulary Video Anomaly Detection (CVPR 2024)** and **Anomize (CVPR 2025)** focus on detecting/categorizing unseen anomaly types. Our target instead concerns samples initially classified as anomalous that later become trusted *personal normal* through feedback.

**Implication:** the proposed task should be framed around **evolving normality / continual personalization**, not new anomaly-class discovery.

## Current evidence matrix

| Research component | Strong top-tier precedent? | Examples | Remaining gap relative to target |
|---|---:|---|---|
| One-class anomaly detection | Yes | Deep SVDD, THOC, OneFlow | Static normality in most methods |
| Video anomaly detection | Yes | GCL, OVVAD, Anomize | Mostly fixed training/deployment distribution |
| Continual anomaly detection | Yes | One-for-More | Not user-personalized; no sparse human false-alarm feedback |
| Online anomaly detection | Yes | LTOAD | Different online assumptions/objective |
| Continual video learning | Yes | Park et al., vCLIMB, continuous-video learning, ST-Prompt | Mostly recognition rather than evolving normality |
| Continual skeleton learning | Yes | Else-Net, DFCIL-HGR | Class/action incremental rather than OCC personalization |
| Personalized normality | **Not yet established in this audited top-tier set** | — | Needs targeted audit |
| Human-confirmed false alarms as continual normal supervision | **Not yet established in this audited top-tier set** | — | Needs targeted audit |
| Safety against absorbing true anomalies during personalization | **Not yet established in this audited top-tier set** | — | Candidate high-value gap; not yet a novelty claim |

## Most important provisional gap

The strongest *working hypothesis* after this first audit is not simply "continual anomaly detection." It is:

> **Safe continual expansion of user-specific normality from sparse human feedback.**

A candidate problem formulation is:

> Given a one-class detector trained on population-level normal behavior, a stream from a new individual, and sparse feedback confirming selected false alarms as normal, update the detector over multiple sessions so that it reduces user-specific false positives while preserving global normal knowledge and maintaining high sensitivity to safety-critical anomalies.

This formulation contains three coupled technical tensions:

1. **Plasticity:** absorb legitimate user-specific normal behavior.
2. **Stability:** avoid forgetting previously learned normal behavior.
3. **Safety:** prevent the normal region from expanding into truly anomalous/dangerous behavior.

The third item is potentially more distinctive than ordinary catastrophic forgetting, but it must be verified by further literature search before being claimed as novel.

## What is NOT yet justified

At this stage, we must not claim any of the following:

- "No prior work studies personalized anomaly detection."
- "No prior work uses human feedback for anomaly adaptation."
- "This is the first continual one-class personalization method."
- "The problem is novel enough for CVPR/ICCV/TPAMI."

Those claims require a second targeted audit centered on personalization, human-in-the-loop anomaly detection, active anomaly detection, evolving normality, and anomaly detection under concept drift, while retaining the strict venue policy.

## Phase 2 search targets

The next audit should search top-tier/Q1 literature for:

1. personalized anomaly detection / personalized behavior modeling;
2. human-in-the-loop and active anomaly detection;
3. continual or incremental one-class learning;
4. anomaly detection under concept drift/evolving distributions;
5. subject adaptation and personalized HAR;
6. conservative/safe adaptation and contamination-resistant anomaly learning;
7. temporal/video/skeleton versions of the above.

Only after Phase 2 should we rank candidate research gaps and design the first baseline experiment.
