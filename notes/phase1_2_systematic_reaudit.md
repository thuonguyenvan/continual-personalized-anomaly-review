# Phase 1–2 Systematic Re-Audit

## Why this re-audit exists

The first two phases were useful for scoping, but they were targeted audits rather than a sufficiently reproducible systematic review. This document upgrades them into a stricter evidence workflow before any novelty claim is made.

The target problem remains:

> Continually personalize a one-class anomaly detector for an individual user using sparse human-confirmed false alarms, reducing user-specific false positives while preserving sensitivity to truly dangerous anomalies.

Server-side updating is allowed in the first-stage research. On-device updating is deliberately deferred.

## Evidence policy

Core evidence is restricted to peer-reviewed main-track papers from recognized top conferences (e.g., CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI, The Web Conference when directly relevant) and strong Q1 journals such as TPAMI/IJCV/IMWUT when directly relevant.

The following are excluded from the core novelty corpus:

- arXiv-only/preprint manuscripts;
- MDPI journals;
- workshops and challenge papers;
- low-rank or unclear-quality venues;
- papers whose only relationship is superficial keyword overlap.

Workshop/preprint papers may be retained only as **novelty sentinels**: they cannot establish the state of the art, but they can invalidate an overly broad first-ever claim.

## Re-audit result: claims already ruled out

The following are not defensible novelty claims:

1. **"Continual learning for anomaly detection is new."**
   - One-for-More (CVPR 2025) directly studies continual anomaly detection.

2. **"Online anomaly detection under distribution change is new."**
   - MemStream (The Web Conference 2022) handles streaming anomaly detection with concept drift and explicitly addresses memory poisoning.
   - Long-Tailed Online Anomaly Detection (ICCV 2025) further establishes online AD as an active top-tier problem.

3. **"Using human feedback in anomaly detection is new."**
   - Self-Trained Deep Ordinal Regression (CVPR 2020) includes an effective human-in-the-loop anomaly-detection setting.
   - A CVPR 2020 workshop paper on continual surveillance anomaly detection is especially close to the false-alarm-feedback idea; although excluded from core evidence, it is a strong novelty sentinel.

4. **"Continual learning for video/skeleton behavior is new."**
   - Video CIL: ICCV 2021, vCLIMB/CVPR 2022, ST-Prompt/ICCV 2023.
   - Skeleton continual action recognition: Else-Net/ICCV 2021 and data-free skeleton gesture CIL/ICCV 2023.

5. **"Contamination-safe anomaly learning is new."**
   - ICML 2024 studies contamination-resilient anomaly detection.
   - NeurIPS 2025 studies post-hoc anomaly adjustment under contaminated training data.
   These are not continual-personalization methods, but they mean any safety claim must be more specific than generic contamination robustness.

6. **"Adaptation can be assumed safe if it improves target accuracy."**
   - Main-track CVPR/ICCV test-time adaptation literature explicitly documents adaptation instability, malicious-test vulnerability, pseudo-label noise, robustness loss, and stability/plasticity trade-offs.

## Strongest verified neighboring literatures

### A. Static one-class/anomaly modeling

- Deep SVDD (ICML 2018): foundational deep one-class representation learning.
- THOC (NeurIPS 2020): temporal hierarchical one-class modeling for time-series anomaly detection.
- OneFlow (TPAMI 2022): one-class boundary learning via a minimal-volume normal region.

These establish the detector side, but not longitudinal subject-specific feedback adaptation.

### B. Continual / online anomaly detection

- MemStream (WWW 2022): online streaming AD, concept drift, dynamic memory, and memory poisoning resistance.
- One-for-More (CVPR 2025): continual anomaly detection with catastrophic-forgetting control.
- Long-Tailed Online Anomaly Detection (ICCV 2025): online class-agnostic anomaly learning.
- RareCLIP (ICCV 2025): online zero-shot AD with historical prototype memory.

These are the most important novelty killers for any generic continual/online claim.

### C. Video / skeleton continual learning

- Class-Incremental Learning for Action Recognition in Videos (ICCV 2021).
- vCLIMB (CVPR 2022).
- Space-time Prompting for Video CIL (ICCV 2023).
- Else-Net (ICCV 2021).
- Data-Free Class-Incremental Hand Gesture Recognition (ICCV 2023).

These show that modality + continual learning alone is not a contribution.

### D. Human feedback / personalization-adjacent work

- Self-Trained Deep Ordinal Regression for End-to-End Video Anomaly Detection (CVPR 2020): human-in-the-loop anomaly detection.
- Personalized Human Activity Recognition Using CNNs (AAAI 2018): minimal-supervision user personalization.
- MobHAR (IMWUT 2025): source-free transfer for HAR under distribution shift.
- Generalizable Sensor-Based Activity Recognition via Categorical Concept Invariant Learning (AAAI 2025): inter-subject variation and unseen distributions.
- ConSense (AAAI 2025): continual activity sensing with privacy/storage motivation.

None of these, in the present audit, directly instantiate the full target loop: population-normal model -> new individual -> false alarms -> sparse confirmations -> repeated normality expansion -> explicit protection of critical anomalies.

### E. Safety / contamination / adaptation robustness

- Contamination-Resilient Anomaly Detection via Adversarial Learning on Partially-Observed Normal and Anomalous Data (ICML 2024).
- An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination (NeurIPS 2025).
- MedBN (CVPR 2024): robust test-time adaptation against malicious test samples.
- Realistic Test-Time Adaptation of Vision-Language Models (CVPR 2025): shows that common TTA can sacrifice initial robustness under realistic non-IID conditions.
- Hybrid-TTA (ICCV 2025): continual TTA explicitly balances adaptation and stabilization.

These papers make the safety component scientifically plausible, but also raise the bar: a new method must protect *anomaly semantics during user-specific normality expansion*, not merely be robust to generic distribution shift.

## Current capability matrix

| Capability | Strong top-tier precedent | Evidence state |
|---|---:|---|
| One-class normal modeling | Yes | Mature |
| Video anomaly detection | Yes | Mature |
| Continual anomaly detection | Yes | Established |
| Streaming/concept-drift AD | Yes | Established |
| Video continual learning | Yes | Established |
| Skeleton continual learning | Yes | Established |
| Human feedback in AD | Yes | Exists |
| User personalization in HAR | Yes | Exists |
| Contamination-resilient AD | Yes | Exists |
| Continual TTA stability/plasticity | Yes | Exists |
| Subject-specific continual **normality** expansion from false-alarm feedback | Not directly established in this audited core corpus | Candidate gap |
| Explicit protection of safety-critical anomaly regions during that personalization | Not directly established in this audited core corpus | Strong candidate gap |
| Repeated-session protocol combining sparse feedback, personalization, forgetting and safety metrics | Not directly established in this audited core corpus | Candidate benchmark gap |

## Revised candidate research gap

The defensible working hypothesis is now narrower:

> **Safety-constrained continual personalization of one-class normality from sparse human feedback.**

The scientific problem is not merely to learn newly confirmed normal samples. It is to solve a three-way tension:

1. **Plasticity** — absorb genuine user-specific normal behaviors quickly.
2. **Stability** — preserve useful global/past normal knowledge across update sessions.
3. **Safety** — prevent adaptation from swallowing truly dangerous anomalies into the expanding normal region.

A fourth practical axis is **feedback efficiency**: achieve the above under sparse caregiver confirmations.

## What remains unproven

Even after this re-audit, we must not yet claim:

- first method for personalized anomaly detection;
- first method for continual one-class personalization;
- first method to learn from false alarms;
- first safe anomaly adaptation method;
- top-tier novelty.

The next phase must intentionally search for papers that could kill the narrowed claim.

## Phase 3 entry criterion

Proceed to algorithm design only after a novelty-killer audit has searched, at minimum:

- safe/conservative anomaly adaptation;
- poisoning-resistant or contamination-aware online AD;
- feedback-driven sequential AD;
- personalized / subject-adaptive anomaly detection;
- continual test-time adaptation with protected knowledge;
- dynamic one-class boundaries / online OCC;
- healthcare/behavior-monitoring personalization in top-tier venues.

The output must identify the 5–10 closest works and state, in one sentence per work, exactly which required capability is absent.
