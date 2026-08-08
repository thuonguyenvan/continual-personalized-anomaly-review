# Phase 3 — Novelty-Killer Audit

## Purpose

This phase deliberately searches for strong prior work that could invalidate the current leading research formulation:

> **Safety-constrained, subject-specific continual normality personalization from sparse human feedback.**

The goal is adversarial: find papers that already solve most or all of the target problem, especially papers from A/A*/Q1 venues.

## Main novelty killers found

### K1. Evolving / “new normal” anomaly detection is already a recognized top-tier problem

**When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection** (AAAI 2024) explicitly formulates the **new normal problem**: normality evolves over time under distribution shift, and the anomaly detector should adapt to new normal patterns at inference time.

This is extremely close to the *evolving normality* component of our target problem.

**Consequence:** we cannot claim novelty from “learning new normal behavior after deployment” by itself.

### K2. Selective false-positive adaptation is now directly addressed

**CANDI: Curated Test-Time Adaptation for Multivariate Time-Series Anomaly Detection Under Distribution Shift** (AAAI 2026) selectively mines **potential false positives** and adapts the anomaly detector while preserving pretrained knowledge.

This is one of the strongest novelty killers found so far because it already combines:

- anomaly detection under distribution shift;
- false-positive-driven adaptation;
- selective curation of adaptation samples;
- preservation of pretrained knowledge;
- fewer adaptation samples.

**Consequence:** “identify false positives and update normality while preserving previous knowledge” is not sufficient as a novelty claim.

### K3. Human-feedback-controlled false-positive adaptation has strong adjacent precedent

**Taming False Positives in Out-of-Distribution Detection with Human Feedback** (AISTATS 2024) proposes expert-feedback-based online threshold updating with explicit statistical FPR guarantees and minimal feedback use.

Although OOD detection is not identical to one-class anomaly detection, the paper is conceptually dangerous to any claim such as:

> “using sparse expert feedback to safely reduce false positives online.”

That broad claim is already occupied.

### K4. Risk monitoring during adaptation is now an explicit top-tier topic

**Monitoring Risks in Test-Time Adaptation** (NeurIPS 2025) treats adaptation itself as a source of deployment risk and develops statistical monitoring mechanisms that raise alerts when performance criteria are violated.

This does not personalize a one-class normal region, but it weakens generic claims about “safe adaptation.”

**Consequence:** our safety contribution must be anomaly-specific and mechanistically tied to preventing dangerous samples from being absorbed into the normal region, not merely risk monitoring in general.

### K5. Robust continual test-time adaptation already addresses stability under realistic streams

**NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation** (NeurIPS 2022), **Test-time Adaptation in Non-stationary Environments via Adaptive Representation Alignment** (NeurIPS 2024), and related TTA work demonstrate that temporal correlation, non-stationarity, representation drift, and stability/plasticity are established problems.

**Consequence:** stability under a stream and continual distribution shift alone are not sufficient novelty.

### K6. Contamination-aware anomaly adaptation is an active high-quality line

**Contamination-Resilient Anomaly Detection via Adversarial Learning on Partially-Observed Normal and Anomalous Data** (ICML 2024) and **An Evidence-Based Post-Hoc Adjustment Framework for Anomaly Detection Under Data Contamination** (NeurIPS 2025) directly address contamination in anomaly learning/adaptation.

**Consequence:** generic “avoid anomaly contamination” is not novel enough. The contribution must focus on a more specific failure mode: **personalization-induced semantic contamination of the evolving normal class**, especially in a safety-critical user-specific stream.

## Important preprint-only novelty sentinels

The following are **not part of the core evidence base** because they are currently preprints, but they are dangerous enough that they must be monitored:

- **Selective Test-Time Adaptation for Unsupervised Anomaly Detection using Neural Implicit Representations** — explicitly motivates selective adaptation because blindly adapting can learn pathology/anomaly.
- **When Normality Shifts: Risk-Aware Test-Time Adaptation for Unsupervised Tabular Anomaly Detection** — explicitly selects high-confidence pseudo-normal samples and constrains anomalous ones during adaptation.

If these become accepted at strong venues, they may further narrow the safe-adaptation gap.

## Closest capability matrix

| Capability | AAAI 2024 New Normals | CANDI AAAI 2026 | AISTATS 2024 Human-feedback OOD | NeurIPS 2025 Risk Monitoring | Target problem |
|---|---:|---:|---:|---:|---:|
| Evolving normality | Yes | Yes | Partial | No | Yes |
| Continual / online adaptation | Yes | Yes | Yes | Yes | Yes |
| False-positive reduction | Indirect | Yes | Yes | Indirect | Yes |
| Sparse human feedback | No | No | Yes | No | Yes |
| Subject-specific personalization | No | No | No | No | Yes |
| One-class / anomaly normal-region update | Yes | Yes | OOD threshold only | No | Yes |
| Preserve prior knowledge | Partial | Yes | N/A | Monitors risk | Yes |
| Prevent true-dangerous samples entering normality | Not explicit | Not semantic/user-specific | No normal-region update | No | **Required** |
| Repeated user-specific sessions | No | No | No | Generic stream | **Required** |
| Safety-critical behavior semantics | No | No | Generic OOD | Generic TTA | **Required** |

## Surviving research gap after Phase 3

The broad versions of the idea have been substantially weakened. The strongest formulation that still survives is:

> **Human-confirmed, subject-specific continual normality expansion with explicit protection against safety-critical anomaly absorption across repeated personalization sessions.**

A stronger formal statement is:

> Given a population-trained one-class anomaly detector, a stream from a new individual, sparse human confirmations that selected alarms are benign, and a protected set or representation of safety-critical anomalies, continually personalize the user's normal region while minimizing false alarms subject to an explicit constraint on critical-anomaly sensitivity and global-normal retention.

The distinctive coupling is now:

1. **Personalization:** the target distribution is a single user's evolving normality, not a generic target domain.
2. **Human-confirmed normal expansion:** adaptation samples are sparse false alarms explicitly confirmed as benign.
3. **Safety constraint:** adaptation must not absorb dangerous behavior into the normal region.
4. **Repeated-session continual learning:** the model must retain old personalized normality across multiple update rounds.
5. **Behavioral/video or skeleton setting:** the deployment semantics are temporal human behaviors rather than generic image/tabular anomalies.

## Claims that are now unsafe

Do **not** claim any of the following:

- “first method for new-normal anomaly detection”;
- “first anomaly detector that adapts to false positives”;
- “first human-feedback anomaly adaptation method”;
- “first safe test-time adaptation method”;
- “first contamination-resistant anomaly adaptation method”;
- “first continual anomaly detector under distribution shift.”

## Current research recommendation

The candidate direction remains viable, but only in a narrow, technically demanding form. The most defensible first-paper question is now:

> **How can sparse, human-confirmed false alarms be used to continually personalize a one-class normality model for a specific person without reducing sensitivity to safety-critical anomalies?**

A publication-strength method will likely need an explicit mechanism such as:

- protected anomaly / safety anchors;
- constrained representation drift;
- trust-weighted feedback assimilation;
- dual global-personal normal memory;
- risk-aware update gating;
- rollback or delayed acceptance of new normal clusters;
- optimization with a safety-retention constraint rather than only an adaptation loss.

## Phase 3 verdict

**The candidate survives, but only after substantial narrowing.**

The strongest novelty is no longer continual learning, evolving normality, false-positive adaptation, human feedback, or contamination robustness individually. It is the **joint safety-constrained personalization problem** and a method that specifically solves the failure mode where benign user-specific behavior must be learned without converting genuinely dangerous behaviors into normality.

Before method design, the next recommended step is a small **gap-validation benchmark experiment** to determine whether naïve feedback-driven adaptation actually exhibits the predicted safety trade-off on realistic data. If the failure does not appear empirically, the gap is not strong enough regardless of literature novelty.
