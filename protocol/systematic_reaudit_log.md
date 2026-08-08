# Systematic Re-Audit Search Log

## Scope

This log records the stricter second-pass search used to reinforce Phase 1 and Phase 2 before the novelty-killer audit.

Date of re-audit: 2026-08-08.

Target years: primarily 2020–2026; older papers are admitted only when foundational or unusually close to the problem.

## Source hierarchy

Primary-source preference:

1. official CVF Open Access pages for CVPR/ICCV/ECCV;
2. NeurIPS proceedings / official virtual pages;
3. PMLR / official ICML pages;
4. AAAI proceedings;
5. ACM Digital Library / official proceedings for The Web Conference and IMWUT;
6. IEEE Xplore / bibliographic records for TPAMI;
7. secondary indexes only to locate a paper, never as the sole evidence for an important claim.

ArXiv was not used as core evidence when a peer-reviewed version existed.

## Search families

The search was intentionally decomposed because the target problem is described differently across communities.

### S1 — continual / online anomaly detection

Representative query concepts:

- `continual anomaly detection`
- `online anomaly detection`
- `streaming anomaly detection concept drift`
- `memory poisoning anomaly detection`
- `long-tailed online anomaly detection`

Purpose: kill any novelty claim based only on sequential model updating.

### S2 — human-feedback anomaly detection

- `human-in-the-loop anomaly detection`
- `human feedback anomaly detection`
- `active anomaly detection`
- `false alarm feedback anomaly detection`
- `limited user feedback anomaly detection`

Purpose: determine whether false-alarm confirmation is itself novel.

### S3 — continual one-class / evolving normality

- `continual one-class classification`
- `incremental one-class learning`
- `online one-class classification`
- `evolving normality anomaly detection`
- `dynamic normal boundary anomaly detection`

Purpose: find the closest mathematical formulation to expanding a normal region over time.

### S4 — personalization / subject adaptation

- `personalized anomaly detection`
- `subject-specific anomaly detection`
- `personalized human activity recognition`
- `subject adaptation human activity recognition`
- `source-free HAR adaptation`

Purpose: establish how strongly personalization and inter-subject shift are already represented in high-quality literature.

### S5 — modality-specific continual learning

- `video class incremental learning`
- `continuous video stream learning`
- `continual skeleton action recognition`
- `class incremental skeleton gesture recognition`

Purpose: rule out modality + CL as sufficient novelty.

### S6 — safety / contamination / poisoning

- `contamination resilient anomaly detection`
- `anomaly detection contaminated normal data`
- `memory poisoning streaming anomaly detection`
- `robust adaptation malicious test samples`
- `safe test-time adaptation`

Purpose: distinguish our proposed safety semantics from generic contamination and adversarial robustness.

### S7 — continual test-time adaptation

- `continual test-time adaptation`
- `lifelong test-time adaptation`
- `robust test-time adaptation dynamic scenarios`
- `error accumulation test-time adaptation`
- `catastrophic forgetting test-time adaptation`

Purpose: understand stability/plasticity mechanisms already available for deployment-time updating.

## Screening decisions

### Core include

A paper is admitted when all are true:

1. peer-reviewed final publication;
2. main track of a recognized top conference or strong Q1 journal;
3. directly informs at least one axis of the target problem;
4. method/problem/evaluation details are sufficiently clear to map into the capability matrix.

### Novelty-sentinel include

A non-core paper may be logged separately when it is unusually close to the exact idea and therefore could invalidate a broad novelty claim. It must never be used to inflate the quality of the core evidence base.

### Exclude

- MDPI publications;
- arXiv-only papers;
- workshop papers as core evidence;
- low/unclear-rank venues;
- papers about anomaly detection only in name but solving a different statistical task;
- generic continual learning papers with no relevance to stream adaptation, anomaly learning, behavior/video/skeleton, feedback, or safety.

## Important novelty sentinel

**Continual Learning for Anomaly Detection in Surveillance Videos (CVPR Workshop 2020)** is excluded from the core corpus because it is a workshop paper, but it is retained conceptually as a novelty sentinel because it is close to continual video anomaly detection and feedback-driven false-alarm handling. Therefore, claims such as "first anomaly detector to learn from confirmed false alarms" should be considered unsafe unless narrowed further.

## Current counts

After the re-audit, `data/core_corpus_v2.csv` contains **32 core papers** spanning:

- one-class/temporal AD;
- continual/online AD;
- contamination robustness;
- video/skeleton continual learning;
- personalized/continual HAR;
- continual test-time adaptation;
- open-vocabulary VAD;
- continuous video-stream learning.

This is not yet the final systematic-review corpus. The goal is 30–50 strong core works with 10–15 papers selected for full deep reading after Phase 3.

## Audit conclusion

The evidence now strongly supports narrowing the candidate contribution away from generic continual learning and toward:

> **safety-constrained, subject-specific continual expansion of one-class normality from sparse human feedback.**

However, this remains a candidate gap, not a first-ever claim. Phase 3 must now search specifically for the closest possible novelty killers.
