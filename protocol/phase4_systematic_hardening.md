# Phase 4 — Systematic Review Hardening

## Goal

Convert the current scoping/targeted review into a reproducible, publication-oriented systematic/scoping review workflow while preserving the strict venue-quality policy.

This phase is **not** allowed to relax quality criteria merely to increase paper count.

## Core evidence policy

Core evidence must be:

- peer reviewed;
- final published version;
- main-track top conference (A/A*) or established Q1 journal;
- directly relevant to one of the defined research questions.

Excluded from core evidence:

- arXiv-only/preprint manuscripts;
- MDPI journals;
- low-rank or questionable venues;
- workshop papers as primary evidence;
- papers whose only relation is application-level elderly/fall detection without methodological relevance.

Workshop/preprint papers may be retained as **novelty sentinels** when they could invalidate an over-broad novelty claim, but they must be labeled separately.

## Research-question families

### RQ1 — Evolving normality
How do anomaly detectors adapt when the definition/distribution of normal behavior changes after deployment?

### RQ2 — Continual anomaly detection
How do top-tier methods update anomaly detectors sequentially while mitigating catastrophic forgetting?

### RQ3 — Human feedback
How is sparse expert/human feedback used to correct anomaly/OOD false positives or update model decisions?

### RQ4 — Personalization
How do models adapt from population-level behavior to user/subject-specific behavior?

### RQ5 — Safety/contamination
How do adaptive anomaly detectors prevent anomalous or unsafe samples from contaminating the learned normal region?

### RQ6 — Repeated-session adaptation
Which methods explicitly evaluate long-term/continual adaptation over non-stationary streams rather than a single batch/domain shift?

### RQ7 — Video/skeleton relevance
Which findings transfer to video, pose, skeleton, HAR, or human-behavior anomaly detection?

## Formal search families

Search each family independently before combining them. Exact database syntax should be adapted to the indexing engine while preserving semantics.

### F1 — Continual / online anomaly detection

```text
("continual anomaly detection" OR "online anomaly detection" OR
 "streaming anomaly detection" OR "incremental anomaly detection")
```

### F2 — Evolving normality / new normal

```text
("new normal" OR "evolving normality" OR "normality adaptation" OR
 "distribution shift")
AND
("anomaly detection" OR "one-class")
```

### F3 — Human feedback / false positives

```text
("human feedback" OR "human-in-the-loop" OR "expert feedback" OR
 "false positive mining" OR "feedback-guided")
AND
("anomaly detection" OR "out-of-distribution" OR "one-class")
```

### F4 — Personalization / subject adaptation

```text
("personalized" OR "subject-specific" OR "user-specific" OR "subject adaptation")
AND
("anomaly detection" OR "normal behavior" OR "human activity recognition")
```

### F5 — Safety / contamination / conservative adaptation

```text
("contamination" OR "safe adaptation" OR "robust adaptation" OR
 "memory poisoning" OR "model drift" OR "error accumulation")
AND
("anomaly detection" OR "test-time adaptation" OR "online learning")
```

### F6 — Human behavior modality

```text
("video" OR "skeleton" OR "pose" OR "human activity")
AND
("anomaly detection" OR "continual" OR "online" OR "personalized")
```

## Source hierarchy

### Tier 1 — primary publication sources

- CVF Open Access / IEEE proceedings for CVPR, ICCV;
- ECCV/Springer proceedings;
- PMLR for ICML/AISTATS;
- NeurIPS proceedings;
- official AAAI proceedings;
- ACM Digital Library / official conference proceedings;
- IEEE Xplore for TPAMI/TNNLS/TKDE and other accepted Q1 journals;
- Springer for IJCV and other established Q1 journals.

### Tier 2 — indexing and citation chaining

- Web of Science;
- Scopus;
- DBLP;
- Google Scholar only for discovery/forward-citation tracing.

A Tier-2 record never substitutes for verification against the final publication source when that source is available.

## Screening workflow

1. **Identification** — collect records from every search family/source.
2. **Deduplication** — DOI first, then normalized title.
3. **Venue-quality screen** — reject non-core venues before topical deep reading.
4. **Title/abstract screen** — map each candidate to at least one RQ.
5. **Full-text eligibility** — extract problem formulation, assumptions, method, protocol, datasets, limitations.
6. **Closest-work tagging** — label papers that satisfy >=3 target capabilities.
7. **Novelty-killer tagging** — label any work that could invalidate a candidate novelty statement.
8. **Backward/forward snowballing** — mandatory for every closest-work/novelty-killer paper.

## Extraction schema

For every included paper record:

- canonical title;
- authors;
- year;
- venue;
- rank/tier;
- DOI;
- final publication URL;
- modality;
- anomaly/OOD/one-class setting;
- continual/online/TTA setting;
- what changes over time;
- supervision at deployment;
- human feedback type;
- personalization level;
- update mechanism;
- memory/replay mechanism;
- contamination/safety mechanism;
- repeated-session evaluation;
- source-data access assumption;
- benchmark/datasets;
- primary metrics;
- key result relevant to this review;
- limitation relative to our target problem;
- novelty-threat level: low / medium / high / killer.

## PRISMA bookkeeping

Maintain counts for:

- records identified per database/search family;
- duplicates removed;
- records excluded by venue-quality policy;
- records excluded after title/abstract;
- full texts assessed;
- full texts excluded with reason;
- final core corpus;
- novelty sentinels retained outside the core corpus.

**Important:** current paper counts in the repository are still provisional because earlier phases were targeted audits rather than exhaustive database exports. Do not publish a PRISMA diagram until database-level searches and counts are frozen.

## Quality assessment

Each core paper receives 0/1 on:

1. peer-reviewed final version verified;
2. top-tier/Q1 venue verified;
3. problem formulation clearly defined;
4. evaluation protocol reproducible from paper/supplement;
5. multiple datasets/settings or strong methodological justification;
6. appropriate baselines;
7. limitations/assumptions identifiable;
8. direct relevance to at least one RQ.

This score is for review transparency, **not** for ranking scientific merit across unrelated subfields.

## Stop rule for literature expansion

The corpus is considered sufficiently stable for research-direction selection when:

1. every closest competitor has backward and forward citation tracing completed;
2. two consecutive targeted searches in each critical family (F2-F5) produce no new high-threat core paper;
3. every proposed novelty claim is contrasted against at least one closest paper;
4. candidate-gap conclusions remain unchanged after the latest search round.

## Current hardening priority

Deep-read and citation-chain these first:

1. When Model Meets New Normals — AAAI 2024.
2. CANDI — AAAI 2026.
3. One-for-More — CVPR 2025.
4. DFM — CVPR 2025.
5. MemStream — The Web Conference 2022.
6. Taming False Positives in OOD Detection with Human Feedback — AISTATS 2024.
7. Contamination-Resilient Anomaly Detection — ICML 2024.
8. Self-Trained Deep Ordinal Regression for End-to-End VAD — CVPR 2020.
9. CoTTA — CVPR 2022.
10. RoTTA / PETAL — CVPR 2023.

## Deliverables before calling Phase 4 complete

- [ ] frozen search strings per database;
- [ ] database-level search log with dates and result counts;
- [ ] deduplicated master candidate CSV;
- [ ] venue-verified included/excluded tables;
- [ ] closest-paper capability matrix;
- [ ] deep-read summaries for 10–15 closest papers;
- [ ] backward/forward citation-chain log;
- [ ] quality-assessment table;
- [ ] provisional PRISMA flow;
- [ ] research-gap statement rewritten only from the final evidence map.
