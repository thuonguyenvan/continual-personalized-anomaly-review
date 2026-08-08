# Continual Personalized Anomaly Review

A living, publication-oriented literature review and research-planning repository for **continual personalization of one-class anomaly detection under evolving normality**, with an initial application focus on elderly behavior monitoring from video and/or skeleton data.

## Research objective

The core setting is:

1. Train an initial one-class anomaly detector on general normal behavior.
2. Deploy it to a specific user.
3. User-specific but benign behaviors may be flagged as anomalies.
4. Sparse human feedback confirms selected false alarms as normal.
5. The model is updated over time to personalize its notion of normality.
6. Adaptation must reduce false alarms **without absorbing dangerous behavior into the normal region** or degrading sensitivity to critical anomalies.

The first research stage allows **server-side updates**. On-device continual adaptation is treated as a later extension.

## Quality policy

The review prioritizes peer-reviewed papers from recognized top venues and journals only, such as CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, TPAMI, IJCV, AAAI when directly relevant, and other established A/A* conferences or Q1 journals. The core evidence base excludes MDPI journals, low-rank or questionable venues, workshops as primary evidence, and arXiv-only/preprint manuscripts.

## Repository map

- `protocol/` — research questions, search strategy, inclusion/exclusion rules, venue-quality policy, screening protocol.
- `data/` — candidate corpus, included/excluded papers, mapping tables, dataset inventory.
- `literature/` — structured paper notes by research branch.
- `notes/` — taxonomy, closest competitors, research gaps, candidate directions.
- `experiments/` — baseline and benchmark protocol for validating literature-derived gaps.
- `manuscript/` — survey/systematic-review outline and eventual manuscript draft.

## Current status

This repository begins from a deep scoping review. The next phase is to convert that seed into a reproducible systematic-review workflow: venue audit, formal screening, deduplication, deep reading of the closest works, and experimental validation of candidate gaps.

## Central working question

> How can a one-class anomaly detector continually absorb user-specific normal behaviors from sparse human feedback while preserving sensitivity to truly dangerous events?
