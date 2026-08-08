# Search Strategy

The review uses multiple neighboring terminologies because the target problem spans several communities.

## Core concept blocks

### A. Continual / online adaptation

- "continual anomaly detection"
- "continual one-class"
- "online anomaly detection"
- "incremental anomaly detection"
- "continual adaptation anomaly"
- "online one-class classification"
- "incremental one-class learning"

### B. Evolving normality / drift

- "evolving normality"
- "concept drift anomaly detection"
- "adaptive anomaly detection"
- "streaming anomaly detection"
- "non-stationary anomaly detection"

### C. Personalization

- "personalized anomaly detection"
- "personalized activity recognition"
- "subject-specific anomaly detection"
- "user-specific normal behavior"
- "subject adaptation anomaly detection"

### D. Human feedback

- "human-in-the-loop anomaly detection"
- "feedback-guided anomaly detection"
- "interactive anomaly detection"
- "active anomaly detection"
- "relevance feedback anomaly detection"
- "weak supervision anomaly feedback"

### E. Video / skeleton behavior

- "video anomaly detection" continual
- "skeleton anomaly detection"
- "pose-based video anomaly detection"
- "human behavior anomaly detection" continual
- "elderly behavior anomaly detection"
- "fall detection" personalization

## Combined query templates

Use combinations such as:

```text
("continual" OR "online" OR "incremental")
AND
("anomaly detection" OR "one-class classification")
```

```text
("personalized" OR "subject-specific" OR "user-specific")
AND
("anomaly detection" OR "normal behavior")
```

```text
("human-in-the-loop" OR "feedback-guided" OR "active learning")
AND
("anomaly detection" OR "one-class")
```

```text
("concept drift" OR "evolving normality" OR "non-stationary")
AND
("anomaly detection" OR "one-class")
```

```text
("video" OR "skeleton" OR "pose")
AND
("anomaly detection")
AND
("continual" OR "online" OR "personalized")
```

## Search-source plan

Prioritize authoritative indexing/proceedings sources:

- IEEE Xplore
- ACM Digital Library
- SpringerLink / ECCV proceedings
- CVF Open Access
- NeurIPS proceedings
- PMLR for ICML
- OpenReview only to verify accepted ICLR papers, not unreviewed submissions
- Web of Science / Scopus for systematic indexing and citation chaining when available
- Google Scholar only as a discovery/citation-chaining aid, never as the sole verification source

## Search logging

Every formal search should record:

- database/source
- exact query string
- date searched
- filters used
- number of returned records
- number after deduplication
- notes on indexing limitations
