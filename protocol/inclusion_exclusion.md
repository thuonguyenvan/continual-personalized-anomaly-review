# Inclusion and Exclusion Criteria

## Inclusion criteria

A study is eligible for the core corpus when all applicable conditions hold:

1. It is a peer-reviewed full paper.
2. It appears in a recognized top conference (A/A* or equivalent) or a verified Q1/high-impact journal.
3. It is methodologically relevant to at least one core branch:
   - one-class anomaly detection;
   - video/skeleton anomaly detection;
   - continual/online anomaly detection;
   - concept drift/evolving normality;
   - personalization/subject adaptation;
   - human-in-the-loop or feedback-guided anomaly learning;
   - continual learning mechanisms directly transferable to this setting.
4. For the primary evidence map, publication year is preferably 2020–2026. Older work is included only when foundational.
5. The paper provides enough methodological and experimental detail to assess assumptions, update mechanism, evaluation protocol, and limitations.

## Exclusion criteria

Exclude from the core corpus when any of the following holds:

- arXiv/preprint only;
- workshop-only publication;
- MDPI venue;
- low-ranked, unclear, or questionable venue;
- not peer reviewed;
- purely application-specific with no relevant methodological contribution;
- unrelated anomaly setting with no plausible transfer to sequential/personalized behavior modeling;
- duplicate of an included final publication;
- inaccessible or insufficient information to verify the claims;
- paper addresses only static offline anomaly detection and contributes no foundational mechanism needed for the taxonomy.

## Screening stages

### Stage 1 — Title/abstract screening

Check topical relevance and obvious venue-quality violations.

### Stage 2 — Venue verification

Verify peer-review status and venue quality before deep inclusion.

### Stage 3 — Full-text screening

Assess problem formulation, supervision, temporal setting, personalization, update mechanism, contamination safeguards, datasets, metrics, and limitations.

### Stage 4 — Closest-work designation

A paper is marked `closest_work = yes` only if it overlaps with multiple defining dimensions of the target problem, not merely one broad keyword.

## Target-problem dimensions

The strongest match is a study that covers several of:

- anomaly/one-class formulation;
- temporal or continual updates;
- user/subject-specific personalization;
- sparse human feedback;
- evolving normality/concept drift;
- safety against over-adaptation or contamination;
- video/skeleton behavior data.
