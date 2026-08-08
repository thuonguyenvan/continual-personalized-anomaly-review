# Research Questions

## Primary research question

**RQ1.** How can one-class anomaly detectors continually personalize their notion of normality to a specific user from sparse human feedback, while preserving sensitivity to safety-critical anomalies?

## Landscape questions

**RQ2.** Which research communities and terms address this setting or its nearest subproblems (continual anomaly detection, online one-class learning, evolving normality, concept drift, personalized anomaly detection, human-in-the-loop anomaly detection, continual test-time adaptation, etc.)?

**RQ3.** What assumptions do existing approaches make about supervision, anomaly labels, temporal order, memory access, replay, user identity, and feedback availability?

**RQ4.** Which methods explicitly model personalization or subject-specific normality, and how do they prevent over-adaptation or contamination of the normal region?

**RQ5.** How are catastrophic forgetting, representation drift, decision-boundary expansion, and false-normal assimilation handled in continual anomaly detection?

## Evaluation questions

**RQ6.** Which datasets and protocols can realistically simulate deployment to a new elderly user whose benign behavior differs from population-level normality?

**RQ7.** Which metrics are appropriate for measuring the trade-off between reduced false alarms and preserved detection of critical anomalies?

**RQ8.** How much human feedback, memory, and computation do current methods require?

## Gap-identification questions

**RQ9.** What concrete failure modes remain unresolved by high-quality existing work?

**RQ10.** Which candidate contribution offers the strongest combination of novelty, methodological significance, practical relevance, and feasibility for a first research paper?

## Scope note

The first-stage research permits server-side updating. On-device continual learning is an extension rather than a mandatory component of the initial study.
