# Baseline Plan for Gap Validation

The first experiments should test whether the literature-derived problem is real before designing a complex method.

## Core deployment simulation

1. Train a global one-class anomaly detector on population-level normal behavior.
2. Evaluate on a new subject whose benign behavior differs from training normality.
3. Identify false alarms on subject-specific normal patterns.
4. Reveal sparse feedback confirming selected false alarms as normal.
5. Update the detector using a naïve strategy.
6. Repeat over multiple sessions.
7. Evaluate whether false alarms fall and whether dangerous-anomaly sensitivity deteriorates.

## Baseline families

- no adaptation;
- full offline retraining / oracle reference;
- naïve fine-tuning on confirmed personal-normal samples;
- replay-based update;
- prototype or memory-bank update;
- one-class boundary update;
- continual-learning regularization where appropriate.

Exact methods will be selected only from the final venue-audited literature corpus.

## Initial detector candidates

Use simple, interpretable baselines before complex architecture work:

- fixed encoder + One-Class SVM;
- fixed encoder + Deep SVDD-style objective;
- fixed encoder + prototype/distance anomaly score;
- a representative top-tier video/skeleton anomaly detector if reproducible.

## Primary metrics

### Personalization benefit

- false-positive rate on user-specific benign behavior;
- false alarms per unit time/session;
- AUROC / AUPRC where appropriate.

### Safety preservation

- recall/sensitivity on critical anomalies;
- false-negative rate on critical anomalies;
- anomaly-score margin between personalized-normal and safety-critical abnormal samples.

### Continual behavior

- performance after each update session;
- retention on global normal data;
- backward forgetting / representation drift where definable;
- cumulative number of feedback labels.

### Efficiency

Initially secondary because updates may run on a server:

- memory footprint;
- stored samples/embeddings;
- update time;
- training compute.

## Central failure test

The most important early hypothesis is:

> Naïvely adapting to feedback-confirmed false alarms reduces false positives, but can over-expand the learned normal region and reduce sensitivity to safety-critical anomalies.

If this failure does not occur consistently, the proposed safe-adaptation method needs to be reconsidered before proceeding.
