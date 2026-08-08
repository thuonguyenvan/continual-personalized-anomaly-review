# Candidate Research Gaps

These are **working hypotheses**, not novelty claims. Each must be validated against the final screened top-tier corpus and, ideally, by baseline experiments.

## G1. Safe continual normality expansion

### Hypothesis

Existing continual or adaptive anomaly detectors may reduce false alarms by adapting to newly observed normal patterns, but do not sufficiently constrain the normal region from expanding into safety-critical abnormal regions.

### Why it matters

In elderly monitoring, adaptation is not useful if it lowers false-positive rate at the cost of missing falls, seizures, unconsciousness, or other dangerous events.

### What evidence is needed

- top-tier methods that adapt anomaly boundaries over time;
- whether they model contamination explicitly;
- whether they preserve critical anomaly sensitivity during personalization;
- controlled experiments showing naïve update can reduce false alarms while increasing false negatives.

## G2. Sparse-feedback continual personalization

### Hypothesis

The literature may address either continual anomaly detection or personalized adaptation, but not the combination where sparse human confirmation of false alarms is the primary supervision signal.

### Key question

How should a system decide which alarms to query, retain, and trust when caregiver feedback is limited?

### Candidate mechanisms

- uncertainty-aware querying;
- recurrence/cluster representativeness;
- diversity-aware selection;
- safety-aware query prioritization.

## G3. Global-personal normality preservation

### Hypothesis

Personalization can overfit to one user's short-term behavior and erase useful population-level normality.

### Candidate formulation

Maintain complementary representations/memories:

- global normal anchors;
- personal normal prototypes;
- safety-critical abnormal anchors or margins.

## G4. Deployment-realistic benchmark protocol

### Hypothesis

Existing anomaly-detection benchmarks may not evaluate the actual temporal loop:

`global training -> new subject deployment -> false alarms -> sparse feedback -> repeated updates -> continued evaluation`.

A rigorous subject-wise temporal protocol may itself be a meaningful contribution if current top-tier work lacks one.

## G5. Resource-aware feedback memory

Potential later direction: retain only representative temporal segments/embeddings from confirmed false alarms under a strict byte budget.

This should support the main personalization/safety problem rather than become a competing first-paper contribution unless evidence shows memory is the dominant bottleneck.

## Current priority

The strongest current candidate is **G1: safe feedback-aware continual normality expansion**, followed by **G2: sparse-feedback personalization**. This ranking is provisional until the venue-audited closest-work matrix is complete.
