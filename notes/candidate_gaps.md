# Candidate Research Gaps

These remain **working hypotheses**, not final novelty claims. Phase 2 substantially narrowed the claim space by finding top-tier precedent for human-in-the-loop anomaly detection and concept-drift adaptation, plus a very close CVPR-workshop novelty sentinel that already learns from human-confirmed false alarms.

## G1. Safety-constrained continual normality personalization

### Hypothesis

The strongest remaining candidate is not generic continual anomaly detection or learning from false alarms. It is **subject-specific continual expansion of normality under an explicit safety constraint**.

The target problem is:

> Given a population-level one-class detector, a stream from a new individual, and sparse feedback confirming selected false alarms as benign, continually personalize the detector so that false positives fall while sensitivity to safety-critical anomalies remains protected.

### Why this remains plausible

Phase 2 confirms that several neighboring components are already established:

- human-in-the-loop anomaly detection;
- continual/online anomaly detection;
- concept-drift-aware online learning;
- prototype evolution under non-stationarity;
- false-alarm control.

A 2020 CVPR Workshop paper is also a close novelty sentinel: human-confirmed false alarms are inserted into memory to prevent similar future alarms. Therefore **"continual learning from human-confirmed false alarms" cannot be used as a broad novelty claim**.

What still needs to be tested is whether high-quality prior work explicitly solves the three-way tension:

1. **plasticity** — absorb legitimate personal normality;
2. **stability** — retain population/global normality;
3. **safety** — prevent true anomalies from entering the normal region.

### Evidence still required

- conservative/safe anomaly adaptation;
- contamination-resistant online anomaly learning;
- personalized anomaly detectors with evolving subject-specific normal regions;
- continual anomaly learning with protected anomaly anchors/memory;
- repeated-session human-feedback adaptation;
- experimental evidence that naive personalization improves FPR but harms critical-anomaly recall.

### Candidate mechanisms

- global normal anchors + personal normal prototypes;
- protected safety-critical abnormal anchors or margins;
- trust/uncertainty gate before accepting feedback-derived normality;
- cluster-level delayed acceptance;
- distillation or score-preservation constraints;
- rollback/update rejection if safety validation degrades.

**Current priority: highest.**

---

## G2. Feedback-efficient continual personalization

### Hypothesis

Human feedback is known in anomaly detection, so the gap cannot be simply "use caregiver feedback." A stronger question is:

> Under a strict feedback budget, which alarms should be queried so that personal normality improves fastest without compromising safety?

### Candidate mechanisms

- uncertainty-aware querying;
- recurrence/cluster representativeness;
- diversity-aware selection;
- expected false-alarm reduction;
- risk-aware query prioritization.

### Remaining risk

This direction may overlap heavily with active anomaly detection / active learning. It requires a dedicated venue-audited search before being promoted to a central contribution.

**Current priority: medium.**

---

## G3. Global–personal normality decomposition

### Hypothesis

A single mutable normal region may cause personalization to overwrite useful population-level knowledge.

### Candidate formulation

Maintain complementary components:

- `N_global`: stable population-level normality;
- `N_personal`: user-specific benign patterns learned after deployment;
- optional `A_safety`: protected safety-critical abnormal anchors/regions.

The anomaly score can combine global and personal evidence while retaining explicit safety separation.

### Evidence needed

- personalized HAR/domain-adaptation methods with global/personal decomposition;
- mixture/prototype anomaly models;
- personalized anomaly detection with subject-specific representations.

**Current priority: medium-high as a possible architecture supporting G1.**

---

## G4. Deployment-realistic longitudinal benchmark protocol

### Hypothesis

Current anomaly benchmarks may not evaluate the actual loop:

`population training -> unseen subject -> false alarms -> sparse feedback -> update -> continued stream -> repeated updates`

A rigorous subject-wise temporal protocol could expose failure modes hidden by static AUROC evaluation.

### Required metrics

- personal-normal false-positive rate;
- safety-critical anomaly recall/sensitivity;
- AUROC/AUPRC;
- false alarms per unit time where possible;
- retention of global normality;
- performance after each update session;
- feedback interactions required;
- unsafe-normalization/contamination rate.

A protocol alone is unlikely to constitute a top-tier contribution unless paired with a strong dataset insight or algorithm.

**Current priority: high as experimental infrastructure, not necessarily the main novelty.**

---

## G5. Resource-aware feedback memory

Potential later direction: retain only representative temporal segments/embeddings from confirmed false alarms under a strict byte budget.

The first-stage research permits server-side updates, so resource-aware/on-device constraints should remain secondary until the algorithmic personalization problem is established.

**Current priority: later extension.**

---

## Updated novelty boundaries after Phase 2

Do **not** claim novelty from any of the following alone:

- continual anomaly detection;
- online anomaly detection;
- concept drift adaptation;
- human-in-the-loop anomaly detection;
- using false alarms to update an anomaly detector;
- continual prototype updates;
- reducing false alarms in online VAD;
- skeleton/video + continual learning.

The current research hypothesis is narrower:

> **Safety-constrained, subject-specific continual normality personalization from sparse human feedback.**

This is still not a final novelty claim. Phase 3 must specifically search for prior work that could invalidate the safety/personalization component before algorithm design begins.
