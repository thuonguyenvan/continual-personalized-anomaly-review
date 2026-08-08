# Working Taxonomy

The review is organized around the mechanisms needed to solve continual personalization of one-class anomaly detection.

## T1. Static one-class / anomaly detection

Purpose: define normality and anomaly scores before considering temporal updates.

Subthemes:
- hypersphere / one-class objectives;
- reconstruction and prediction;
- memory/prototype-based normality;
- density and distance-based approaches;
- representation learning for anomaly detection.

## T2. Video and skeleton anomaly detection

Purpose: understand spatio-temporal representations and evaluation for human behavior.

Subthemes:
- video anomaly detection;
- pose/skeleton-based anomaly detection;
- temporal modeling;
- normal-only video learning;
- fall and safety-critical behavior detection.

## T3. Continual / online anomaly detection

Purpose: study repeated updates under sequential distribution change.

Subthemes:
- catastrophic forgetting;
- replay/memory;
- parameter regularization;
- dynamic model expansion;
- continual anomaly benchmarks.

## T4. Evolving normality and concept drift

Purpose: model the fact that the definition/distribution of normal changes over time.

Subthemes:
- streaming anomaly detection;
- adaptive boundaries;
- drift detection;
- non-stationary normal distributions;
- delayed labels and uncertain updates.

## T5. Personalization / subject adaptation

Purpose: distinguish population-level normality from user-specific normality.

Subthemes:
- subject-specific adaptation;
- personalized activity recognition;
- global-versus-personal representation;
- domain/subject shift;
- long-term personalization.

## T6. Human feedback and feedback-efficient adaptation

Purpose: exploit sparse caregiver confirmation rather than requiring dense labels.

Subthemes:
- human-in-the-loop learning;
- active learning;
- interactive anomaly detection;
- sparse feedback;
- query selection and feedback budgets.

## T7. Safe continual normality expansion

This is the current candidate gap area.

Key concerns:
- false-normal contamination;
- unsafe decision-boundary expansion;
- preserving critical anomaly sensitivity;
- uncertainty-aware update gating;
- global-normal knowledge preservation;
- rollback or conservative adaptation.

## T8. Resource-aware / on-device continual learning

Future extension rather than first-stage requirement.

Subthemes:
- compressed replay;
- latent replay;
- parameter-efficient updates;
- memory/FLOP/latency constraints;
- local update on edge devices.
