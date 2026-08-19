# STG-NF strong population detector audit

Purpose: establish a strong published skeleton/pose anomaly detector before any continual-personalization contribution is evaluated.

## Method provenance

- Base method: official `orhir/STG-NF` implementation of *Normalizing Flows for Human Pose Anomaly Detection* (ICCV 2023).
- Pinned upstream commit: `edb5f3220332e160e4d20ce258787d5e2d7e0200`.
- The published STG-NF architecture and normalizing-flow NLL objective are not modified.

## NTU120 adapter

STG-NF is a 2-D pose anomaly detector, whereas NTU RGB+D provides XYZ skeleton coordinates. For this exact-method audit, the project performs a deterministic global XY projection (`XYZ -> XY`) and keeps the NTU vertical Y axis. This is an input/dataset adapter, not a new architecture. The official STG-NF graph implementation already contains the `ntu-rgb+d` 25-joint topology; the adapter redirects the model's graph constructor to that official layout.

This 2-D projection limitation must be reported when interpreting the result. A poor result does not establish that all strong skeleton anomaly detectors fail on NTU120 3-D skeletons.

## Locked protocol

Training uses only:

- `inner_split=encoder_train`
- `role=global_normal`

Checkpoint selection uses only mean NLL on:

- `inner_split=encoder_val`
- `role=global_normal`

Threshold calibration uses the 95th percentile of NLL on:

- `inner_split=detector_calib`
- `role=global_normal`

Session-0 evaluation reports:

- retention FPR on `retention_val/global_normal`
- personal-normal FPR on `deployment_test/candidate_personal_normal` (A42)
- protected safety-anomaly recall on `deployment_test/protected_anomaly` (A43)

A42 and A43 are not used for model fitting or checkpoint selection. A43 is not used for threshold selection or hyperparameter tuning.

## Default upstream-style settings

- segment length: 24
- batch size: 256
- epochs: 8
- optimizer: Adamax
- learning rate: 5e-4
- exponential LR decay: 0.99
- K: 8
- L: 1
- hidden dimension: 0
- affine coupling
- uniform adjacency strategy
- max hops: 8

The project uses seed 1337 for deterministic protocol integration.

## Decision rule

This audit is not a hyperparameter sweep. Run the locked configuration once. If Session-0 is strong enough to serve as a population detector and still produces a meaningful personal false-positive problem, freeze it and move to continual-personalization experiments. If it is unsuitable, move to another published anomaly detector rather than modifying STG-NF into a new backbone contribution.
