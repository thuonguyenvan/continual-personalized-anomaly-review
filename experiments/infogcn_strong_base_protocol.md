# Strong Base Audit: Official InfoGCN

## Purpose

This stage is not a backbone contribution. It replaces the earlier diagnostic GRU/compact-ST-GCN encoders with an established strong skeleton representation model before continual-personalization experiments.

The architecture is loaded from the official CVPR 2022 InfoGCN repository (`stnoah1/infogcn`) pinned to commit `873feaa85160317335a83e04013e0ffa3f63525e`. The official paper/repository reports 89.8% NTU RGB+D 120 cross-subject action-recognition accuracy under its standard 120-class protocol. Our experiment deliberately does **not** use an official 120-class pretrained checkpoint because that would expose A42/A43 during representation training.

## Leakage-safe adaptation of the official model

The official architecture and InfoGCN training objective are retained, but the data protocol is replaced by the project's subject-disjoint normal-only protocol:

- train: `encoder_train` + `global_normal` only;
- checkpoint selection: `encoder_val` + `global_normal` only;
- detector threshold: `detector_calib` + `global_normal` only;
- retention audit: `retention_val` + `global_normal` only;
- deployment evaluation: A42 candidate personal normal and A43 protected anomaly.

A42 and A43 must never be used for training, checkpoint selection, detector hyperparameter selection, or threshold calibration.

Because the project manifest reader currently selects one primary body, the adapter instantiates InfoGCN with `num_person=1`. Inputs are reshaped from `[N,T,V,C]` to the official model interface `[N,C,T,V,M]`. The default preprocessing is `sequence_origin`, which preserves displacement relative to the first frame. This is an adapter choice required by the custom split and should be reported separately from claims about the original InfoGCN benchmark recipe.

## Training objective

The adapter follows the official InfoGCN loss form:

`CE + lambda_2 * MMD + lambda_1 * ||mean(z)||_2`

with default repository hyperparameters `lambda_1=1e-4`, `lambda_2=1e-1`, `base_lr=0.1`, SGD+Nesterov, weight decay `5e-4`, warm-up 5 epochs, and LR steps at 90/100 for 110 epochs.

The number of classes is 24 because only the predefined global-normal actions participate in representation training.

## Decision rule

After training, extract InfoGCN latent `z` for all experiment-relevant samples and run the existing detector audit. This is a diagnostic comparison only. A43 remains evaluation-only; do not choose OCSVM/kNN/KMeans hyperparameters by A43 recall.

If a pre-specified detector on InfoGCN embeddings yields adequate retention FPR, nontrivial A42 false positives, and materially stronger A43 recall, freeze that base protocol and move to continual personalization. If not, do not modify InfoGCN architecture as the primary research direction; instead consider another established strong base model such as CTR-GCN under the same normal-only protocol.
