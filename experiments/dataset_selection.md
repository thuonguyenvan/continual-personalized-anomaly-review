# Dataset Selection for Gap Validation

## Decision

Use **NTU RGB+D 120** as the primary controlled public benchmark for the first falsification experiment, then use **ETRI-Activity3D** as an elderly-specific external validation of the personalization side of the problem.

The reason for using two datasets is that no single public dataset found in this audit simultaneously provides all of the following at sufficient scale:

- many individual subjects;
- elderly participants;
- RGB/depth/skeleton modalities;
- diverse normal daily behavior;
- safety-critical abnormal behavior such as falls;
- a natural repeated-feedback personalization protocol.

This mismatch is itself important for later benchmark design, but it should not be used as a novelty claim until the dataset landscape is audited more comprehensively.

## 1. Primary benchmark: NTU RGB+D 120

### Why it is the best first choice

The official NTU RGB+D 120 benchmark contains 114,480 samples from 106 subjects and provides RGB, depth, infrared, and 3D skeleton sequences. Its action taxonomy includes daily actions and medical-condition actions, including **staggering (A42)** and **falling down (A43)**.

This enables a controlled experiment in which both subject-specific normal data and dangerous-event proxies come from the same acquisition ecosystem. That is preferable to mixing a normal-only elderly dataset with a separate fall dataset in the first experiment, because cross-dataset domain shift could otherwise be mistaken for a safety/adaptation effect.

### Proposed one-class split

Do not use the original 120-way classification objective. Re-purpose the dataset as a one-class/personalization benchmark.

#### Global-normal pool

Select benign daily actions such as:

- drink/eat;
- sit down / stand up;
- reading/writing;
- phone/tablet use;
- dressing actions;
- object manipulation;
- other low-risk ADL classes.

Exclude ambiguous medical-condition actions from the normal pool in the first version.

#### Safety-critical anomaly pool

Minimum hard anomaly set:

- A43 falling down.

Additional anomaly-like classes for robustness analysis:

- A42 staggering;
- selected pain/distress or violent-interaction classes only if their semantics fit the research question.

The paper should report the exact semantic policy rather than silently relabeling all medical classes as anomalies.

### Subject-wise continual simulation

Split by subject, not randomly by clip.

1. **Global subjects**: used to train the initial normal model.
2. **Personalization subjects**: completely unseen during global training.
3. For every personalization subject, divide benign clips into chronological pseudo-sessions (or deterministic action/trial sessions if real timestamps are unavailable).
4. At session `t`, reveal only a small subset of benign samples that were initially scored as anomalous and mark them as **human-confirmed normal**.
5. Update the model.
6. Re-evaluate on:
   - remaining benign clips from the same subject;
   - normal clips from global subjects;
   - falling/staggering clips withheld from adaptation.

### Why pseudo-time is acceptable initially

NTU is not a natural longitudinal deployment dataset, so the session order is synthetic. That prevents it from being the final deployment benchmark, but it is sufficient for a **controlled falsification experiment** testing whether adaptation can reduce subject-specific false alarms while degrading anomaly sensitivity.

Run several deterministic/random session orderings and report variance so conclusions are not tied to one artificial order.

## 2. External elderly validation: ETRI-Activity3D

ETRI-Activity3D was introduced at IROS 2020 specifically for elderly-care robots. It contains 112,620 RGB/depth/skeleton samples from 100 subjects performing 55 daily activities and explicitly studies age-related domain differences.

This makes it much more semantically aligned with the target application than NTU.

### Intended role

Use ETRI to answer:

> Does population-level normality produce meaningful subject-specific false alarms in actual elderly activity data, and can sparse personalization reduce them?

Because the dataset does not provide a matched safety-critical abnormal set suitable for our target experiment, it should **not by itself** establish the safety-preservation claim.

Possible protocol:

- train global normality on a subset of elderly subjects;
- deploy to held-out elderly subjects;
- measure per-subject false-positive/anomaly-score heterogeneity;
- personalize with a small confirmed-normal budget;
- quantify reduction in subject-specific false alarms and retention of global normality.

## 3. Toyota Smarthome Untrimmed

Toyota Smarthome contains long, real-world daily-living videos of senior subjects and RGB/depth/skeleton modalities. It is attractive because its videos are much less trimmed and more deployment-like.

However, it lacks a dedicated dangerous-anomaly set and has far fewer subjects than ETRI. Use it later to test temporal/ecological realism rather than as the first benchmark.

## 4. Fall-specific datasets

UR Fall and UP-Fall can be used only as secondary sanity checks. They are substantially smaller and/or involve healthy young participants, and mixing them with elderly-normal datasets introduces domain confounds.

They should not be the main evidence for a top-tier method.

## 5. Recommended staged experiment

### Stage A — controlled falsification

**Dataset:** NTU RGB+D 120 skeleton modality.

Goal:

> Test whether naïve continual personalization produces the hypothesized FPR-vs-safety trade-off under controlled same-domain data.

Baselines:

1. no adaptation;
2. threshold recalibration only;
3. prototype update;
4. naïve fine-tuning / one-class update;
5. replay-preserving update.

Key outputs after every session:

- personal-normal FPR;
- global-normal FPR;
- falling recall/FNR;
- staggering recall/FNR;
- anomaly-score margin;
- forgetting;
- number of confirmed-normal samples consumed.

### Stage B — elderly relevance

**Dataset:** ETRI-Activity3D skeleton modality.

Goal:

> Verify that subject-specific normality shift and feedback-efficient personalization are real in elderly behavior rather than an artifact of NTU actors.

The safety claim remains grounded in Stage A until an elderly dataset with true dangerous behavior is available.

### Stage C — application-level validation

Use the supervisor/company elderly normal and abnormal/fall data, provided appropriate ethics/data permissions exist.

This stage can test the complete target loop:

`global normal model -> elderly user -> false alarm -> caregiver confirmation -> repeated personalization -> true dangerous-event evaluation`

Private data should complement, not replace, public-benchmark evidence.

## 6. Current recommendation on modality

Start with **3D skeleton sequences**, not RGB.

Reasons:

- both NTU and ETRI provide skeleton data;
- it isolates behavior dynamics from appearance/background confounds;
- it makes repeated experiments and continual updates cheaper;
- it is closer to later privacy/edge deployment constraints;
- it makes the first research question about evolving normality rather than about video-backbone capacity.

RGB can be added later as a generalization experiment if the core phenomenon is validated.

## 7. Dataset-induced limitations that must be disclosed

1. NTU participants are not elderly.
2. NTU session order is not naturally longitudinal.
3. Falls are acted rather than clinical events.
4. ETRI provides elderly normal behavior but not the full safety-anomaly spectrum.
5. Cross-dataset anomaly testing should not be used as the primary safety evidence because of domain shift.

These limitations motivate, but do not by themselves prove the novelty of, a future personalized continual anomaly benchmark.

## Decision gate

Proceed with NTU skeleton experiments first **only if** access to the dataset is obtainable under its academic-use agreement. In parallel, request/access ETRI-Activity3D for the elderly-normal validation stage.

If NTU access is delayed, prototype the pipeline on any already-available skeleton action dataset but do not treat that result as publication evidence.
