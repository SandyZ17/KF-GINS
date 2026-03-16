# Related Work Notes: GNSS/INS + Learning (2024+)

## Current target direction

Working direction for this project:

- Error-state EKF / KF-GINS remains the main estimator
- A lightweight learning module estimates a low-dimensional GNSS disturbance state
- The disturbance is used to correct the GNSS update term in real time
- Focus is on urban degradation, not full GNSS outage

This note tracks recent work after 2024 that is close to this direction and clarifies where the novelty boundary still exists.

## Summary judgment

What is already common:

- neural network for adaptive `Q/R` tuning
- neural network for pseudo-observation generation during GNSS outage
- learning-assisted GNSS quality/NLOS detection
- deep learning aided GNSS urban positioning

What still appears less saturated:

- explicit low-dimensional GNSS disturbance state modeling
- structurally embedding the learned disturbance into the GNSS update equation of an ESKF
- real-time, small-data, physics-dominant design for urban degradation rather than outage replacement

## Most relevant papers

### 1. pyrtklib: An open-source package for tightly coupled deep learning and GNSS integration for positioning in urban canyons

- Year: 2024
- Venue/status: arXiv preprint
- Link: https://arxiv.org/abs/2409.12996
- Why relevant:
  - directly targets urban canyon GNSS positioning
  - combines deep learning with GNSS integration
  - useful as a nearby reference for "learning + GNSS in urban canyon"
- Difference from current direction:
  - appears more like a deep-learning-enhanced GNSS integration framework/toolchain
  - not clearly the same as "low-dimensional GNSS disturbance state inside ESKF update"

### 2. Sky-GVIO: an enhanced GNSS/INS/Vision navigation with FCN-based sky-segmentation in urban canyon

- Year: 2024
- Venue/status: preprint / later journal version
- Preprint index: https://www.alphaxiv.org/abs/2404.11070
- Search/index page: https://www.catalyzex.com/paper/sky-gvio-an-enhanced-gnss-ins-vision
- Why relevant:
  - urban canyon
  - GNSS/INS hybrid system
  - learning is used to improve measurement reliability
- Difference from current direction:
  - uses vision/sky segmentation
  - more about NLOS/environment perception
  - not centered on a learned GNSS disturbance state in an ESKF

### 3. A Method for Assisting GNSS/INS Integrated Navigation System during GNSS Outage Based on CNN-GRU and Factor Graph

- Year: 2024
- Venue/status: Applied Sciences
- Link: https://doi.org/10.3390/app14188131
- Why relevant:
  - learning-assisted GNSS/INS
  - explicit outage assistance
- Difference from current direction:
  - focuses on GNSS outage/interruption
  - uses factor graph instead of KF-GINS style error-state filtering
  - current target is urban degradation with GNSS still available

### 4. Artificial neural network based on strong track and square root UKF for INS/GNSS intelligence integrated system during GPS outage

- Year: 2024
- Venue/status: Scientific Reports
- Link: https://www.nature.com/articles/s41598-024-64918-4
- Why relevant:
  - combines ANN with a nonlinear Kalman-type filter
  - useful as a reference for "learning + Bayesian estimator"
- Difference from current direction:
  - focuses on outage
  - uses UKF family rather than ESKF
  - not disturbance-state-centric

### 5. RBF Neural Network-Aided Robust Adaptive GNSS/INS Integrated Navigation Algorithm in Urban Environments

- Year: 2025
- Venue/status: Sensors
- Link: https://doi.org/10.3390/s25237286
- Why relevant:
  - one of the closest published papers to the current topic
  - urban GNSS/INS
  - learning-aided robust adaptive filtering
- Difference from current direction:
  - closer to RAKF + pseudo-position increment compensation
  - not clearly formulated as a low-dimensional disturbance state embedded in the measurement equation

### 6. Assessing the robustness of machine learning strategy for GNSS/INS vehicle positioning solutions enhancement

- Year: 2025
- Venue/status: GPS Solutions
- Link: https://doi.org/10.1007/s10291-025-01909-6
- Why relevant:
  - recent GNSS/INS + machine learning paper
  - useful for discussing robustness and training-data sensitivity
- Difference from current direction:
  - more about ML enhancement of positioning solution quality
  - not centered on disturbance-state modeling in the filter

### 7. A Hybrid Algorithm of LSTM and Factor Graph for Improving Combined GNSS/INS Positioning Accuracy during GNSS Interruptions

- Year: 2024
- Venue/status: Sensors
- Link: https://doi.org/10.3390/s24175605
- Why relevant:
  - recent hybrid learning + GNSS/INS work
  - useful for interruption/outage related work discussion
- Difference from current direction:
  - factor graph
  - interruption-focused
  - not real-time low-dimensional disturbance estimation in ESKF

### 8. Intelligent urban GNSS measurement uncertainty prediction by exploring the spatial characteristics with transformer

- Year: 2025
- Venue/status: Applied Soft Computing
- Link: https://doi.org/10.1016/j.asoc.2025.113773
- Why relevant:
  - directly related to urban GNSS measurement uncertainty prediction
  - useful for measurement-side learning references
- Difference from current direction:
  - focuses on measurement uncertainty / pseudorange error prediction
  - not specifically GNSS/INS ESKF fusion with a disturbance state

## How these papers group

### Group A: Adaptive noise / robust weighting

Representative ideas:

- neural network predicts `Q`, `R`, or confidence/weight
- robust adaptive Kalman filtering

Closest papers:

- RBF Neural Network-Aided Robust Adaptive GNSS/INS Integrated Navigation Algorithm in Urban Environments

Difference from current direction:

- current target is not just tuning covariance
- current target is estimating a physically interpretable disturbance term

### Group B: Outage assistance / pseudo-observation generation

Representative ideas:

- use learning to replace GNSS updates during outage
- predict pseudo-position or state increments

Closest papers:

- CNN-GRU + Factor Graph during GNSS Outage
- ANN + ST-SR-UKF during GPS Outage
- robustness of ML strategy for GNSS/INS enhancement

Difference from current direction:

- current target focuses on urban degradation when GNSS is still present but biased
- current target does not mainly solve "full outage replacement"

### Group C: Measurement quality / NLOS / uncertainty prediction

Representative ideas:

- learning to infer GNSS measurement quality
- learning to identify NLOS or sky visibility
- learning uncertainty in urban scenes

Closest papers:

- Sky-GVIO
- Intelligent urban GNSS measurement uncertainty prediction by transformer

Difference from current direction:

- current target aims to place a learned disturbance state inside the GNSS update, not only classify or weight observations

### Group D: Deep-learning-aided GNSS integration frameworks

Representative ideas:

- broader deep learning + GNSS integration systems and toolchains

Closest papers:

- pyrtklib

Difference from current direction:

- current target is narrower and more filter-structure-specific

## Possible novelty statement for the current project

Avoid weak framing like:

- "we use a neural network to improve EKF"
- "we adaptively tune measurement noise"

Prefer framing like:

- "We introduce a learning-driven low-dimensional GNSS disturbance state for urban degradation mitigation in real-time error-state GNSS/INS integration."
- "Instead of only down-weighting unreliable GNSS observations, the proposed method explicitly estimates a structured GNSS disturbance term and injects it into the ESKF measurement update."
- "The method is designed for small-data, real-time operation, with learning restricted to the hard-to-model disturbance component while keeping the physics-based estimator unchanged."

## Recommended related-work positioning in a paper

Suggested sequence:

1. Classical robust GNSS/INS filtering
2. Learning-based adaptive covariance / weighting methods
3. Learning-based GNSS outage compensation methods
4. Learning-based urban GNSS quality/NLOS/uncertainty prediction methods
5. Gap statement:
   existing works either tune filter statistics, replace updates during outage, or estimate measurement quality, while fewer works explicitly model a low-dimensional GNSS disturbance state and structurally couple it with a real-time ESKF update under urban degradation

## Current working claim

Safe claim:

- There are many recent papers after 2024 on learning-assisted GNSS/INS, urban GNSS uncertainty, and outage compensation.
- However, a paper exactly matching the current formulation,
  "ESKF main filter + learned low-dimensional GNSS disturbance state + structured insertion into GNSS update + real-time/small-data emphasis",
  has not yet been clearly identified in this search round.

Conservative wording for later use:

- "To the best of our current survey, closely related directions exist, but an identical formulation has not been clearly found."

## Caution

- Do not overclaim novelty before a stricter database search on IEEE Xplore, Scopus, Web of Science, and Google Scholar.
- Before paper submission, rerun a dedicated related-work review with exact keywords around:
  - GNSS disturbance state
  - GNSS bias learning
  - urban degradation
  - learning-aided ESKF
  - measurement residual correction
