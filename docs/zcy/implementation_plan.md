# Implementation Plan: Learning-Aided GNSS Disturbance State for KF-GINS

## 1. Objective

Implement a real-time GNSS/INS enhancement method based on:

- original KF-GINS error-state EKF remains unchanged as the main estimator
- a lightweight learning module estimates a 3D GNSS disturbance state
- the disturbance is used to correct the GNSS innovation before EKF update

Target scenario:

- urban degradation
- multipath / blockage / NLOS
- GNSS available but biased

## 2. Overall pipeline

The practical implementation pipeline is:

1. prepare dataset
2. convert dataset into KF-GINS-compatible format
3. run vanilla KF-GINS baseline
4. export training logs from GNSS update steps
5. build supervision labels from reference trajectory
6. train a lightweight disturbance estimator
7. integrate the learned disturbance into `gnssUpdate()`
8. evaluate offline
9. simplify for online real-time inference

## 3. Stage A: Data preparation

### A1. Select dataset

Recommended first dataset:

- UrbanNav Tokyo

Reasons:

- has IMU
- has GNSS raw files
- has reference trajectory
- lower preprocessing cost than Hong Kong ROS bag subset

### A2. Build conversion outputs

KF-GINS needs:

- IMU text file:
  - `time, dtheta_x, dtheta_y, dtheta_z, dvel_x, dvel_y, dvel_z`
- GNSS text file:
  - `time, lat, lon, h, std_n, std_e, std_d`
- reference trajectory file for evaluation

Required task:

- write a converter from UrbanNav raw format to KF-GINS plain-text format

### A3. Reference coordinate alignment

Need to ensure:

- GNSS time system is consistent with KF-GINS input time
- reference trajectory is aligned to GNSS update epochs
- all position outputs use the same geographic/navigation frame convention

## 4. Stage B: KF-GINS baseline preparation

### B1. Run baseline without learning

Goal:

- obtain baseline navigation performance
- verify dataset conversion is correct

Required outputs:

- navigation result
- IMU error estimate
- state STD output

### B2. Add internal logging

Add a dedicated debug/training log file from the GNSS update stage.

Minimum fields to export at each GNSS update epoch:

- timestamp
- current nominal position
- current nominal velocity
- current GNSS measurement
- GNSS standard deviation
- raw innovation
- corrected antenna position
- innovation after lever-arm compensation
- covariance diagonal summary
- update type flag

Recommended implementation location:

- `KF-GINS/src/kf-gins/gi_engine.cpp`
- `KF-GINS/src/kf_gins.cpp`

Suggested output files:

- `KF_GINS_GNSS_LOG.txt`
- `KF_GINS_FEATURE_LOG.txt`

## 5. Stage C: Disturbance-label construction

### C1. Define disturbance label

Recommended first definition:

- navigation-frame GNSS disturbance vector

\[
\mathbf{d}_{k}^{b,*} = \mathbf{z}_{k}^{g} - \mathbf{z}_{k,\mathrm{ref}}^{g}
\]

Equivalent innovation-space version:

\[
\mathbf{d}_{k}^{b,*} = \mathbf{r}_k - \mathbf{r}_{k,\mathrm{ideal}}
\]

### C2. Build sample-wise training tuples

For each GNSS update epoch, build:

- feature vector `u_k`
- label `d_k^{b,*}`

### C3. First-version feature set

Use only real-time-equivalent low-dimensional features:

- current GNSS innovation `r_k`
- GNSS standard deviations
- current velocity
- short-window IMU statistics
- recent innovation history

Do not use in the first version:

- LiDAR
- camera
- sky image
- map prior
- future information
- long sequence context

## 6. Stage D: Learning model

### D1. First model choice

Use:

- small MLP

Reason:

- low data demand
- easy training
- easy real-time deployment
- easy ablation study

### D2. Input/output

Input:

- `u_k`

Output:

- `\hat{d}_k^b \in R^3`

### D3. Loss

First version:

\[
\mathcal{L} = \left\| \hat{\mathbf{d}}_k^b - \mathbf{d}_{k}^{b,*} \right\|_2^2
+ \lambda \left\| \hat{\mathbf{d}}_k^b \right\|_2^2
\]

Optional later:

- innovation-consistency term
- robust loss

## 7. Stage E: Online integration into KF-GINS

### E1. Integration point

Insert learned disturbance estimation right before GNSS EKF update.

Current update:

- compute GNSS innovation
- call `EKFUpdate(...)`

Modified update:

1. compute raw innovation
2. extract feature vector
3. run lightweight model
4. obtain disturbance estimate
5. correct innovation
6. call `EKFUpdate(...)`

### E2. Code insertion location

Main target function:

- `GIEngine::gnssUpdate(...)`

Core equation:

\[
\tilde{\mathbf{r}}_k = \mathbf{r}_k - \hat{\mathbf{d}}_k^b
\]

Then use:

- `\tilde{r}_k` instead of `r_k`

### E3. Real-time requirement

Online inference must use only:

- current filter state
- current GNSS observation
- short-window historical statistics

No dependence on:

- RINEX file parsing
- future samples
- offline smoothing results

## 8. Stage F: Experiments

### F1. Baselines

At minimum compare with:

- vanilla KF-GINS
- KF-GINS + adaptive noise inflation if available

If time allows:

- a simple heuristic innovation clipping baseline

### F2. Metrics

Use:

- position RMSE
- horizontal error
- vertical error
- maximum error
- robustness under urban degraded segments

### F3. Ablation study

Need at least:

- without learning
- learning disturbance state without IMU-stat features
- learning disturbance state with full first-version features
- different label definitions

## 9. Stage G: Paper structure

Suggested method story:

1. standard ESKF works well in nominal cases but suffers under urban GNSS degradation
2. adaptive covariance methods only change confidence, not the structured bias itself
3. propose a low-dimensional GNSS disturbance state
4. estimate it using a lightweight learning model
5. inject it into the GNSS update of a real-time ESKF

## 10. Immediate next actions

Recommended order of execution:

1. implement UrbanNav-to-KF-GINS converter
2. run baseline KF-GINS on converted data
3. add GNSS update logging
4. generate disturbance labels
5. train first MLP model
6. integrate model into `gnssUpdate()`
7. run ablation and evaluation

## 11. Minimal deliverables for the first working version

The first successful milestone should include:

- converted UrbanNav subset
- runnable KF-GINS baseline on that subset
- GNSS-update feature/label log file
- standalone training script for `R^3` disturbance prediction
- integrated inference path in KF-GINS
- one quantitative comparison against vanilla KF-GINS
