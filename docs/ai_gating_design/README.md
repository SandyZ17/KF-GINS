# AI-Assisted Robust Gating Design (Draft v1)

## Goal
Design an **AI-assisted gating and covariance adaptation layer** for the current KF-GINS pipeline, targeting:

- Urban canyon / blockage
- Indoor-outdoor transition
- Tunnel entrance/exit
- Ground vehicle (`XY`-dominant) navigation

This layer does **not** replace the filter (`ESKF/UKF/SR_UKF`); it assists:

- GNSS pre-gate
- NIS gate
- Measurement noise scaling (`R`)
- Update mode switching (`xyz` / `xy`)

## Core Idea
Use a lightweight ML model to estimate **measurement reliability** from online features, then control fusion strategy.

### Inputs (runtime features)
- GNSS quality features (covariance, status, jump speed)
- Filter consistency features (`NIS`, innovation norm)
- Motion features (IMU dynamics, speed)
- Optional VO/LiDAR quality features (future)

### Outputs (control variables)
- `gate_prob` or `reliability_score` in `[0, 1]`
- `R_scale_xy`, `R_scale_z`
- `suggested_mode` (`xy` / `xyz`)
- Optional `hard_reject` flag

## Proposed Architecture
1. **Pre-Gate (Rule-based)**
- Remove obvious invalid GNSS:
  - invalid status
  - absurd covariance
  - jump speed too large

2. **AI Reliability Estimator**
- Predict reliability score from filtered feature vector
- Low compute, runs at GNSS rate (1-30 Hz)

3. **Fusion Policy**
- If `score < T_reject`: reject update
- Else apply update with adaptive noise:
  - `R_xy *= f(score)`
  - `R_z *= g(score)` (or disable Z)
- Optional mode switch:
  - `xy` for degraded scenes
  - `xyz` for open-sky

4. **Filter Update**
- Existing `ESKF/UKF/SR_UKF` stays unchanged except using chosen `R` / mode

## Why This Fits Current Code
Current code already has:
- GNSS pre-gate hooks
- NIS output and NIS gate
- `xyz/xy` GNSS update mode
- Multiple filters (`ESKF/UKF/SR_UKF`)

So AI can be added as a **policy layer**, minimizing risk.

## Suggested Model (v1)
Use a lightweight model first:

- `XGBoost` / `LightGBM` / small `MLP`

Reason:
- Easy training/debugging
- Good tabular performance
- Low inference latency
- Easier ablation than deep sequence models

## Training Targets (v1 options)
### Option A: Binary reliability classification (recommended first)
Label each GNSS update as:
- `reliable`
- `unreliable`

Label source:
- NIS threshold + truth error (offline)
- Example:
  - reliable if `NIS < chi2_95` and `XY error < e_thr`
  - unreliable otherwise

### Option B: Regression for `R_scale`
Predict continuous scaling factor:
- `R_scale_xy`
- `R_scale_z`

This is stronger, but start after Option A.

## Feature Set (v1)
See `feature_spec.yaml`.

## Runtime Integration Plan
See `integration_plan.md`.

## Experiments (minimum set)
1. No gate
2. Rule pre-gate only
3. NIS gate only
4. Pre-gate + NIS gate (current best baseline)
5. AI gate + NIS gate
6. AI gate + adaptive `R` + NIS gate

Report:
- `RMSE_xy`, `P95_xy`, `Max_xy`
- `NIS mean/p95/p99`, over-threshold ratio
- GNSS acceptance ratio
- Runtime overhead

## Dataset Strategy
- **UrbanNav**: urban canyon/tunnel robustness
- **M2DGR**: indoor-outdoor transition and GNSS degradation
- (Future) self-collected city canyon data with labels

## Deliverables (next implementation stage)
- `scripts/build_ai_gate_dataset.py` (offline feature extraction)
- `scripts/train_ai_gate_model.py` (training)
- `scripts/eval_ai_gate_model.py` (offline metrics)
- `src/.../gnss_ai_gate.*` (runtime inference wrapper)
- Config switch:
  - `ai_gate_enable`
  - `ai_gate_model_path`
  - `ai_gate_policy_mode`

