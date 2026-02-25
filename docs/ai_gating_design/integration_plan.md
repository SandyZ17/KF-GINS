# Integration Plan (Draft v1)

## Scope

Integrate AI gating with minimal changes to current KF-GINS code.

## Phase 0: Offline only (recommended first)

No runtime inference yet.

### Tasks

1. Export training dataset from existing bags:
   - GNSS features
   - NIS
   - odom states
   - truth-aligned XY error (offline label)
2. Train model offline
3. Evaluate vs rule-based gate baseline

### Success Criteria

- Better GNSS reliability classification than hand thresholds
- Offline ablation shows reduced XY error or reduced outlier updates

## Phase 1: Runtime inference (policy assist)

Add lightweight runtime inference wrapper.

### Integration point

`kf_gins_node.cpp` in `gnssCallback()` after basic parsing and before queue push:

1. Rule pre-gate (keep existing)
2. Build feature vector
3. Run AI model (if enabled)
4. Apply policy:
   - reject OR
   - set `R` scales / mode flags for this GNSS sample
5. Push to queue

## Data Structure Suggestion

Extend `GnssSample` with optional policy metadata:

- `bool ai_gate_used`
- `float ai_reliability_score`
- `float ai_r_scale_xy`
- `float ai_r_scale_z`
- `int ai_mode_override`

This keeps GIEngine changes small.

## Config Parameters (proposed)

```yaml
ai_gate_enable: false
ai_gate_model_path: ""
ai_gate_policy_mode: "score_to_r"   # score_to_r | hard_gate | hybrid
ai_gate_reject_threshold: 0.2
ai_gate_xy_only_threshold: 0.4
ai_gate_log_debug: false
```

## Logging / Evaluation

Add optional topics or logs:

- `/kf_gins/ai_gate_score`
- `/kf_gins/ai_gate_decision`
- acceptance ratio stats

This is critical for debugging and paper plots.

## Paper-Oriented Ablation Matrix

Rows:

- ESKF
- UKF
- SR-UKF

Columns:

- No gate
- Rule pre-gate
- NIS gate
- Pre+NIS
- AI+NIS
- AI+AdaptiveR+NIS

Metrics:

- `RMSE_xy`, `P95_xy`, `Max_xy`
- NIS stats
- update acceptance ratio
- runtime cost (ms/update)

## Risks and Mitigations

### Risk: AI overfits one dataset

Mitigation:

- Train on UrbanNav + M2DGR mixed
- Cross-scenario validation

### Risk: Runtime complexity too high

Mitigation:

- Use tree model / tiny MLP
- Run at GNSS rate only

### Risk: Hard to interpret

Mitigation:

- Keep rule pre-gate + NIS gate as baseline
- Report feature importance / SHAP (offline)
