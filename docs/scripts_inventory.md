# Scripts Inventory

This file is the current script inventory for:

1. `src/KF-GINS/scripts`
2. `src/imu_lstm_experiment/scripts`

The purpose is not to document every line of behavior. The goal is to define:

1. which scripts are the mainline tools
2. which scripts are auxiliary utilities
3. which scripts are legacy / transitional and should not be treated as the primary workflow

## Recommended Mainline

### KF-GINS mainline

1. `export_measurements_from_bag.py`
   Purpose: export `imu.csv` and `gnss.csv` from ROS2 bags with KF-GINS-compatible preprocessing.
   Use when: building replay inputs from raw bag data.

2. `export_gps_imu_from_bag.py`
   Purpose: export `/gps/imu` to CSV.
   Use when: building heading/reference data from raw bag data.

3. `analyze_replay_nav_vs_reference.py`
   Purpose: compare `replay_nav.csv` against a unified `reference.csv`.
   Use when: evaluating replay results against a reference trajectory.
   Notes: this is the preferred replay analysis script for `kf_gins_demo`, `test_demo`, and similar replay result folders.

4. `tune_csv_replay.py`
   Purpose: tune KF-GINS replay parameters using exported `imu.csv` and `gnss.csv`.
   Use when: doing parameter search on replay.
   Notes: use only after the replay evaluation metric is clearly defined.

5. `analyze_all.py`
   Purpose: unified entry for bag-level analysis.
   Use when: generating multiple outputs for a bag in one run.
   Notes: this is more of a high-level orchestration script than a single-purpose tool.

### IMU-LSTM mainline

1. `build_reference_from_gps_fix_heading.py`
   Purpose: build a unified `reference.csv` from RTK `gnss.csv` plus heading-only `/gps/imu`.
   Use when: current reference source is RTK position + dual-antenna heading.

2. `prepare_bias_dataset.py`
   Purpose: build train/val/test dataset splits for IMU bias learning.
   Use when: preparing supervised training data.
   Notes: supports `split.mode: ratio` and `split.mode: intervals`. Prefer `intervals` for real experiments.

3. `train_bias_lstm.py`
   Purpose: train the main LSTM bias model.
   Use when: training the current mainline model.
   Notes: includes early stopping, LR scheduler, `best_model.pt`, `last_model.pt`, and `train_summary.json`.

4. `run_lstm_ukf_experiment.py`
   Purpose: run baseline vs corrected IMU evaluation with the Python UKF.
   Use when: doing algorithm experiments on the IMU-LSTM pipeline.
   Notes: current version is test-split oriented and should be treated as the research experiment runner, not the production KF-GINS evaluator.

## KF-GINS Utility Scripts

1. `analyze_odom_vs_gps.py`
   Purpose: compare odometry against GPS.

2. `analyze_odom_vs_truth.py`
   Purpose: compare odometry against truth/reference.

3. `analyze_odom_comparison.py`
   Purpose: compare two odometry topics.

4. `analyze_flatness.py`
   Purpose: analyze flatness-related trajectory behavior.

5. `analyze_shm.py`
   Purpose: analyze SHM-related outputs.

6. `plot_altitude_heatmap.py`
   Purpose: altitude heatmap plotting.

7. `plot_navresult.py`
   Purpose: legacy navresult plotting helper.

8. `plot_odom_live.py`
   Purpose: live odometry plotting.

9. `plot_odom_vs_gps_live.py`
   Purpose: live odometry vs GPS plotting.

10. `plot_path_from_bag.py`
    Purpose: quick bag path plotting.

11. `bag_check_timeline.py`
    Purpose: inspect timing consistency in bag files.

12. `ref_convert_inspvax_bag.py`
    Purpose: convert INS/PVA-like bag outputs into unified reference format.
    Use when: a commercial/high-grade reference system is available in bag form.

13. `export_odom_to_tum.py`
    Purpose: export odometry into TUM format for external evaluation tools.

14. `config_generate_urbannav.py`
    Purpose: generate UrbanNav-related configuration.

15. `bag_make_ros2.py`
    Purpose: helper for bag creation.

16. `bag_merge_ros2.py`
    Purpose: merge bag directories.

17. `bag_merge_ros2_sqlite.py`
    Purpose: merge bag SQLite storage.

## KF-GINS AI-Gate Scripts

These are a separate line of work and should not be mixed with the IMU-LSTM pipeline.

1. `ai_gate_build_dataset.py`
2. `ai_gate_build_features.py`
3. `ai_gate_train_model.py`
4. `ai_gate_eval_model.py`

Status:

1. separate research direction
2. not part of the main KF-GINS replay baseline
3. keep isolated from IMU-LSTM experiments

## IMU-LSTM Utility / Transitional Scripts

1. `common.py`
   Purpose: shared helpers for configs, normalization, IO, seeds.
   Status: required support module.

2. `convert_public_truth_to_reference.py`
   Purpose: convert public truth files into unified `reference.csv`.
   Status: useful converter.

3. `merge_bias_dataset_dirs.py`
   Purpose: merge multiple prepared dataset directories.
   Status: useful when training across multiple bags.

4. `benchmark_inference.py`
   Purpose: benchmark LSTM inference latency.
   Status: useful for deployment/performance checks.

5. `run_alpha_sweep.py`
   Purpose: sweep the IMU correction alpha parameter.
   Status: tuning helper, not part of the minimal core workflow.

## IMU-LSTM Legacy / Transitional Scripts

These scripts were created during the early scaffold phase. They still work as utilities, but they are not the preferred mainline anymore.

1. `legacy/build_supervised_dataset.py`
   Status: early scaffold version.
   Preferred replacement: `prepare_bias_dataset.py`

2. `legacy/train_lstm_imu_comp.py`
   Status: early scaffold trainer.
   Preferred replacement: `train_bias_lstm.py`

3. `legacy/run_lstm_imu_comp.py`
   Status: early scaffold inference script.
   Preferred replacement: `run_lstm_ukf_experiment.py` for algorithm experiments.

4. `legacy/replay_with_corrected_imu.py`
   Status: replay bridge helper from the early scaffold.
   Keep only if you still need the older corrected-CSV replay path.

## Practical Recommendation

If you want a clean working set, focus on these scripts first.

### KF-GINS

1. `export_measurements_from_bag.py`
2. `export_gps_imu_from_bag.py`
3. `analyze_replay_nav_vs_reference.py`
4. `tune_csv_replay.py`

### IMU-LSTM

1. `build_reference_from_gps_fix_heading.py`
2. `prepare_bias_dataset.py`
3. `train_bias_lstm.py`
4. `run_lstm_ukf_experiment.py`

## Suggested Next Cleanup

If we do a real cleanup later, the least risky plan is:

1. move legacy IMU-LSTM scripts into `src/imu_lstm_experiment/scripts/legacy/`
2. keep KF-GINS AI-gate scripts under a separate subgroup or document them as a separate pipeline
3. leave the current mainline scripts in place to avoid breaking existing paths
