# KF-GINS Script Index

This directory is kept flat on purpose. File names are grouped by prefix so usage is easier to scan without subfolders.

## Mainline

1. `export_measurements_from_bag.py`
   Export KF-GINS-compatible `imu.csv` and `gnss.csv` from bag data.
2. `export_gps_imu_from_bag.py`
   Export `/gps/imu` for heading/reference use.
3. `analyze_replay_nav_vs_reference.py`
   Main replay-vs-reference evaluator.
4. `tune_csv_replay.py`
   Main replay tuning entry.
5. `analyze_all.py`
   High-level bag analysis orchestrator.

## Prefix Guide

1. `analyze_*.py`
   Offline analysis and evaluation scripts.
2. `export_*.py`
   Data export scripts.
3. `plot_*.py`
   Visualization scripts.
4. `bag_*.py`
   Bag inspection / creation / merge helpers.
5. `ai_gate_*.py`
   AI-assisted GNSS gating research scripts.
6. `ref_*.py`
   Reference-trajectory conversion scripts.
7. `config_*.py`
   Config generation helpers.
8. `tune_*.py` / `tune_*.sh`
   Parameter tuning scripts.

## Current File Roles

### Analysis
1. `analyze_all.py`
2. `analyze_flatness.py`
3. `analyze_navresult_modern.py`
4. `analyze_odom_comparison.py`
5. `analyze_odom_vs_gps.py`
6. `analyze_odom_vs_truth.py`
7. `analyze_replay_nav_vs_reference.py`
8. `analyze_shm.py`

### Export
1. `export_gps_imu_from_bag.py`
2. `export_measurements_from_bag.py`
3. `export_odom_to_tum.py`

### Plot
1. `plot_altitude_heatmap.py`
2. `plot_navresult.py`
3. `plot_odom_live.py`
4. `plot_odom_vs_gps_live.py`
5. `plot_path_from_bag.py`

### Bag tools
1. `bag_check_timeline.py`
2. `bag_make_ros2.py`
3. `bag_merge_ros2.py`
4. `bag_merge_ros2_sqlite.py`

### AI gate
1. `ai_gate_build_dataset.py`
2. `ai_gate_build_features.py`
3. `ai_gate_train_model.py`
4. `ai_gate_eval_model.py`

### Reference / config / tuning
1. `ref_convert_inspvax_bag.py`
2. `config_generate_urbannav.py`
3. `tune_csv_replay.py`
4. `tune_run_ab_xyz_focus.sh`
