#!/usr/bin/env bash
set -euo pipefail

WS_DIR="/home/sandyz/Documents/navigation/kf_gins_ws"
BASE_PARAMS="${WS_DIR}/src/KF-GINS/config/params_hangzhou_town_AB_xyz.yaml"
IMU_CSV="${WS_DIR}/src/KF-GINS/results/A_B_xyz_measurements_latest/imu.csv"
GNSS_CSV="${WS_DIR}/src/KF-GINS/results/A_B_xyz_measurements_latest/gnss.csv"
OUT_DIR="${WS_DIR}/src/KF-GINS/results/A_B_xyz_auto_tune_focus_$(date +%Y%m%d_%H%M%S)"

cd "${WS_DIR}"
set +u
source install/setup.bash
set -u

python3 src/KF-GINS/scripts/tune_csv_replay.py \
  --base-params "${BASE_PARAMS}" \
  --imu-csv "${IMU_CSV}" \
  --gnss-csv "${GNSS_CSV}" \
  --out-dir "${OUT_DIR}" \
  --offset-grid=0.08,0.10,0.12,0.15,0.18 \
  --std-xy-grid=1.0 \
  --std-z-grid=1.0 \
  --initpos-xy-grid=1.5,2.0,2.5,3.0 \
  --initpos-z-grid=3.0,4.0,5.0,6.0 \
  --initvel-grid=0.5,0.8,1.0,1.2

echo
echo "Tuning output:"
echo "  ${OUT_DIR}"
echo
echo "Check these files first:"
echo "  ${OUT_DIR}/summary.csv"
echo "  ${OUT_DIR}/best_params.yaml"
echo "  ${OUT_DIR}/best_nav_trajectory.svg"
echo "  ${OUT_DIR}/best_nav_vs_gnss_trajectory.svg"
