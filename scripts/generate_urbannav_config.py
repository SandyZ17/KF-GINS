#!/usr/bin/env python3
"""
Generate KF-GINS parameter snippet from UrbanNav extrinsic and IMU noise YAML files.

Input:
- extinct.yaml (UrbanNav extrinsics; OpenCV YAML format)
- imu_noise.yaml (UrbanNav IMU noise summary)

Output:
- ROS2 params YAML under KF-GINS config/

Notes:
- ANTENNA_T_IMU in UrbanNav docs uses x-right, y-forward, z-up.
- KF-GINS antlever expects IMU body frame FRD (x-forward, y-right, z-down).
- IMU noise conversion uses approximate unit mapping; verify against your sensor model.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import yaml


def parse_opencv_matrix_data(text: str, key: str) -> list[float]:
    pat = re.compile(rf"{re.escape(key)}:.*?data:\s*\[(.*?)\]", re.S)
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"Matrix '{key}' not found")
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", m.group(1))
    return [float(x) for x in nums]


def parse_imu_noise_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # OpenCV-style YAML header trips pyyaml; strip the first directive line.
    lines = text.splitlines()
    if lines and lines[0].startswith("%YAML"):
        lines = lines[1:]
    text = "\n".join(lines)
    data = yaml.safe_load(text)
    return data


def rad_sqrt_s_to_deg_sqrt_hr(v: float) -> float:
    return v * 180.0 / math.pi * math.sqrt(3600.0)


def mps2_sqrt_s_to_mps_sqrt_hr(v: float) -> float:
    return v * math.sqrt(3600.0)


def rad_s_drift_to_deg_hr(v: float) -> float:
    # Approximate mapping for bias std if source provides steady-state std in rad/s.
    return v * 180.0 / math.pi * 3600.0


def mps2_to_mgal(v: float) -> float:
    # 1 mGal = 1e-5 m/s^2
    return v * 1e5


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate KF-GINS config from UrbanNav extrinsics/noise")
    ap.add_argument("--extrinsic", required=True, help="UrbanNav extinct.yaml path")
    ap.add_argument("--imu-noise", required=True, help="UrbanNav imu_noise.yaml path")
    ap.add_argument("--out", required=True, help="Output ROS2 params YAML path")
    ap.add_argument("--node-name", default="kf_gins_node", help="ROS2 node key")
    args = ap.parse_args()

    ext_text = Path(args.extrinsic).read_text(encoding="utf-8")
    ant = parse_opencv_matrix_data(ext_text, "ANTENNA_T_IMU")
    if len(ant) != 16:
        raise RuntimeError("ANTENNA_T_IMU matrix is not 4x4")

    # UrbanNav ANTENNA_T_IMU translation convention in comments:
    # x-right, y-forward, z-up
    x_right = ant[3]
    y_forward = ant[7]
    z_up = ant[11]

    # Convert to KF-GINS FRD antlever: x-forward, y-right, z-down
    ant_frd = [y_forward, x_right, -z_up]

    ino = parse_imu_noise_yaml(Path(args.imu_noise))
    gyr = ino["Gyr"]["avg-axis"]
    acc = ino["Acc"]["avg-axis"]

    # Approximate conversions. Validate with dataset docs if available.
    arw = [rad_sqrt_s_to_deg_sqrt_hr(gyr["gyr_n"])] * 3
    vrw = [mps2_sqrt_s_to_mps_sqrt_hr(acc["acc_n"])] * 3
    gbstd = [rad_s_drift_to_deg_hr(gyr["gyr_w"])] * 3
    abstd = [mps2_to_mgal(acc["acc_w"])] * 3

    out = {
        args.node_name: {
            "ros__parameters": {
                "imu_in_flu": True,
                "output_enu": True,
                "antlever": [round(v, 6) for v in ant_frd],
                "imunoise": {
                    "arw": [round(v, 6) for v in arw],
                    "vrw": [round(v, 6) for v in vrw],
                    "gbstd": [round(v, 6) for v in gbstd],
                    "abstd": [round(v, 6) for v in abstd],
                    # Conservative defaults kept for scale and correlation time.
                    "gsstd": [300.0, 300.0, 300.0],
                    "asstd": [300.0, 300.0, 300.0],
                    "corrtime": 4.0,
                },
                "urban_nav_source_notes": {
                    "antenna_t_imu_input_convention": "x-right,y-forward,z-up",
                    "kf_gins_antlever_output_convention": "FRD x-forward,y-right,z-down",
                    "imu_noise_conversion": "approximate_from_imu_noise_yaml_avg_axis_verify_units",
                },
            }
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=False), encoding="utf-8")
    print(f"Generated config: {out_path}")
    print(f"antlever_frd={ant_frd}")
    print(f"arw_deg_sqrt_hr≈{arw[0]:.6f}, vrw_mps_sqrt_hr≈{vrw[0]:.6f}")


if __name__ == "__main__":
    main()

