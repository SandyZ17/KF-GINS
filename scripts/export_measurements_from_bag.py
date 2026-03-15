#!/usr/bin/env python3
"""
Export imu.csv / gnss.csv directly from a ROS2 bag using KF-GINS-style preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import yaml
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_node_key(data: Dict[str, Any]) -> str:
    if "kf_gins_node" in data:
        return "kf_gins_node"
    if len(data) == 1:
        return next(iter(data.keys()))
    raise RuntimeError("Cannot detect node key in params yaml")


def to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_reader(bag_path: Path) -> SequentialReader:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def clamp_nonneg(x: float) -> float:
    return x if x > 0.0 else 0.0


def split_text_fields(line: str) -> list[str]:
    return line.replace(",", " ").split()


def read_first_timestamp_from_text(path: Path) -> float:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = split_text_fields(line)
            if not fields:
                continue
            if len(fields) >= 2:
                try:
                    year = int(float(fields[0]))
                    if 1900 <= year <= 2100:
                        return float(fields[1])
                except ValueError:
                    pass
            return float(fields[0])
    raise RuntimeError(f"Cannot read first timestamp from anchor file: {path}")


def infer_default_anchor_file(bag_path: Path) -> Path | None:
    candidates = [
        bag_path.parent / "GNSS-RTK.txt",
        bag_path.parent / "truth.nav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Export KF-GINS measurement CSVs directly from rosbag2")
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--params-file", required=True, help="KF-GINS ROS2 params yaml")
    ap.add_argument("--out-dir", required=True, help="Output directory for imu.csv and gnss.csv")
    ap.add_argument("--imu-topic", default=None, help="Override IMU topic")
    ap.add_argument("--gnss-topic", default=None, help="Override GNSS topic")
    ap.add_argument(
        "--absolute-time-anchor-file",
        default=None,
        help="Optional text/nav file whose first timestamp is used to anchor exported absolute time",
    )
    ap.add_argument(
        "--absolute-time-offset-sec",
        type=float,
        default=None,
        help="Optional manual offset added to exported relative timestamps to produce absolute timestamps",
    )
    args = ap.parse_args()

    params_yaml = load_yaml(Path(args.params_file))
    node_key = detect_node_key(params_yaml)
    params = params_yaml[node_key]["ros__parameters"]

    imu_topic = args.imu_topic or params.get("imu_topic", "/imu/data")
    gnss_topic = args.gnss_topic or params.get("gps_topic", "/gps/fix")
    imu_in_flu = bool(params.get("imu_in_flu", True))
    gnss_time_offset_sec = float(params.get("gnss_time_offset_sec", 0.0))
    use_navsatfix_covariance = bool(params.get("use_navsatfix_covariance", True))
    navsat_cov_scale = float(params.get("navsat_cov_scale", 1.0))
    navsat_cov_min_std_xy = float(params.get("navsat_cov_min_std_xy", 0.0))
    navsat_cov_min_std_z = float(params.get("navsat_cov_min_std_z", 0.0))
    gnss_std = params.get("gnss_std", [1.0, 1.0, 2.0])
    if not isinstance(gnss_std, list) or len(gnss_std) != 3:
        gnss_std = [1.0, 1.0, 2.0]

    bag_path = Path(args.bag)
    out_dir = Path(args.out_dir)
    imu_csv = out_dir / "imu.csv"
    gnss_csv = out_dir / "gnss.csv"
    ensure_parent(imu_csv)
    ensure_parent(gnss_csv)

    reader = open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if imu_topic not in type_map:
        raise RuntimeError(f"IMU topic not found: {imu_topic}")
    if gnss_topic not in type_map:
        raise RuntimeError(f"GNSS topic not found: {gnss_topic}")

    imu_type = get_message(type_map[imu_topic])
    gnss_type = get_message(type_map[gnss_topic])

    imu_rows: List[Dict[str, float]] = []
    gnss_rows: List[Dict[str, float]] = []
    last_imu_time = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == imu_topic:
            msg = deserialize_message(data, imu_type)
            stamp_sec = to_sec(msg.header.stamp)
            dt = 0.0 if last_imu_time is None else max(0.0, stamp_sec - last_imu_time)
            last_imu_time = stamp_sec

            raw_gyro = [float(msg.angular_velocity.x), float(msg.angular_velocity.y), float(msg.angular_velocity.z)]
            raw_acc = [float(msg.linear_acceleration.x), float(msg.linear_acceleration.y), float(msg.linear_acceleration.z)]
            filter_gyro = raw_gyro[:]
            filter_acc = raw_acc[:]
            if imu_in_flu:
                filter_gyro[1] *= -1.0
                filter_gyro[2] *= -1.0
                filter_acc[1] *= -1.0
                filter_acc[2] *= -1.0

            imu_rows.append(
                {
                    "stamp_sec": stamp_sec,
                    "imu_time_sec": stamp_sec,
                    "dt_sec": dt,
                    "raw_gyro_x": raw_gyro[0],
                    "raw_gyro_y": raw_gyro[1],
                    "raw_gyro_z": raw_gyro[2],
                    "raw_acc_x": raw_acc[0],
                    "raw_acc_y": raw_acc[1],
                    "raw_acc_z": raw_acc[2],
                    "filter_gyro_x": filter_gyro[0],
                    "filter_gyro_y": filter_gyro[1],
                    "filter_gyro_z": filter_gyro[2],
                    "filter_acc_x": filter_acc[0],
                    "filter_acc_y": filter_acc[1],
                    "filter_acc_z": filter_acc[2],
                }
            )
        elif topic == gnss_topic:
            msg = deserialize_message(data, gnss_type)
            stamp_sec = to_sec(msg.header.stamp)
            fused_time_sec = stamp_sec + gnss_time_offset_sec
            cov_xx = float(msg.position_covariance[0])
            cov_yy = float(msg.position_covariance[4])
            cov_zz = float(msg.position_covariance[8])
            used_cov = 0
            if use_navsatfix_covariance and int(msg.position_covariance_type) != 0:
                if cov_xx > 0.0 and cov_yy > 0.0 and cov_zz > 0.0:
                    std_x = math.sqrt(cov_xx)
                    std_y = math.sqrt(cov_yy)
                    std_z = math.sqrt(cov_zz)
                    used_cov = 1
                else:
                    std_x, std_y, std_z = gnss_std
            else:
                std_x, std_y, std_z = gnss_std

            std_x = max(std_x, navsat_cov_min_std_xy) * navsat_cov_scale
            std_y = max(std_y, navsat_cov_min_std_xy) * navsat_cov_scale
            std_z = max(std_z, navsat_cov_min_std_z) * navsat_cov_scale

            gnss_rows.append(
                {
                    "stamp_sec": stamp_sec,
                    "fused_time_sec": fused_time_sec,
                    "latitude_deg": float(msg.latitude),
                    "longitude_deg": float(msg.longitude),
                    "altitude_m": float(msg.altitude),
                    "status": int(msg.status.status),
                    "used_covariance": used_cov,
                    "covariance_type": int(msg.position_covariance_type),
                    "cov_xx": clamp_nonneg(cov_xx),
                    "cov_yy": clamp_nonneg(cov_yy),
                    "cov_zz": clamp_nonneg(cov_zz),
                    "std_x": std_x,
                    "std_y": std_y,
                    "std_z": std_z,
                    "pre_gate_rejected": 0,
                }
            )

    absolute_time_offset_sec = 0.0
    anchor_file = None
    if args.absolute_time_offset_sec is not None:
        absolute_time_offset_sec = float(args.absolute_time_offset_sec)
    else:
        if args.absolute_time_anchor_file:
            anchor_file = Path(args.absolute_time_anchor_file)
        else:
            anchor_file = infer_default_anchor_file(bag_path)
        if anchor_file is not None:
            anchor_start_sec = read_first_timestamp_from_text(anchor_file)
            first_rel_stamp_sec = None
            if gnss_rows:
                first_rel_stamp_sec = gnss_rows[0]["stamp_sec"]
            elif imu_rows:
                first_rel_stamp_sec = imu_rows[0]["stamp_sec"]
            if first_rel_stamp_sec is not None:
                absolute_time_offset_sec = anchor_start_sec - first_rel_stamp_sec

    if absolute_time_offset_sec != 0.0:
        for row in imu_rows:
            row["stamp_sec"] += absolute_time_offset_sec
            row["imu_time_sec"] += absolute_time_offset_sec
        for row in gnss_rows:
            row["stamp_sec"] += absolute_time_offset_sec
            row["fused_time_sec"] += absolute_time_offset_sec

    with imu_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stamp_sec",
                "imu_time_sec",
                "dt_sec",
                "raw_gyro_x",
                "raw_gyro_y",
                "raw_gyro_z",
                "raw_acc_x",
                "raw_acc_y",
                "raw_acc_z",
                "filter_gyro_x",
                "filter_gyro_y",
                "filter_gyro_z",
                "filter_acc_x",
                "filter_acc_y",
                "filter_acc_z",
            ]
        )
        for row in imu_rows:
            w.writerow(
                [
                    f"{row['stamp_sec']:.9f}",
                    f"{row['imu_time_sec']:.9f}",
                    f"{row['dt_sec']:.9f}",
                    f"{row['raw_gyro_x']:.9f}",
                    f"{row['raw_gyro_y']:.9f}",
                    f"{row['raw_gyro_z']:.9f}",
                    f"{row['raw_acc_x']:.9f}",
                    f"{row['raw_acc_y']:.9f}",
                    f"{row['raw_acc_z']:.9f}",
                    f"{row['filter_gyro_x']:.9f}",
                    f"{row['filter_gyro_y']:.9f}",
                    f"{row['filter_gyro_z']:.9f}",
                    f"{row['filter_acc_x']:.9f}",
                    f"{row['filter_acc_y']:.9f}",
                    f"{row['filter_acc_z']:.9f}",
                ]
            )

    with gnss_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stamp_sec",
                "fused_time_sec",
                "latitude_deg",
                "longitude_deg",
                "altitude_m",
                "status",
                "used_covariance",
                "covariance_type",
                "cov_xx",
                "cov_yy",
                "cov_zz",
                "std_x",
                "std_y",
                "std_z",
                "pre_gate_rejected",
            ]
        )
        for row in gnss_rows:
            w.writerow(
                [
                    f"{row['stamp_sec']:.9f}",
                    f"{row['fused_time_sec']:.9f}",
                    f"{row['latitude_deg']:.9f}",
                    f"{row['longitude_deg']:.9f}",
                    f"{row['altitude_m']:.9f}",
                    str(int(row["status"])),
                    str(int(row["used_covariance"])),
                    str(int(row["covariance_type"])),
                    f"{row['cov_xx']:.9f}",
                    f"{row['cov_yy']:.9f}",
                    f"{row['cov_zz']:.9f}",
                    f"{row['std_x']:.9f}",
                    f"{row['std_y']:.9f}",
                    f"{row['std_z']:.9f}",
                    str(int(row["pre_gate_rejected"])),
                ]
            )

    print(f"bag={args.bag}")
    print(f"imu_topic={imu_topic}")
    print(f"gnss_topic={gnss_topic}")
    print(f"imu_count={len(imu_rows)}")
    print(f"gnss_count={len(gnss_rows)}")
    print(f"absolute_time_offset_sec={absolute_time_offset_sec:.9f}")
    print(f"absolute_time_anchor_file={anchor_file if anchor_file is not None else ''}")
    print(f"imu_csv={imu_csv}")
    print(f"gnss_csv={gnss_csv}")


if __name__ == "__main__":
    main()
