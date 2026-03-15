#!/usr/bin/env python3
"""
Export /gps/imu (sensor_msgs/msg/Imu) from a ROS2 bag into CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def open_reader(bag_path: Path) -> SequentialReader:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def main() -> None:
    ap = argparse.ArgumentParser(description="Export /gps/imu from a rosbag2 directory")
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--topic", default="/gps/imu", help="IMU topic to export")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    reader = open_reader(bag_path)
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in topics:
        available = ", ".join(sorted(topics))
        raise RuntimeError(f"Topic not found: {args.topic}. Available topics: {available}")

    msg_type = get_message(topics[args.topic])

    rows = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != args.topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = to_sec(msg.header.stamp)
        rows.append(
            [
                f"{t:.9f}",
                f"{msg.orientation.x:.9f}",
                f"{msg.orientation.y:.9f}",
                f"{msg.orientation.z:.9f}",
                f"{msg.orientation.w:.9f}",
                f"{msg.angular_velocity.x:.9f}",
                f"{msg.angular_velocity.y:.9f}",
                f"{msg.angular_velocity.z:.9f}",
                f"{msg.linear_acceleration.x:.9f}",
                f"{msg.linear_acceleration.y:.9f}",
                f"{msg.linear_acceleration.z:.9f}",
                f"{msg.orientation_covariance[0]:.9f}",
                f"{msg.orientation_covariance[4]:.9f}",
                f"{msg.orientation_covariance[8]:.9f}",
                f"{msg.angular_velocity_covariance[0]:.9f}",
                f"{msg.angular_velocity_covariance[4]:.9f}",
                f"{msg.angular_velocity_covariance[8]:.9f}",
                f"{msg.linear_acceleration_covariance[0]:.9f}",
                f"{msg.linear_acceleration_covariance[4]:.9f}",
                f"{msg.linear_acceleration_covariance[8]:.9f}",
            ]
        )

    if not rows:
        raise RuntimeError(f"No messages found on topic {args.topic}")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "qx",
                "qy",
                "qz",
                "qw",
                "wx",
                "wy",
                "wz",
                "ax",
                "ay",
                "az",
                "cov_qx",
                "cov_qy",
                "cov_qz",
                "cov_wx",
                "cov_wy",
                "cov_wz",
                "cov_ax",
                "cov_ay",
                "cov_az",
            ]
        )
        w.writerows(rows)

    print(f"[export_gps_imu_from_bag] topic={args.topic} rows={len(rows)} out={out_csv}")


if __name__ == "__main__":
    main()
