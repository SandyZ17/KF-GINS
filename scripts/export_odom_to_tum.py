#!/usr/bin/env python3
"""
Export nav_msgs/Odometry from a ROS2 bag to TUM trajectory format.

TUM line format:
  timestamp tx ty tz qx qy qz qw

Example:
  python3 src/KF-GINS/scripts/export_odom_to_tum.py \
    --bag <rosbag2_dir> \
    --topic /kf_gins/odom_pred \
    --out traj.tum
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main() -> None:
    ap = argparse.ArgumentParser(description="Export odometry topic to TUM trajectory text")
    ap.add_argument("--bag", required=True, help="ROS2 bag directory")
    ap.add_argument("--topic", default="/kf_gins/odom_pred", help="nav_msgs/Odometry topic")
    ap.add_argument("--out", required=True, help="Output .tum file")
    ap.add_argument("--segment-start-sec", type=float, default=None, help="Keep samples with (t-t0) >= start [s]")
    ap.add_argument("--segment-end-sec", type=float, default=None, help="Keep samples with (t-t0) <= end [s]")
    args = ap.parse_args()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.topic not in type_map:
        raise RuntimeError(f"Topic not found: {args.topic}")
    msg_type = get_message(type_map[args.topic])

    rows = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != args.topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        rows.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    if not rows:
        raise RuntimeError(f"No messages found on topic: {args.topic}")

    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    if args.segment_start_sec is not None or args.segment_end_sec is not None:
        filtered = []
        for r in rows:
            tr = r[0] - t0
            if args.segment_start_sec is not None and tr < args.segment_start_sec:
                continue
            if args.segment_end_sec is not None and tr > args.segment_end_sec:
                continue
            filtered.append(r)
        rows = filtered
        if not rows:
            raise RuntimeError("No samples remain after segment filtering.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                f"{r[0]:.9f} {r[1]:.9f} {r[2]:.9f} {r[3]:.9f} {r[4]:.9f} {r[5]:.9f} {r[6]:.9f} {r[7]:.9f}\n"
            )

    print(f"bag={args.bag}")
    print(f"topic={args.topic}")
    print(f"count={len(rows)}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()

