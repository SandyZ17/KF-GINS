#!/usr/bin/env python3
import argparse
import datetime as dt
import math
from typing import Tuple

from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

SECONDS_PER_WEEK = 604800


def unix_ns_to_year_sow(unix_ns: int) -> Tuple[int, float]:
    unix_sec = float(unix_ns) * 1e-9
    utc_dt = dt.datetime.fromtimestamp(unix_sec, dt.timezone.utc)
    year = utc_dt.year
    # "truth.nav"-style second column follows GNSS week-second style.
    gps_weekday = (utc_dt.weekday() + 1) % 7  # Monday=1 ... Saturday=6, Sunday=0
    sow = float(gps_weekday) * 86400.0 + utc_dt.hour * 3600.0 + utc_dt.minute * 60.0 + utc_dt.second
    sow += utc_dt.microsecond * 1e-6
    sow %= SECONDS_PER_WEEK
    return year, sow


def quat_to_rpy_deg(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    # ROS quaternion -> roll/pitch/yaw (deg), ZYX convention
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def append_path_rows_to_nav(
    bag_dir: str, topic: str, out_path: str, truncate: bool, time_source: str
) -> int:
    storage = StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage, converter)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    if topic not in type_map:
        raise RuntimeError(f"Topic not found in bag: {topic}")

    msg_type = get_message(type_map[topic])
    last_stamp_ns = -1
    written = 0

    if truncate:
        open(out_path, "w", encoding="utf-8").close()

    with open(out_path, "a", encoding="utf-8") as f:
        while reader.has_next():
            topic_name, data, bag_time_ns = reader.read_next()
            if topic_name != topic:
                continue
            msg = deserialize_message(data, msg_type)
            if not msg.poses:
                continue

            # Path message typically contains full history; use newest pose only.
            pose = msg.poses[-1]
            stamp_ns = int(pose.header.stamp.sec) * 1_000_000_000 + int(pose.header.stamp.nanosec)
            if stamp_ns == last_stamp_ns:
                continue
            last_stamp_ns = stamp_ns

            if time_source == "header":
                unix_ns = int(pose.header.stamp.sec) * 1_000_000_000 + int(pose.header.stamp.nanosec)
            else:
                unix_ns = int(bag_time_ns)
            year, sow = unix_ns_to_year_sow(unix_ns)
            px = pose.pose.position.x
            py = pose.pose.position.y
            pz = pose.pose.position.z
            q = pose.pose.orientation
            roll, pitch, yaw = quat_to_rpy_deg(q.x, q.y, q.z, q.w)
            f.write(
                f"{int(year):5d} {sow:10.3f} {px:16.10f} {py:16.10f} {pz:10.3f} "
                f"{0.0:10.3f} {0.0:10.3f} {0.0:10.3f} {roll:12.8f} {pitch:12.8f} {yaw:12.8f}\n"
            )
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Convert nav_msgs/Path in rosbag2 to truth.nav-like text")
    parser.add_argument("--bag", required=True, help="rosbag2 directory")
    parser.add_argument("--topic", default="/kf_gins/path", help="nav_msgs/Path topic")
    parser.add_argument("--out", required=True, help="output nav text path")
    parser.add_argument("--truncate", action="store_true", help="truncate output file before appending")
    parser.add_argument(
        "--time-source",
        choices=["bag", "header"],
        default="bag",
        help="timestamp source for GPS week/sow conversion (default: bag)",
    )
    args = parser.parse_args()

    written = append_path_rows_to_nav(args.bag, args.topic, args.out, args.truncate, args.time_source)
    if written == 0:
        raise RuntimeError("No poses found in Path messages.")

    print(f"Appended {written} rows to {args.out}")


if __name__ == "__main__":
    main()
