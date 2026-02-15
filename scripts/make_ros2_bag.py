#!/usr/bin/env python3
import argparse
import math
import os

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus


def time_from_seconds(t: float) -> Time:
    sec = int(math.floor(t))
    nanosec = int((t - sec) * 1e9)
    msg = Time()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def read_next_imu(fp):
    for line in fp:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        t = float(parts[0])
        dtheta = [float(parts[1]), float(parts[2]), float(parts[3])]
        dvel = [float(parts[4]), float(parts[5]), float(parts[6])]
        return t, dtheta, dvel
    return None


def read_next_gnss(fp):
    for line in fp:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        t = float(parts[0])
        lat = float(parts[1])
        lon = float(parts[2])
        alt = float(parts[3])
        std = [float(parts[4]), float(parts[5]), float(parts[6])]
        return t, lat, lon, alt, std
    return None


def make_writer(output_dir: str, overwrite: bool) -> SequentialWriter:
    if os.path.exists(output_dir):
        if overwrite:
            if os.path.isdir(output_dir):
                for name in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, name))
                os.rmdir(output_dir)
            else:
                os.remove(output_dir)
        else:
            raise RuntimeError(
                f"Output directory already exists: {output_dir}")
    writer = SequentialWriter()
    storage = StorageOptions(uri=output_dir, storage_id="sqlite3")
    converter = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr")
    writer.open(storage, converter)

    writer.create_topic(
        TopicMetadata(
            name="/imu/data",
            type="sensor_msgs/msg/Imu",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        TopicMetadata(
            name="/gps/fix",
            type="sensor_msgs/msg/NavSatFix",
            serialization_format="cdr",
        )
    )
    return writer


def main():
    parser = argparse.ArgumentParser(
        description="Convert KF-GINS dataset to ROS2 bag")
    parser.add_argument("--imu-file", required=True,
                        help="IMU dataset file (Leador-A15.txt)")
    parser.add_argument("--gnss-file", required=True,
                        help="GNSS dataset file (GNSS-RTK.txt)")
    parser.add_argument("--output", required=True,
                        help="Output rosbag2 directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory if exists")
    parser.add_argument("--imu-rate", type=float,
                        default=200.0, help="IMU rate for fallback dt")
    parser.add_argument("--max-imu-dt", type=float,
                        default=0.1, help="Max dt before fallback")
    parser.add_argument("--start-time-zero",
                        action="store_true", help="Shift start time to 0")
    parser.add_argument(
        "--imu-frame-id", default="imu_link", help="IMU frame_id")
    parser.add_argument(
        "--gps-frame-id", default="gps_link", help="GNSS frame_id")
    args = parser.parse_args()

    with open(args.imu_file, "r") as f_imu, open(args.gnss_file, "r") as f_gnss:
        imu_next = read_next_imu(f_imu)
        gnss_next = read_next_gnss(f_gnss)
        if imu_next is None or gnss_next is None:
            raise RuntimeError("IMU or GNSS file is empty")

        t0 = min(imu_next[0], gnss_next[0]) if args.start_time_zero else 0.0
        writer = make_writer(args.output, args.overwrite)

        # Write GNSS first, then IMU, preserving original timestamps
        while gnss_next is not None:
            t, lat, lon, alt, std = gnss_next
            t_ns = int((t - t0) * 1e9)

            gnss_msg = NavSatFix()
            gnss_msg.header.stamp = time_from_seconds(t - t0)
            gnss_msg.header.frame_id = args.gps_frame_id
            gnss_msg.status.status = NavSatStatus.STATUS_FIX
            gnss_msg.status.service = NavSatStatus.SERVICE_GPS
            gnss_msg.latitude = lat
            gnss_msg.longitude = lon
            gnss_msg.altitude = alt
            gnss_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            gnss_msg.position_covariance[0] = std[0] * std[0]
            gnss_msg.position_covariance[4] = std[1] * std[1]
            gnss_msg.position_covariance[8] = std[2] * std[2]

            writer.write("/gps/fix", serialize_message(gnss_msg), t_ns)
            gnss_next = read_next_gnss(f_gnss)

        prev_imu_time = imu_next[0]
        while imu_next is not None:
            t, dtheta, dvel = imu_next
            dt = t - prev_imu_time
            if dt <= 0.0 or dt > args.max_imu_dt:
                dt = 1.0 / args.imu_rate
            prev_imu_time = t

            t_ns = int((t - t0) * 1e9)

            imu_msg = Imu()
            imu_msg.header.stamp = time_from_seconds(t - t0)
            imu_msg.header.frame_id = args.imu_frame_id
            imu_msg.angular_velocity.x = dtheta[0] / dt
            imu_msg.angular_velocity.y = dtheta[1] / dt
            imu_msg.angular_velocity.z = dtheta[2] / dt
            imu_msg.linear_acceleration.x = dvel[0] / dt
            imu_msg.linear_acceleration.y = dvel[1] / dt
            imu_msg.linear_acceleration.z = dvel[2] / dt

            writer.write("/imu/data", serialize_message(imu_msg), t_ns)
            imu_next = read_next_imu(f_imu)


if __name__ == "__main__":
    main()
