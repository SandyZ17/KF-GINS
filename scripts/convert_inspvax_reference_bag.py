#!/usr/bin/env python3
"""
Convert NovAtel INSPVAX messages into reference ROS2 topics:
- sensor_msgs/NavSatFix
- sensor_msgs/Imu (orientation only; angular vel/acc unavailable)
- nav_msgs/Odometry (local ENU, referenced to first INSPVAX sample)

Supports ROS1/ROS2 source bags via rosbags AnyReader.
Writes a ROS2 bag (sqlite3) via rosbag2_py.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata
from rosbags.highlevel import AnyReader
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus


WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013


def radiusmn(lat_rad: float):
    s2 = math.sin(lat_rad) ** 2
    t = 1.0 - WGS84_E1 * s2
    st = math.sqrt(t)
    rm = WGS84_RA * (1.0 - WGS84_E1) / (st * t)
    rn = WGS84_RA / st
    return rm, rn


def time_from_ns(t_ns: int) -> Time:
    sec = t_ns // 1_000_000_000
    nanosec = t_ns % 1_000_000_000
    msg = Time()
    msg.sec = int(sec)
    msg.nanosec = int(nanosec)
    return msg


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def quat_from_rotmat(r):
    q = Quaternion()
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q.w = 0.25 * s
        q.x = (r[2][1] - r[1][2]) / s
        q.y = (r[0][2] - r[2][0]) / s
        q.z = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        q.w = (r[2][1] - r[1][2]) / s
        q.x = 0.25 * s
        q.y = (r[0][1] + r[1][0]) / s
        q.z = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        q.w = (r[0][2] - r[2][0]) / s
        q.x = (r[0][1] + r[1][0]) / s
        q.y = 0.25 * s
        q.z = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        q.w = (r[1][0] - r[0][1]) / s
        q.x = (r[0][2] + r[2][0]) / s
        q.y = (r[1][2] + r[2][1]) / s
        q.z = 0.25 * s
    return q


def matmul3(a, b):
    return [
        [
            a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def novatel_rpy_to_ros_quat(roll_deg: float, pitch_deg: float, az_deg: float) -> Quaternion:
    """Convert NovAtel INSPVAX attitude (NED/FRD, azimuth cw-from-north) to ROS ENU/FLU quaternion.

    INSPVAX attitude is commonly expressed in local NED navigation frame with body frame FRD.
    ROS odom/imu orientation is expected in ENU with body FLU semantics.
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(az_deg)  # In NED(+Down), positive yaw is clockwise from North.

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Rotation from body(FRD) to nav(NED), ZYX yaw-pitch-roll.
    r_ned_frd = [
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ]

    # Frame basis transforms: v_enu = T_enu_ned * v_ned, v_frd = T_frd_flu * v_flu
    t_enu_ned = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    t_frd_flu = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]

    r_enu_flu = matmul3(matmul3(t_enu_ned, r_ned_frd), t_frd_flu)
    return quat_from_rotmat(r_enu_flu)


def make_writer(out_dir: str) -> SequentialWriter:
    if os.path.exists(out_dir):
        raise RuntimeError(f"Output exists: {out_dir}")
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=out_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    writer.create_topic(
        TopicMetadata(
            name="/reference/inspvax/navsatfix",
            type="sensor_msgs/msg/NavSatFix",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        TopicMetadata(
            name="/reference/inspvax/imu",
            type="sensor_msgs/msg/Imu",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        TopicMetadata(
            name="/reference/inspvax/odom",
            type="nav_msgs/msg/Odometry",
            serialization_format="cdr",
        )
    )
    return writer


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert INSPVAX to reference ROS2 topics")
    ap.add_argument("--src-bag", required=True, help="Source bag (ROS1 or ROS2)")
    ap.add_argument("--inspvax-topic", default="/novatel_data/inspvax", help="INSPVAX topic")
    ap.add_argument("--out-bag", required=True, help="Output ROS2 bag dir")
    ap.add_argument("--frame-id", default="map")
    ap.add_argument("--child-frame-id", default="base_link_ref")
    args = ap.parse_args()

    src_path = Path(args.src_bag)
    with AnyReader([src_path]) as reader:
        conns = [c for c in reader.connections if c.topic == args.inspvax_topic]
        if not conns:
            raise RuntimeError(f"Topic not found: {args.inspvax_topic}")

        writer = make_writer(args.out_bag)
        lat0 = lon0 = h0 = None
        rm0 = rn0 = None
        count = 0

        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            lat_deg = float(msg.latitude)
            lon_deg = float(msg.longitude)
            alt_m = float(msg.altitude)
            vn = float(msg.north_velocity)
            ve = float(msg.east_velocity)
            vu = float(msg.up_velocity)
            roll_deg = float(msg.roll)
            pitch_deg = float(msg.pitch)
            az_deg = float(msg.azimuth)

            lat = math.radians(lat_deg)
            lon = math.radians(lon_deg)
            if lat0 is None:
                lat0, lon0, h0 = lat, lon, alt_m
                rm0, rn0 = radiusmn(lat0)

            # Local ENU from first sample
            north = (lat - lat0) * (rm0 + h0)
            east = (lon - lon0) * (rn0 + h0) * math.cos(lat0)
            up = alt_m - h0

            # Convert INSPVAX attitude (typically NED/FRD) into ROS ENU/FLU quaternion.
            q = novatel_rpy_to_ros_quat(roll_deg, pitch_deg, az_deg)

            stamp = time_from_ns(int(t_ns))

            nav = NavSatFix()
            nav.header.stamp = stamp
            nav.header.frame_id = "gps_link_ref"
            nav.status.status = NavSatStatus.STATUS_FIX
            nav.status.service = NavSatStatus.SERVICE_GPS
            nav.latitude = lat_deg
            nav.longitude = lon_deg
            nav.altitude = alt_m
            if getattr(msg, "latitude_std", 0.0) > 0 and getattr(msg, "longitude_std", 0.0) > 0 and getattr(msg, "altitude_std", 0.0) > 0:
                # Approximate conversion: degree std to meter variance at current latitude for lat/lon.
                rm, rn = radiusmn(lat)
                sigma_n = math.radians(float(msg.latitude_std)) * (rm + alt_m)
                sigma_e = math.radians(float(msg.longitude_std)) * (rn + alt_m) * math.cos(lat)
                sigma_u = float(msg.altitude_std)
                nav.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
                nav.position_covariance[0] = sigma_e * sigma_e
                nav.position_covariance[4] = sigma_n * sigma_n
                nav.position_covariance[8] = sigma_u * sigma_u
            else:
                nav.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = args.child_frame_id
            imu.orientation = q
            # INSPVAX does not provide raw gyro/accel; mark unavailable.
            imu.angular_velocity_covariance[0] = -1.0
            imu.linear_acceleration_covariance[0] = -1.0
            if getattr(msg, "roll_std", 0.0) > 0 and getattr(msg, "pitch_std", 0.0) > 0 and getattr(msg, "azimuth_std", 0.0) > 0:
                imu.orientation_covariance[0] = math.radians(float(msg.roll_std)) ** 2
                imu.orientation_covariance[4] = math.radians(float(msg.pitch_std)) ** 2
                imu.orientation_covariance[8] = math.radians(float(msg.azimuth_std)) ** 2
            else:
                imu.orientation_covariance[0] = -1.0

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = args.frame_id
            odom.child_frame_id = args.child_frame_id
            odom.pose.pose.position.x = east
            odom.pose.pose.position.y = north
            odom.pose.pose.position.z = up
            odom.pose.pose.orientation = q
            odom.twist.twist.linear.x = ve
            odom.twist.twist.linear.y = vn
            odom.twist.twist.linear.z = vu

            writer.write("/reference/inspvax/navsatfix", serialize_message(nav), int(t_ns))
            writer.write("/reference/inspvax/imu", serialize_message(imu), int(t_ns))
            writer.write("/reference/inspvax/odom", serialize_message(odom), int(t_ns))
            count += 1

    print(f"Converted {count} INSPVAX messages into reference ROS2 bag: {args.out_bag}")


if __name__ == "__main__":
    main()
