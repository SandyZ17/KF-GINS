#!/usr/bin/env python3

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rosbags.highlevel import AnyReader


@dataclass
class ImuSample:
    ros_time: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class InspvaxSample:
    bag_time: float
    gps_week: int
    sow: float
    lat_deg: float
    lon_deg: float
    height_m: float
    vel_n_mps: float
    vel_e_mps: float
    vel_u_mps: float
    roll_deg: float
    pitch_deg: float
    azimuth_deg: float
    lat_std_m: float
    lon_std_m: float
    height_std_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UrbanNav HK ROS bag data into KF-GINS plain-text inputs."
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Sequence directory such as dataset/urban_nav/medium_urban",
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=None,
        help="Explicit bag path. Defaults to the only .bag file under sequence_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sequence_dir>/kf_gins_text",
    )
    parser.add_argument(
        "--imu-topic",
        default="/imu/data",
        help="IMU topic name.",
    )
    parser.add_argument(
        "--inspvax-topic",
        default="/novatel_data/inspvax",
        help="NovAtel INSPVAX topic name used for pseudo-GNSS/reference output.",
    )
    parser.add_argument(
        "--gnss-std",
        type=float,
        nargs=3,
        default=(5.0, 5.0, 8.0),
        metavar=("STD_N", "STD_E", "STD_D"),
        help="Fallback GNSS position standard deviations in meters when INSPVAX std fields are zero.",
    )
    return parser.parse_args()


def resolve_bag_path(sequence_dir: Path, bag_arg: Optional[Path]) -> Path:
    if bag_arg is not None:
        bag_path = bag_arg.resolve()
        if not bag_path.exists():
            raise FileNotFoundError(f"Missing bag file: {bag_path}")
        return bag_path

    bags = sorted(sequence_dir.glob("*.bag"))
    if not bags:
        raise FileNotFoundError(f"No .bag file found under {sequence_dir}")
    if len(bags) > 1:
        raise ValueError(
            f"Found multiple bag files under {sequence_dir}; pass --bag explicitly."
        )
    return bags[0].resolve()


def read_bag(
    bag_path: Path,
    imu_topic: str,
    inspvax_topic: str,
) -> Tuple[List[ImuSample], List[InspvaxSample]]:
    imu_samples: List[ImuSample] = []
    inspvax_samples: List[InspvaxSample] = []

    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == imu_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                imu_samples.append(
                    ImuSample(
                        ros_time=ros_time,
                        accel_x=msg.linear_acceleration.x,
                        accel_y=msg.linear_acceleration.y,
                        accel_z=msg.linear_acceleration.z,
                        gyro_x=msg.angular_velocity.x,
                        gyro_y=msg.angular_velocity.y,
                        gyro_z=msg.angular_velocity.z,
                    )
                )
            elif connection.topic == inspvax_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                inspvax_samples.append(
                    InspvaxSample(
                        bag_time=timestamp * 1e-9,
                        gps_week=int(msg.header.gps_week),
                        sow=msg.header.gps_week_seconds / 1000.0,
                        lat_deg=msg.latitude,
                        lon_deg=msg.longitude,
                        height_m=msg.altitude,
                        vel_n_mps=msg.north_velocity,
                        vel_e_mps=msg.east_velocity,
                        vel_u_mps=msg.up_velocity,
                        roll_deg=msg.roll,
                        pitch_deg=msg.pitch,
                        azimuth_deg=msg.azimuth,
                        lat_std_m=msg.latitude_std,
                        lon_std_m=msg.longitude_std,
                        height_std_m=msg.altitude_std,
                    )
                )

    if len(imu_samples) < 2:
        raise ValueError(f"IMU topic {imu_topic} does not contain enough samples")
    if not inspvax_samples:
        raise ValueError(f"INSPVAX topic {inspvax_topic} is empty")

    return imu_samples, inspvax_samples


def ensure_single_week(samples: Sequence[InspvaxSample]) -> int:
    weeks = sorted({sample.gps_week for sample in samples})
    if len(weeks) != 1:
        raise ValueError(f"INSPVAX spans multiple GPS weeks: {weeks}")
    return weeks[0]


def infer_gps_offset(samples: Sequence[InspvaxSample]) -> float:
    offsets = [sample.sow - sample.bag_time for sample in samples]
    return statistics.median(offsets)


def infer_nominal_dt(times: Sequence[float]) -> float:
    dts = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:]) if t2 > t1]
    if not dts:
        raise ValueError("Unable to infer IMU dt from timestamps")
    return statistics.median(dts[: min(len(dts), 500)])


def positive_or_default(value: float, default: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return default
    return value


def parse_antlever(sequence_dir: Path) -> Tuple[float, float, float]:
    extrinsic_path = sequence_dir / "extrinsic.yaml"
    if not extrinsic_path.exists():
        return 0.0, 0.0, 0.0

    text = extrinsic_path.read_text()
    match = re.search(
        r"ANTENNA_T_IMU:.*?data:\s*\[(.*?)\]",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return 0.0, 0.0, 0.0

    values = [
        float(token.strip())
        for token in match.group(1).replace("\n", " ").split(",")
        if token.strip()
    ]
    if len(values) != 16:
        return 0.0, 0.0, 0.0

    x_right = values[3]
    y_forward = values[7]
    z_up = values[11]

    return y_forward, x_right, -z_up


def write_kf_gins_imu(
    samples: Sequence[ImuSample],
    gps_offset: float,
    output_path: Path,
) -> float:
    times = [sample.ros_time + gps_offset for sample in samples]
    nominal_dt = infer_nominal_dt(times)

    with output_path.open("w", newline="") as f:
        for index, sample in enumerate(samples):
            time_sow = times[index]
            if index == 0:
                dt = nominal_dt
            else:
                dt = time_sow - times[index - 1]
                if dt <= 0 or dt > 0.1:
                    dt = nominal_dt

            # UrbanNav extrinsics document the body/IMU axes as right-forward-up.
            # KF-GINS expects IMU increments in forward-right-down order.
            dtheta_x = sample.gyro_y * dt
            dtheta_y = sample.gyro_x * dt
            dtheta_z = -sample.gyro_z * dt
            dvel_x = sample.accel_y * dt
            dvel_y = sample.accel_x * dt
            dvel_z = -sample.accel_z * dt
            f.write(
                f"{time_sow:.6f} "
                f"{dtheta_x:.12e} {dtheta_y:.12e} {dtheta_z:.12e} "
                f"{dvel_x:.12e} {dvel_y:.12e} {dvel_z:.12e}\n"
            )

    return nominal_dt


def write_kf_gins_gnss(
    samples: Sequence[InspvaxSample],
    default_std_ned: Sequence[float],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="") as f:
        for sample in samples:
            std_n = positive_or_default(sample.lat_std_m, default_std_ned[0])
            std_e = positive_or_default(sample.lon_std_m, default_std_ned[1])
            std_d = positive_or_default(sample.height_std_m, default_std_ned[2])
            f.write(
                f"{sample.sow:.3f} "
                f"{sample.lat_deg:.8f} {sample.lon_deg:.8f} {sample.height_m:.4f} "
                f"{std_n:.3f} {std_e:.3f} {std_d:.3f}\n"
            )


def write_reference_nav(
    samples: Sequence[InspvaxSample],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="") as f:
        for sample in samples:
            vel_d = -sample.vel_u_mps
            f.write(
                f"{sample.gps_week:d} {sample.sow:.3f} "
                f"{sample.lat_deg:.8f} {sample.lon_deg:.8f} {sample.height_m:.4f} "
                f"{sample.vel_n_mps:.6f} {sample.vel_e_mps:.6f} {vel_d:.6f} "
                f"{sample.roll_deg:.6f} {sample.pitch_deg:.6f} {sample.azimuth_deg:.6f}\n"
            )


def write_config(
    output_path: Path,
    imu_path: Path,
    gnss_path: Path,
    output_dir: Path,
    init: InspvaxSample,
    imu_nominal_dt: float,
    antlever: Tuple[float, float, float],
) -> None:
    imu_rate = int(round(1.0 / imu_nominal_dt))
    vel_d = -init.vel_u_mps
    starttime = max(init.sow - 1e-3, 0.0)

    content = f"""# Auto-generated for UrbanNav HK bag conversion
# GNSS output is derived from /novatel_data/inspvax (NovAtel SPAN solution),
# not from a standalone raw GNSS position topic.
imupath: "{imu_path.resolve()}"
gnsspath: "{gnss_path.resolve()}"
outputpath: "{output_dir.resolve()}"

imudatalen: 7
imudatarate: {imu_rate}
starttime: {starttime:.3f}
endtime: -1

initpos: [ {init.lat_deg:.8f}, {init.lon_deg:.8f}, {init.height_m:.4f} ]
initvel: [ {init.vel_n_mps:.6f}, {init.vel_e_mps:.6f}, {vel_d:.6f} ]
initatt: [ {init.roll_deg:.6f}, {init.pitch_deg:.6f}, {init.azimuth_deg:.6f} ]

initgyrbias: [ 0, 0, 0 ]
initaccbias: [ 0, 0, 0 ]
initgyrscale: [ 0, 0, 0 ]
initaccscale: [ 0, 0, 0 ]

initposstd: [ 5.0, 5.0, 8.0 ]
initvelstd: [ 1.0, 1.0, 1.0 ]
initattstd: [ 1.0, 1.0, 2.0 ]

imunoise:
  arw: [0.24, 0.24, 0.24]
  vrw: [0.24, 0.24, 0.24]
  gbstd: [50.0, 50.0, 50.0]
  abstd: [250.0, 250.0, 250.0]
  gsstd: [1000.0, 1000.0, 1000.0]
  asstd: [1000.0, 1000.0, 1000.0]
  corrtime: 1.0

antlever: [ {antlever[0]:.3f}, {antlever[1]:.3f}, {antlever[2]:.3f} ]
"""
    output_path.write_text(content)


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir.resolve()
    bag_path = resolve_bag_path(sequence_dir, args.bag)
    output_dir = (args.output_dir or (sequence_dir / "kf_gins_text")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_samples, inspvax_samples = read_bag(
        bag_path=bag_path,
        imu_topic=args.imu_topic,
        inspvax_topic=args.inspvax_topic,
    )

    gps_week = ensure_single_week(inspvax_samples)
    gps_offset = infer_gps_offset(inspvax_samples)
    antlever = parse_antlever(sequence_dir)

    imu_out = output_dir / "urbannav_imu.txt"
    gnss_out = output_dir / "urbannav_gnss.txt"
    reference_out = output_dir / "urbannav_reference.nav"
    config_out = output_dir / "urbannav_kf_gins.yaml"

    imu_nominal_dt = write_kf_gins_imu(imu_samples, gps_offset, imu_out)
    write_kf_gins_gnss(inspvax_samples, args.gnss_std, gnss_out)
    write_reference_nav(inspvax_samples, reference_out)
    write_config(
        output_path=config_out,
        imu_path=imu_out,
        gnss_path=gnss_out,
        output_dir=output_dir,
        init=inspvax_samples[0],
        imu_nominal_dt=imu_nominal_dt,
        antlever=antlever,
    )

    print(f"Bag input:          {bag_path}")
    print(f"GPS week:           {gps_week}")
    print(f"GPS offset:         {gps_offset:.6f} s")
    print(f"IMU samples:        {len(imu_samples)}")
    print(f"INSPVAX samples:    {len(inspvax_samples)}")
    print(f"IMU nominal dt:     {imu_nominal_dt:.6f} s")
    print(f"IMU output:         {imu_out}")
    print(f"GNSS output:        {gnss_out}")
    print(f"Reference output:   {reference_out}")
    print(f"Config output:      {config_out}")
    print("Note: GNSS output is generated from /novatel_data/inspvax.")


if __name__ == "__main__":
    main()
