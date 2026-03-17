#!/usr/bin/env python3

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class ImuSample:
    tow: float
    week: int
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class ReferenceSample:
    tow: float
    week: int
    lat_deg: float
    lon_deg: float
    height_m: float
    roll_deg: float
    pitch_deg: float
    heading_deg: float
    vel_e_mps: float
    vel_n_mps: float
    vel_u_mps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UrbanNav Tokyo data into KF-GINS plain-text inputs."
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Sequence directory such as dataset/urban_nav/Tokyo_Data/Odaiba",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sequence_dir>/kf_gins_text",
    )
    parser.add_argument(
        "--gnss-decimation",
        type=int,
        default=10,
        help="Keep one reference row every N rows when generating placeholder GNSS.",
    )
    parser.add_argument(
        "--gnss-std",
        type=float,
        nargs=3,
        default=(5.0, 5.0, 8.0),
        metavar=("STD_N", "STD_E", "STD_D"),
        help="Placeholder GNSS standard deviations in meters.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> Iterable[List[str]]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                yield row


def load_imu(path: Path) -> List[ImuSample]:
    samples: List[ImuSample] = []
    for row in load_csv_rows(path):
        samples.append(
            ImuSample(
                tow=float(row[0]),
                week=int(float(row[1])),
                accel_x=float(row[2]),
                accel_y=float(row[3]),
                accel_z=float(row[4]),
                gyro_x=float(row[5]),
                gyro_y=float(row[6]),
                gyro_z=float(row[7]),
            )
        )
    if len(samples) < 2:
        raise ValueError(f"IMU file {path} does not contain enough samples")
    return samples


def load_reference(path: Path) -> List[ReferenceSample]:
    samples: List[ReferenceSample] = []
    for row in load_csv_rows(path):
        samples.append(
            ReferenceSample(
                tow=float(row[0]),
                week=int(float(row[1])),
                lat_deg=float(row[2]),
                lon_deg=float(row[3]),
                height_m=float(row[4]),
                roll_deg=float(row[8]),
                pitch_deg=float(row[9]),
                heading_deg=float(row[10]),
                vel_e_mps=float(row[11]),
                vel_n_mps=float(row[12]),
                vel_u_mps=float(row[13]),
            )
        )
    if not samples:
        raise ValueError(f"Reference file {path} is empty")
    return samples


def infer_nominal_dt(times: Sequence[float]) -> float:
    dts = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:]) if t2 > t1]
    if not dts:
        raise ValueError("Unable to infer dt from timestamps")
    return statistics.median(dts[: min(len(dts), 200)])


def ensure_single_week(name: str, weeks: Sequence[int]) -> int:
    unique_weeks = sorted(set(weeks))
    if len(unique_weeks) != 1:
        raise ValueError(f"{name} spans multiple GPS weeks: {unique_weeks}")
    return unique_weeks[0]


def write_kf_gins_imu(samples: Sequence[ImuSample], output_path: Path) -> float:
    nominal_dt = infer_nominal_dt([s.tow for s in samples])
    with output_path.open("w", newline="") as f:
        for index, sample in enumerate(samples):
            if index == 0:
                dt = nominal_dt
            else:
                dt = sample.tow - samples[index - 1].tow
                if dt <= 0:
                    raise ValueError("Non-increasing IMU timestamps detected")
            dtheta_x = sample.gyro_x * dt
            dtheta_y = sample.gyro_y * dt
            dtheta_z = sample.gyro_z * dt
            dvel_x = sample.accel_x * dt
            dvel_y = sample.accel_y * dt
            dvel_z = sample.accel_z * dt
            f.write(
                f"{sample.tow:.6f} "
                f"{dtheta_x:.12e} {dtheta_y:.12e} {dtheta_z:.12e} "
                f"{dvel_x:.12e} {dvel_y:.12e} {dvel_z:.12e}\n"
            )
    return nominal_dt


def write_kf_gins_reference(samples: Sequence[ReferenceSample], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        for sample in samples:
            vel_n = sample.vel_n_mps
            vel_e = sample.vel_e_mps
            vel_d = -sample.vel_u_mps
            f.write(
                f"{sample.week:d} {sample.tow:.2f} "
                f"{sample.lat_deg:.8f} {sample.lon_deg:.8f} {sample.height_m:.4f} "
                f"{vel_n:.6f} {vel_e:.6f} {vel_d:.6f} "
                f"{sample.roll_deg:.6f} {sample.pitch_deg:.6f} {sample.heading_deg:.6f}\n"
            )


def write_placeholder_gnss(
    samples: Sequence[ReferenceSample],
    decimation: int,
    std_ned: Sequence[float],
    output_path: Path,
) -> None:
    if decimation <= 0:
        raise ValueError("--gnss-decimation must be positive")
    with output_path.open("w", newline="") as f:
        for index, sample in enumerate(samples):
            if index % decimation != 0:
                continue
            f.write(
                f"{sample.tow:.2f} "
                f"{sample.lat_deg:.8f} {sample.lon_deg:.8f} {sample.height_m:.4f} "
                f"{std_ned[0]:.3f} {std_ned[1]:.3f} {std_ned[2]:.3f}\n"
            )


def write_config(
    output_path: Path,
    imu_path: Path,
    gnss_path: Path,
    output_dir: Path,
    reference_samples: Sequence[ReferenceSample],
    imu_nominal_dt: float,
) -> None:
    init = reference_samples[0]
    init_vel_n = init.vel_n_mps
    init_vel_e = init.vel_e_mps
    init_vel_d = -init.vel_u_mps
    imu_rate = int(round(1.0 / imu_nominal_dt))

    content = f"""# Auto-generated for UrbanNav Tokyo
imupath: "{imu_path.resolve()}"
gnsspath: "{gnss_path.resolve()}"
outputpath: "{output_dir.resolve()}"

imudatalen: 7
imudatarate: {imu_rate}
starttime: {init.tow:.2f}
endtime: -1

initpos: [ {init.lat_deg:.8f}, {init.lon_deg:.8f}, {init.height_m:.4f} ]
initvel: [ {init_vel_n:.6f}, {init_vel_e:.6f}, {init_vel_d:.6f} ]
initatt: [ {init.roll_deg:.6f}, {init.pitch_deg:.6f}, {init.heading_deg:.6f} ]

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

antlever: [ 0.136, -0.301, -0.184 ]
"""
    output_path.write_text(content)


def main() -> None:
    args = parse_args()
    sequence_dir = args.sequence_dir.resolve()
    output_dir = (args.output_dir or (sequence_dir / "kf_gins_text")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_path = sequence_dir / "imu.csv"
    reference_path = sequence_dir / "reference.csv"
    if not imu_path.exists():
        raise FileNotFoundError(f"Missing {imu_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing {reference_path}")

    imu_samples = load_imu(imu_path)
    reference_samples = load_reference(reference_path)

    imu_week = ensure_single_week("IMU", [s.week for s in imu_samples])
    reference_week = ensure_single_week("reference", [s.week for s in reference_samples])
    if imu_week != reference_week:
        raise ValueError(f"IMU week {imu_week} does not match reference week {reference_week}")

    imu_out = output_dir / "urbannav_imu.txt"
    reference_out = output_dir / "urbannav_reference.nav"
    gnss_out = output_dir / "urbannav_gnss_from_reference.txt"
    config_out = output_dir / "urbannav_kf_gins.yaml"

    imu_nominal_dt = write_kf_gins_imu(imu_samples, imu_out)
    write_kf_gins_reference(reference_samples, reference_out)
    write_placeholder_gnss(reference_samples, args.gnss_decimation, args.gnss_std, gnss_out)
    write_config(config_out, imu_out, gnss_out, output_dir, reference_samples, imu_nominal_dt)

    print(f"Converted sequence: {sequence_dir}")
    print(f"IMU output:        {imu_out}")
    print(f"Reference output:  {reference_out}")
    print(f"GNSS output:       {gnss_out}")
    print(f"Config output:     {config_out}")
    print(f"IMU nominal dt:    {imu_nominal_dt:.6f} s")
    print(f"GNSS decimation:   {args.gnss_decimation}")
    print("Note: GNSS output is currently a placeholder generated from reference.csv.")


if __name__ == "__main__":
    main()
