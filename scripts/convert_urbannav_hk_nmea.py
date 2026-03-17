#!/usr/bin/env python3

import argparse
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from convert_urbannav_hk_bag import (
    parse_antlever,
    read_bag,
    resolve_bag_path,
    write_kf_gins_imu,
    write_reference_nav,
)


GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
LEAP_SECOND_DATES = (
    (dt.datetime(1981, 7, 1, tzinfo=dt.timezone.utc), 1),
    (dt.datetime(1982, 7, 1, tzinfo=dt.timezone.utc), 2),
    (dt.datetime(1983, 7, 1, tzinfo=dt.timezone.utc), 3),
    (dt.datetime(1985, 7, 1, tzinfo=dt.timezone.utc), 4),
    (dt.datetime(1988, 1, 1, tzinfo=dt.timezone.utc), 5),
    (dt.datetime(1990, 1, 1, tzinfo=dt.timezone.utc), 6),
    (dt.datetime(1991, 1, 1, tzinfo=dt.timezone.utc), 7),
    (dt.datetime(1992, 7, 1, tzinfo=dt.timezone.utc), 8),
    (dt.datetime(1993, 7, 1, tzinfo=dt.timezone.utc), 9),
    (dt.datetime(1994, 7, 1, tzinfo=dt.timezone.utc), 10),
    (dt.datetime(1996, 1, 1, tzinfo=dt.timezone.utc), 11),
    (dt.datetime(1997, 7, 1, tzinfo=dt.timezone.utc), 12),
    (dt.datetime(1999, 1, 1, tzinfo=dt.timezone.utc), 13),
    (dt.datetime(2006, 1, 1, tzinfo=dt.timezone.utc), 14),
    (dt.datetime(2009, 1, 1, tzinfo=dt.timezone.utc), 15),
    (dt.datetime(2012, 7, 1, tzinfo=dt.timezone.utc), 16),
    (dt.datetime(2015, 7, 1, tzinfo=dt.timezone.utc), 17),
    (dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc), 18),
)


@dataclass
class NmeaGnssSample:
    gps_week: int
    sow: float
    lat_deg: float
    lon_deg: float
    height_m: float
    fix_quality: int
    hdop: float
    std_n: float
    std_e: float
    std_d: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UrbanNav HK bag + NMEA into KF-GINS text inputs."
    )
    parser.add_argument(
        "sequence_dir",
        type=Path,
        help="Sequence directory such as dataset/urban_nav/deep_urban",
    )
    parser.add_argument(
        "--nmea",
        type=Path,
        default=None,
        help="NMEA file path. Defaults to <sequence_dir>/GNSS/*ublox.f9p.nmea if unique.",
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
        help="Output directory. Defaults to <sequence_dir>/kf_gins_text_ublox_f9p",
    )
    parser.add_argument(
        "--fill-gaps-from",
        type=Path,
        nargs='*',
        default=(),
        help=(
            "Optional backup NMEA files. Missing epochs in the primary NMEA will be "
            "filled from these files in the provided order."
        ),
    )
    parser.add_argument(
        "--imu-topic",
        default="/imu/data",
        help="IMU topic name.",
    )
    parser.add_argument(
        "--inspvax-topic",
        default="/novatel_data/inspvax",
        help="NovAtel INSPVAX topic name used for IMU timing/reference initialization.",
    )
    parser.add_argument(
        "--min-std-h",
        type=float,
        default=1.0,
        help="Minimum horizontal GNSS std in meters.",
    )
    parser.add_argument(
        "--min-std-v",
        type=float,
        default=1.5,
        help="Minimum vertical GNSS std in meters.",
    )
    return parser.parse_args()


def resolve_nmea_path(sequence_dir: Path, nmea_arg: Optional[Path]) -> Path:
    if nmea_arg is not None:
        path = nmea_arg.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing NMEA file: {path}")
        return path

    candidates = sorted((sequence_dir / "GNSS").glob("*.ublox.f9p.nmea"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(f"No *.ublox.f9p.nmea file found under {sequence_dir / 'GNSS'}")
    raise ValueError(f"Found multiple candidate NMEA files: {[p.name for p in candidates]}")


def strip_checksum(value: str) -> str:
    return value.split("*", 1)[0]


def parse_lat_lon(value: str, hemi: str, is_lat: bool) -> float:
    if not value:
        raise ValueError("Empty latitude/longitude field")
    deg_width = 2 if is_lat else 3
    degrees = float(value[:deg_width])
    minutes = float(value[deg_width:])
    result = degrees + minutes / 60.0
    if hemi in ("S", "W"):
        result *= -1.0
    return result


def parse_time_hms(value: str) -> Tuple[int, int, float]:
    if not value:
        raise ValueError("Empty UTC time field")
    hours = int(value[0:2])
    minutes = int(value[2:4])
    seconds = float(value[4:])
    return hours, minutes, seconds


def parse_rmc_date(value: str) -> dt.date:
    if len(value) != 6:
        raise ValueError(f"Unexpected RMC date field: {value}")
    day = int(value[0:2])
    month = int(value[2:4])
    year = int(value[4:6]) + 2000
    return dt.date(year, month, day)


def leap_seconds_at(utc_time: dt.datetime) -> int:
    offset = 0
    for effective_date, value in LEAP_SECOND_DATES:
        if utc_time >= effective_date:
            offset = value
        else:
            break
    return offset


def utc_to_gps_week_sow(utc_time: dt.datetime) -> Tuple[int, float]:
    gps_time = utc_time + dt.timedelta(seconds=leap_seconds_at(utc_time))
    delta = gps_time - GPS_EPOCH
    total_seconds = delta.total_seconds()
    gps_week = int(total_seconds // 604800)
    sow = total_seconds - gps_week * 604800
    return gps_week, sow


def estimate_std_from_gga(
    fix_quality: int,
    hdop: float,
    min_std_h: float,
    min_std_v: float,
) -> Tuple[float, float, float]:
    quality_scale = {
        1: 2.0,
        2: 1.5,
        4: 0.7,
        5: 1.0,
    }.get(fix_quality, 2.5)
    hdop = hdop if math.isfinite(hdop) and hdop > 0 else 5.0
    std_h = max(min_std_h, quality_scale * hdop)
    std_v = max(min_std_v, 2.0 * std_h)
    return std_h, std_h, std_v


def read_nmea_gnss(
    nmea_path: Path,
    min_std_h: float,
    min_std_v: float,
) -> List[NmeaGnssSample]:
    current_date: Optional[dt.date] = None
    first_date: Optional[dt.date] = None
    samples: List[NmeaGnssSample] = []
    seen_sow: Dict[float, None] = {}

    with nmea_path.open("r", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("$"):
                continue

            fields = line.split(",")
            sentence = fields[0]

            if sentence in ("$GNRMC", "$GPRMC"):
                if len(fields) < 10 or fields[2] != "A":
                    continue
                try:
                    current_date = parse_rmc_date(strip_checksum(fields[9]))
                    first_date = first_date or current_date
                except ValueError:
                    continue
                continue

            if sentence not in ("$GNGGA", "$GPGGA"):
                continue
            if len(fields) < 12:
                continue

            if current_date is None and first_date is None:
                continue
            date_value = current_date or first_date
            if date_value is None:
                continue

            try:
                hours, minutes, seconds = parse_time_hms(fields[1])
                second_int = int(seconds)
                microsecond = int(round((seconds - second_int) * 1e6))
                if microsecond == 1000000:
                    second_int += 1
                    microsecond = 0
                utc_time = dt.datetime(
                    date_value.year,
                    date_value.month,
                    date_value.day,
                    hours,
                    minutes,
                    second_int,
                    microsecond,
                    tzinfo=dt.timezone.utc,
                )
                lat_deg = parse_lat_lon(fields[2], fields[3], is_lat=True)
                lon_deg = parse_lat_lon(fields[4], fields[5], is_lat=False)
                fix_quality = int(strip_checksum(fields[6] or "0"))
                hdop = float(strip_checksum(fields[8] or "nan"))
                altitude_msl = float(strip_checksum(fields[9] or "nan"))
                geoid_sep = float(strip_checksum(fields[11] or "nan"))
            except ValueError:
                continue

            if fix_quality <= 0 or not math.isfinite(lat_deg) or not math.isfinite(lon_deg):
                continue

            height_m = altitude_msl + geoid_sep
            gps_week, sow = utc_to_gps_week_sow(utc_time)
            sow_key = round(sow, 3)
            if sow_key in seen_sow:
                continue
            seen_sow[sow_key] = None

            std_n, std_e, std_d = estimate_std_from_gga(
                fix_quality=fix_quality,
                hdop=hdop,
                min_std_h=min_std_h,
                min_std_v=min_std_v,
            )
            samples.append(
                NmeaGnssSample(
                    gps_week=gps_week,
                    sow=sow,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    height_m=height_m,
                    fix_quality=fix_quality,
                    hdop=hdop,
                    std_n=std_n,
                    std_e=std_e,
                    std_d=std_d,
                )
            )

    if not samples:
        raise ValueError(f"No valid GGA samples found in {nmea_path}")

    return samples


def write_kf_gins_gnss(samples: Sequence[NmeaGnssSample], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        for sample in samples:
            f.write(
                f"{sample.sow:.3f} "
                f"{sample.lat_deg:.8f} {sample.lon_deg:.8f} {sample.height_m:.4f} "
                f"{sample.std_n:.3f} {sample.std_e:.3f} {sample.std_d:.3f}\n"
            )


def merge_gnss_samples(
    primary_samples: Sequence[NmeaGnssSample],
    backup_samples_list: Sequence[Sequence[NmeaGnssSample]],
) -> Tuple[List[NmeaGnssSample], List[int]]:
    merged = {round(sample.sow, 3): sample for sample in primary_samples}
    primary_start = primary_samples[0].sow
    primary_end = primary_samples[-1].sow
    fill_counts: List[int] = []

    for backup_samples in backup_samples_list:
        filled = 0
        for sample in backup_samples:
            key = round(sample.sow, 3)
            if key in merged:
                continue
            if sample.sow < primary_start or sample.sow > primary_end:
                continue
            merged[key] = sample
            filled += 1
        fill_counts.append(filled)

    ordered = [merged[key] for key in sorted(merged)]
    return ordered, fill_counts


def write_config(
    output_path: Path,
    imu_path: Path,
    gnss_path: Path,
    output_dir: Path,
    init_ref,
    imu_nominal_dt: float,
    antlever: Tuple[float, float, float],
    nmea_path: Path,
) -> None:
    imu_rate = int(round(1.0 / imu_nominal_dt))
    vel_d = -init_ref.vel_u_mps
    starttime = max(init_ref.sow - 1e-3, 0.0)

    content = f"""# Auto-generated for UrbanNav HK NMEA conversion
# IMU/reference are read from the ROS bag.
# GNSS position input is generated from:
#   {nmea_path.resolve()}
imupath: "{imu_path.resolve()}"
gnsspath: "{gnss_path.resolve()}"
outputpath: "{output_dir.resolve()}"

imudatalen: 7
imudatarate: {imu_rate}
starttime: {starttime:.3f}
endtime: -1

initpos: [ {init_ref.lat_deg:.8f}, {init_ref.lon_deg:.8f}, {init_ref.height_m:.4f} ]
initvel: [ {init_ref.vel_n_mps:.6f}, {init_ref.vel_e_mps:.6f}, {vel_d:.6f} ]
initatt: [ {init_ref.roll_deg:.6f}, {init_ref.pitch_deg:.6f}, {init_ref.azimuth_deg:.6f} ]

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
    nmea_path = resolve_nmea_path(sequence_dir, args.nmea)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (sequence_dir / "kf_gins_text_ublox_f9p").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_samples, inspvax_samples = read_bag(
        bag_path=bag_path,
        imu_topic=args.imu_topic,
        inspvax_topic=args.inspvax_topic,
    )
    antlever = parse_antlever(sequence_dir)
    primary_nmea_samples = read_nmea_gnss(
        nmea_path=nmea_path,
        min_std_h=args.min_std_h,
        min_std_v=args.min_std_v,
    )
    backup_paths = [path.resolve() for path in args.fill_gaps_from]
    backup_nmea_samples = [
        read_nmea_gnss(
            nmea_path=path,
            min_std_h=args.min_std_h,
            min_std_v=args.min_std_v,
        )
        for path in backup_paths
    ]
    nmea_samples, fill_counts = merge_gnss_samples(primary_nmea_samples, backup_nmea_samples)

    imu_out = output_dir / "urbannav_imu.txt"
    gnss_out = output_dir / "ublox_f9p_nmea_gnss.txt"
    reference_out = output_dir / "urbannav_reference.nav"
    config_out = output_dir / "urbannav_kf_gins_ublox_f9p_nmea.yaml"

    gps_offset = inspvax_samples[0].sow - inspvax_samples[0].bag_time
    imu_nominal_dt = write_kf_gins_imu(imu_samples, gps_offset, imu_out)
    write_kf_gins_gnss(nmea_samples, gnss_out)
    write_reference_nav(inspvax_samples, reference_out)
    write_config(
        output_path=config_out,
        imu_path=imu_out,
        gnss_path=gnss_out,
        output_dir=output_dir,
        init_ref=inspvax_samples[0],
        imu_nominal_dt=imu_nominal_dt,
        antlever=antlever,
        nmea_path=nmea_path,
    )

    print(f"Bag input:          {bag_path}")
    print(f"NMEA input:         {nmea_path}")
    print(f"IMU samples:        {len(imu_samples)}")
    print(f"Primary GNSS samples:{len(primary_nmea_samples)}")
    if backup_paths:
        for path, count in zip(backup_paths, fill_counts):
            print(f"Filled from {path.name}: {count}")
    print(f"Merged GNSS samples: {len(nmea_samples)}")
    print(f"GPS week:           {nmea_samples[0].gps_week}")
    print(f"GNSS time span:     {nmea_samples[0].sow:.3f} -> {nmea_samples[-1].sow:.3f}")
    print(f"IMU nominal dt:     {imu_nominal_dt:.6f} s")
    print(f"IMU output:         {imu_out}")
    print(f"GNSS output:        {gnss_out}")
    print(f"Reference output:   {reference_out}")
    print(f"Config output:      {config_out}")


if __name__ == "__main__":
    main()
