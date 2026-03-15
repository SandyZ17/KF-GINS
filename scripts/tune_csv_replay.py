#!/usr/bin/env python3
"""
Tune KF-GINS replay parameters directly from exported imu.csv and gnss.csv.

The tuner runs the offline replay executable repeatedly, records each trial's
config and metrics, and writes everything into a single run directory.

Example:
  python3 src/KF-GINS/scripts/tune_csv_replay.py \
    --base-params src/KF-GINS/config/params_hangzhou_town_AB_xyz.yaml \
    --imu-csv src/KF-GINS/results/A_B_xyz_measurements_latest/imu.csv \
    --gnss-csv src/KF-GINS/results/A_B_xyz_measurements_latest/gnss.csv \
    --out-dir src/KF-GINS/results/A_B_xyz_tuning_run
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


@dataclass
class TrialResult:
    stage: str
    trial_name: str
    trial_dir: Path
    params_path: Path
    metrics_path: Path
    residuals_path: Path
    console_path: Path
    exit_code: int
    success: bool
    metrics: Dict[str, Any]
    updates: Dict[str, Any]


@dataclass
class GnssPathSample:
    time_sec: float
    north_m: float
    east_m: float
    down_m: float
    pre_gate_rejected: bool


@dataclass
class NavPathSample:
    time_sec: float
    north_m: float
    east_m: float
    down_m: float


def parse_float_grid(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_text_grid(text: str) -> List[str]:
    if not text.strip():
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[3]
    default_exe = workspace_root / "install" / "kf_gins" / "lib" / "kf_gins" / "kf_gins_csv_replay"
    default_out = workspace_root / "src" / "KF-GINS" / "results" / f"csv_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    ap = argparse.ArgumentParser(description="Auto-tune KF-GINS replay params from imu.csv and gnss.csv")
    ap.add_argument("--base-params", required=True, help="Base ROS2 params yaml")
    ap.add_argument("--imu-csv", required=True, help="Path to exported imu.csv")
    ap.add_argument("--gnss-csv", required=True, help="Path to exported gnss.csv")
    ap.add_argument("--out-dir", default=str(default_out), help="Output directory for all trials")
    ap.add_argument("--replay-exe", default=str(default_exe), help="Path to kf_gins_csv_replay executable")
    ap.add_argument("--node-name", default="kf_gins_node", help="Node name used in params yaml")
    ap.add_argument("--offset-grid", default="-0.10,-0.05,-0.02,0.00,0.02,0.05,0.10",
                    help="Replay GNSS time-shift candidates [s]")
    ap.add_argument("--filter-scheme-grid", default="",
                    help="Optional filter scheme candidates, e.g. ESKF,UKF,SR_UKF")
    ap.add_argument("--std-xy-grid", default="0.75,1.00,1.25,1.50,2.00",
                    help="Replay GNSS XY std scale candidates")
    ap.add_argument("--std-z-grid", default="0.75,1.00,1.25,1.50,2.00",
                    help="Replay GNSS Z std scale candidates")
    ap.add_argument("--initpos-xy-grid", default="0.5,1.0,1.5,2.0",
                    help="Initial position std XY candidates [m]")
    ap.add_argument("--initpos-z-grid", default="1.0,2.0,3.0,4.0",
                    help="Initial position std Z candidates [m]")
    ap.add_argument("--initvel-grid", default="0.1,0.3,0.5,1.0",
                    help="Initial velocity std candidates [m/s]")
    ap.add_argument("--auto-init-window-grid", default="",
                    help="Optional auto-init GNSS window candidates [s]")
    ap.add_argument("--auto-init-min-speed-grid", default="",
                    help="Optional auto-init min speed for yaw candidates [m/s]")
    ap.add_argument("--antlever-z-grid", default="",
                    help="Optional antenna leverarm Z candidates [m], e.g. -0.20,-0.15,-0.10")
    ap.add_argument("--save-nav-for-all", action="store_true",
                    help="Save replay_nav.csv for every trial instead of only the final best")
    ap.add_argument("--skip-plots", action="store_true",
                    help="Skip generating SVG trajectory/residual plots")
    ap.add_argument("--dry-run", action="store_true", help="Plan trials and write configs, but do not execute replay")
    return ap.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"Invalid params yaml: {path}")
    return data


def detect_node_key(data: Dict[str, Any], preferred: str) -> str:
    if preferred in data:
        return preferred
    if len(data) == 1:
        return next(iter(data.keys()))
    raise RuntimeError(f"Cannot find node key '{preferred}' in params yaml")


def get_ros_params(data: Dict[str, Any], node_key: str) -> Dict[str, Any]:
    node = data.get(node_key)
    if not isinstance(node, dict) or "ros__parameters" not in node:
        raise RuntimeError(f"Params yaml missing '{node_key}.ros__parameters'")
    params = node["ros__parameters"]
    if not isinstance(params, dict):
        raise RuntimeError(f"'{node_key}.ros__parameters' must be a mapping")
    return params


def set_nested_param(params: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = params
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def materialize_trial_yaml(base_data: Dict[str, Any], node_key: str, updates: Dict[str, Any], out_path: Path) -> None:
    data = copy.deepcopy(base_data)
    params = get_ros_params(data, node_key)
    for key, value in updates.items():
        set_nested_param(params, key, value)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def rmse(values: Iterable[float]) -> float:
    values = list(values)
    return math.sqrt(sum(v * v for v in values) / max(1, len(values)))


def percentile(values: List[float], p: float) -> float:
    values = sorted(values)
    idx = int(p * (len(values) - 1))
    return values[idx]


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_trial_meta(path: Path, result: TrialResult) -> None:
    write_json(
        path,
        {
            "stage": result.stage,
            "trial_name": result.trial_name,
            "trial_dir": str(result.trial_dir),
            "params_path": str(result.params_path),
            "metrics_path": str(result.metrics_path),
            "residuals_path": str(result.residuals_path),
            "console_path": str(result.console_path),
            "exit_code": result.exit_code,
            "success": result.success,
            "updates": result.updates,
            "metrics": result.metrics,
        },
    )


def wgs84_to_ecef(lat_rad: float, lon_rad: float, h_m: float) -> Tuple[float, float, float]:
    a = 6378137.0
    e2 = 6.69437999014e-3
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = (n * (1.0 - e2) + h_m) * sin_lat
    return x, y, z


def blh_to_local_ne(lat0_deg: float, lon0_deg: float, h0_m: float, lat_deg: float, lon_deg: float, h_m: float) -> Tuple[float, float]:
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x0, y0, z0 = wgs84_to_ecef(lat0, lon0, h0_m)
    x, y, z = wgs84_to_ecef(lat, lon, h_m)
    dx = x - x0
    dy = y - y0
    dz = z - z0
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)
    east = -sin_lon0 * dx + cos_lon0 * dy
    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    return north, east


def load_gnss_path(gnss_csv: Path, skip_pregate_rejected: bool) -> List[GnssPathSample]:
    samples: List[GnssPathSample] = []
    with gnss_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return samples
    lat0 = float(rows[0]["latitude_deg"])
    lon0 = float(rows[0]["longitude_deg"])
    h0 = float(rows[0]["altitude_m"])
    for row in rows:
        pre_gate_rejected = float(row["pre_gate_rejected"]) > 0.5
        if skip_pregate_rejected and pre_gate_rejected:
            continue
        north_m, east_m = blh_to_local_ne(
            lat0_deg=lat0,
            lon0_deg=lon0,
            h0_m=h0,
            lat_deg=float(row["latitude_deg"]),
            lon_deg=float(row["longitude_deg"]),
            h_m=float(row["altitude_m"]),
        )
        down_m = h0 - float(row["altitude_m"])
        samples.append(
            GnssPathSample(
                time_sec=float(row["fused_time_sec"]),
                north_m=north_m,
                east_m=east_m,
                down_m=down_m,
                pre_gate_rejected=pre_gate_rejected,
            )
        )
    return samples


def load_nav_path(nav_csv: Path) -> List[NavPathSample]:
    samples: List[NavPathSample] = []
    with nav_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                NavPathSample(
                    time_sec=float(row["time_sec"]),
                    north_m=float(row["local_north_m"]),
                    east_m=float(row["local_east_m"]),
                    down_m=float(row["local_down_m"]),
                )
            )
    return samples


def compute_metrics_from_residuals(csv_path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        raise RuntimeError(f"No residual rows found in {csv_path}")

    xy = [r["res_xy_m"] for r in rows]
    d3 = [r["res_3d_m"] for r in rows]
    z = [abs(r["res_d_m"]) for r in rows]
    dt = [r["match_dt_sec"] for r in rows]

    return {
        "residual_matched_count": len(rows),
        "residual_rmse_xy_m": rmse(xy),
        "residual_p95_xy_m": percentile(xy, 0.95),
        "residual_max_xy_m": max(xy),
        "residual_rmse_z_m": rmse(z),
        "residual_p95_abs_z_m": percentile(z, 0.95),
        "residual_max_abs_z_m": max(z),
        "residual_rmse_3d_m": rmse(d3),
        "residual_p95_3d_m": percentile(d3, 0.95),
        "residual_max_3d_m": max(d3),
        "max_match_dt_s": max(dt),
    }


def interp_nav_samples(nav_samples: List[NavPathSample], query_times: List[float]) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    if not nav_samples:
        return [], []
    nav_times = [s.time_sec for s in nav_samples]
    nav_n = [s.north_m for s in nav_samples]
    nav_e = [s.east_m for s in nav_samples]
    nav_d = [s.down_m for s in nav_samples]
    matched_positions: List[Tuple[float, float, float]] = []
    matched_dt: List[float] = []
    j = 0
    last = len(nav_samples) - 1
    for qt in query_times:
        if qt < nav_times[0] or qt > nav_times[-1]:
            continue
        while j + 1 < len(nav_times) and nav_times[j + 1] < qt:
            j += 1
        if j >= last:
            continue
        t0 = nav_times[j]
        t1 = nav_times[j + 1]
        if t1 <= t0:
            continue
        alpha = (qt - t0) / (t1 - t0)
        matched_positions.append((
            nav_n[j] + alpha * (nav_n[j + 1] - nav_n[j]),
            nav_e[j] + alpha * (nav_e[j + 1] - nav_e[j]),
            nav_d[j] + alpha * (nav_d[j + 1] - nav_d[j]),
        ))
        matched_dt.append(min(abs(qt - t0), abs(qt - t1)))
    return matched_positions, matched_dt


def find_nearest_index(times: List[float], target: float) -> Tuple[int, float]:
    lo = 0
    hi = len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return 0, abs(times[0] - target)
    if lo >= len(times):
        idx = len(times) - 1
        return idx, abs(times[idx] - target)
    prev_idx = lo - 1
    if abs(times[lo] - target) < abs(times[prev_idx] - target):
        return lo, abs(times[lo] - target)
    return prev_idx, abs(times[prev_idx] - target)


def compute_rpe_metrics(
    times: List[float],
    nav_positions: List[Tuple[float, float, float]],
    gt_positions: List[Tuple[float, float, float]],
    delta_sec: float,
    tolerance_sec: float,
) -> Dict[str, Any]:
    err_xy: List[float] = []
    err_3d: List[float] = []
    err_z: List[float] = []
    if len(times) < 2:
        return {
            "rpe_delta_sec": delta_sec,
            "rpe_pair_count": 0,
            "rpe_rmse_xy_m": math.inf,
            "rpe_p95_xy_m": math.inf,
            "rpe_max_xy_m": math.inf,
            "rpe_rmse_z_m": math.inf,
            "rpe_p95_abs_z_m": math.inf,
            "rpe_max_abs_z_m": math.inf,
            "rpe_rmse_3d_m": math.inf,
            "rpe_p95_3d_m": math.inf,
            "rpe_max_3d_m": math.inf,
        }

    for i, t0 in enumerate(times):
        target = t0 + delta_sec
        if target > times[-1]:
            break
        j, dt = find_nearest_index(times, target)
        if j <= i or dt > tolerance_sec:
            continue
        dn_nav = nav_positions[j][0] - nav_positions[i][0]
        de_nav = nav_positions[j][1] - nav_positions[i][1]
        dd_nav = nav_positions[j][2] - nav_positions[i][2]
        dn_gt = gt_positions[j][0] - gt_positions[i][0]
        de_gt = gt_positions[j][1] - gt_positions[i][1]
        dd_gt = gt_positions[j][2] - gt_positions[i][2]
        en = dn_nav - dn_gt
        ee = de_nav - de_gt
        ed = dd_nav - dd_gt
        xy = math.hypot(ee, en)
        d3 = math.sqrt(en * en + ee * ee + ed * ed)
        err_xy.append(xy)
        err_3d.append(d3)
        err_z.append(abs(ed))

    if not err_xy:
        return {
            "rpe_delta_sec": delta_sec,
            "rpe_pair_count": 0,
            "rpe_rmse_xy_m": math.inf,
            "rpe_p95_xy_m": math.inf,
            "rpe_max_xy_m": math.inf,
            "rpe_rmse_z_m": math.inf,
            "rpe_p95_abs_z_m": math.inf,
            "rpe_max_abs_z_m": math.inf,
            "rpe_rmse_3d_m": math.inf,
            "rpe_p95_3d_m": math.inf,
            "rpe_max_3d_m": math.inf,
        }

    return {
        "rpe_delta_sec": delta_sec,
        "rpe_pair_count": len(err_xy),
        "rpe_rmse_xy_m": rmse(err_xy),
        "rpe_p95_xy_m": percentile(err_xy, 0.95),
        "rpe_max_xy_m": max(err_xy),
        "rpe_rmse_z_m": rmse(err_z),
        "rpe_p95_abs_z_m": percentile(err_z, 0.95),
        "rpe_max_abs_z_m": max(err_z),
        "rpe_rmse_3d_m": rmse(err_3d),
        "rpe_p95_3d_m": percentile(err_3d, 0.95),
        "rpe_max_3d_m": max(err_3d),
    }


def compute_ape_rpe_metrics(
    gnss_csv: Path,
    nav_csv: Path,
    skip_pregate_rejected: bool,
    rpe_delta_sec: float = 1.0,
    rpe_tolerance_sec: float = 0.15,
) -> Dict[str, Any]:
    gnss_samples = load_gnss_path(gnss_csv, skip_pregate_rejected=skip_pregate_rejected)
    nav_samples = load_nav_path(nav_csv)
    if not gnss_samples:
        raise RuntimeError(f"No GNSS path samples found in {gnss_csv}")
    if not nav_samples:
        raise RuntimeError(f"No nav path samples found in {nav_csv}")

    query_times = [s.time_sec for s in gnss_samples]
    nav_interp, nav_dt = interp_nav_samples(nav_samples, query_times)
    gt_positions: List[Tuple[float, float, float]] = []
    matched_times: List[float] = []
    nav_time_min = nav_samples[0].time_sec
    nav_time_max = nav_samples[-1].time_sec
    for s in gnss_samples:
        if nav_time_min <= s.time_sec <= nav_time_max:
            gt_positions.append((s.north_m, s.east_m, s.down_m))
            matched_times.append(s.time_sec)

    matched_count = min(len(nav_interp), len(gt_positions))
    nav_interp = nav_interp[:matched_count]
    gt_positions = gt_positions[:matched_count]
    matched_times = matched_times[:matched_count]
    nav_dt = nav_dt[:matched_count]

    if matched_count == 0:
        raise RuntimeError("No overlapping nav/GNSS samples for APE evaluation")

    ape_xy: List[float] = []
    ape_3d: List[float] = []
    ape_z: List[float] = []
    for nav_p, gt_p in zip(nav_interp, gt_positions):
        en = nav_p[0] - gt_p[0]
        ee = nav_p[1] - gt_p[1]
        ed = nav_p[2] - gt_p[2]
        ape_xy.append(math.hypot(ee, en))
        ape_3d.append(math.sqrt(en * en + ee * ee + ed * ed))
        ape_z.append(abs(ed))

    metrics = {
        "ape_matched_count": matched_count,
        "ape_rmse_xy_m": rmse(ape_xy),
        "ape_p95_xy_m": percentile(ape_xy, 0.95),
        "ape_max_xy_m": max(ape_xy),
        "ape_rmse_z_m": rmse(ape_z),
        "ape_p95_abs_z_m": percentile(ape_z, 0.95),
        "ape_max_abs_z_m": max(ape_z),
        "ape_rmse_3d_m": rmse(ape_3d),
        "ape_p95_3d_m": percentile(ape_3d, 0.95),
        "ape_max_3d_m": max(ape_3d),
        "ape_max_match_dt_s": max(nav_dt) if nav_dt else math.inf,
    }
    metrics.update(compute_rpe_metrics(matched_times, nav_interp, gt_positions, rpe_delta_sec, rpe_tolerance_sec))
    metrics["score"] = (
        metrics["ape_rmse_xy_m"]
        + 0.50 * metrics["rpe_rmse_xy_m"]
        + 0.10 * metrics["ape_p95_xy_m"]
        + 0.05 * metrics["rpe_p95_xy_m"]
        + 0.05 * metrics["ape_max_match_dt_s"]
    )
    return metrics


def load_residual_points(residual_csv: Path) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    with residual_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row["res_e_m"]), float(row["res_n_m"])))
    return points


def sample_polyline(points: List[Tuple[float, float]], target_count: int = 4000) -> List[Tuple[float, float]]:
    if len(points) <= target_count:
        return points
    step = max(1, len(points) // target_count)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def svg_bounds(series_list: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for series in series_list:
        for x, y in series:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return -1.0, 1.0, -1.0, 1.0
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if abs(max_x - min_x) < 1e-9:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-9:
        min_y -= 1.0
        max_y += 1.0
    return min_x, max_x, min_y, max_y


def make_transform(min_x: float, max_x: float, min_y: float, max_y: float, width: int, height: int,
                   pad_left: int, pad_right: int, pad_top: int, pad_bottom: int):
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min(plot_w / span_x, plot_h / span_y)
    offset_x = pad_left + 0.5 * (plot_w - span_x * scale)
    offset_y = pad_top + 0.5 * (plot_h - span_y * scale)

    def transform(x: float, y: float) -> Tuple[float, float]:
        px = offset_x + (x - min_x) * scale
        py = height - pad_bottom - (y - min_y) * scale - 0.5 * (plot_h - span_y * scale)
        return px, py

    return transform


def format_polyline(points: List[Tuple[float, float]], transform) -> str:
    coords = []
    for x, y in points:
        px, py = transform(x, y)
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def write_svg_xy_plot(path: Path, title: str, subtitle: str, series: List[Tuple[str, str, List[Tuple[float, float]], str]]) -> None:
    width = 1200
    height = 900
    pad_left = 90
    pad_right = 40
    pad_top = 90
    pad_bottom = 70
    sampled_series = [(label, color, sample_polyline(points), style) for label, color, points, style in series if points]
    min_x, max_x, min_y, max_y = svg_bounds([points for _, _, points, _ in sampled_series])
    transform = make_transform(min_x, max_x, min_y, max_y, width, height, pad_left, pad_right, pad_top, pad_bottom)

    axis_lines = []
    if min_x <= 0.0 <= max_x:
        x0, y1 = transform(0.0, min_y)
        _, y2 = transform(0.0, max_y)
        axis_lines.append(f'<line x1="{x0:.2f}" y1="{y1:.2f}" x2="{x0:.2f}" y2="{y2:.2f}" stroke="#d0d7de" stroke-width="1"/>')
    if min_y <= 0.0 <= max_y:
        x1, y0 = transform(min_x, 0.0)
        x2, _ = transform(max_x, 0.0)
        axis_lines.append(f'<line x1="{x1:.2f}" y1="{y0:.2f}" x2="{x2:.2f}" y2="{y0:.2f}" stroke="#d0d7de" stroke-width="1"/>')

    shapes = []
    legend_items = []
    legend_y = 36
    for idx, (label, color, points, style) in enumerate(sampled_series):
        if style == "scatter":
            for x, y in points:
                px, py = transform(x, y)
                shapes.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.8" fill="{color}" fill-opacity="0.55"/>')
            legend_items.append(f'<circle cx="930" cy="{legend_y + idx * 24}" r="4" fill="{color}" fill-opacity="0.8"/>')
        else:
            stroke_dash = ' stroke-dasharray="8 6"' if style == "dashed" else ""
            shapes.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{stroke_dash} points="{format_polyline(points, transform)}"/>'
            )
            ly = legend_y + idx * 24
            legend_items.append(f'<line x1="910" y1="{ly}" x2="950" y2="{ly}" stroke="{color}" stroke-width="3"{stroke_dash}/>')
        ly = legend_y + idx * 24
        legend_items.append(f'<text x="960" y="{ly + 5}" font-size="16" fill="#111827">{label}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{pad_left}" y="36" font-size="28" font-family="monospace" fill="#111827">{title}</text>
<text x="{pad_left}" y="62" font-size="16" font-family="monospace" fill="#4b5563">{subtitle}</text>
<rect x="{pad_left}" y="{pad_top}" width="{width - pad_left - pad_right}" height="{height - pad_top - pad_bottom}" fill="#fafafa" stroke="#d1d5db"/>
{''.join(axis_lines)}
{''.join(shapes)}
<text x="{width / 2:.1f}" y="{height - 18}" font-size="18" font-family="monospace" text-anchor="middle" fill="#111827">East [m]</text>
<text x="24" y="{height / 2:.1f}" font-size="18" font-family="monospace" text-anchor="middle" fill="#111827" transform="rotate(-90 24 {height / 2:.1f})">North [m]</text>
{''.join(legend_items)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def generate_best_plots(out_dir: Path, gnss_csv: Path, nav_csv: Path, residual_csv: Path, skip_pregate_rejected: bool,
                        metrics: Dict[str, Any]) -> None:
    gnss_samples = load_gnss_path(gnss_csv, skip_pregate_rejected=skip_pregate_rejected)
    nav_samples = load_nav_path(nav_csv)
    residual_points = load_residual_points(residual_csv)
    nav_xy = [(s.east_m, s.north_m) for s in nav_samples]
    if nav_xy:
        write_svg_xy_plot(
            out_dir / "best_nav_trajectory.svg",
            title="KF-GINS Replay Trajectory",
            subtitle="Replay navigation trajectory in local EN plane",
            series=[
                ("Replay Nav", "#dc2626", nav_xy, "solid"),
            ],
        )
    if gnss_samples and nav_xy:
        gnss_xy = [(s.east_m, s.north_m) for s in gnss_samples]
        subtitle = (
            f"APExy={metrics['ape_rmse_xy_m']:.3f} m, RPExy={metrics['rpe_rmse_xy_m']:.3f} m, "
            f"APE3d={metrics['ape_rmse_3d_m']:.3f} m"
        )
        write_svg_xy_plot(
            out_dir / "best_nav_vs_gnss_trajectory.svg",
            title="KF-GINS vs GNSS Trajectory",
            subtitle=subtitle,
            series=[
                ("GNSS", "#2563eb", gnss_xy, "dashed"),
                ("Replay Nav", "#dc2626", nav_xy, "solid"),
            ],
        )
        # Backward-compatible filename.
        write_svg_xy_plot(
            out_dir / "best_xy_trajectory.svg",
            title="KF-GINS vs GNSS Trajectory",
            subtitle=subtitle,
            series=[
                ("GNSS", "#2563eb", gnss_xy, "dashed"),
                ("Replay Nav", "#dc2626", nav_xy, "solid"),
            ],
        )
    if residual_points:
        write_svg_xy_plot(
            out_dir / "best_xy_residual.svg",
            title="KF-GINS Replay XY Residual",
            subtitle=f"Residual points in local EN plane, count={len(residual_points)}",
            series=[
                ("Residual E/N", "#059669", residual_points, "scatter"),
            ],
        )


def run_replay_trial(
    replay_exe: Path,
    node_name: str,
    imu_csv: Path,
    gnss_csv: Path,
    trial_dir: Path,
    params_path: Path,
    save_nav: bool,
    dry_run: bool,
) -> TrialResult:
    residuals_path = trial_dir / "replay_residuals.csv"
    nav_path = trial_dir / "replay_nav.csv"
    metrics_path = trial_dir / "metrics.json"
    console_path = trial_dir / "console.log"

    cmd = [
        str(replay_exe),
        "--ros-args",
        "-r",
        f"__node:={node_name}",
        "--params-file",
        str(params_path),
        "-p",
        f"imu_csv:={imu_csv}",
        "-p",
        f"gnss_csv:={gnss_csv}",
        "-p",
        f"replay_residual_csv:={residuals_path}",
        "-p",
        f"replay_nav_csv:={nav_path}",
    ]

    if dry_run:
        console_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        metrics = {
            "ape_matched_count": 0,
            "ape_rmse_xy_m": math.nan,
            "ape_p95_xy_m": math.nan,
            "ape_max_xy_m": math.nan,
            "ape_rmse_z_m": math.nan,
            "ape_p95_abs_z_m": math.nan,
            "ape_max_abs_z_m": math.nan,
            "ape_rmse_3d_m": math.nan,
            "ape_p95_3d_m": math.nan,
            "ape_max_3d_m": math.nan,
            "ape_max_match_dt_s": math.nan,
            "rpe_delta_sec": math.nan,
            "rpe_pair_count": 0,
            "rpe_rmse_xy_m": math.nan,
            "rpe_p95_xy_m": math.nan,
            "rpe_max_xy_m": math.nan,
            "rpe_rmse_z_m": math.nan,
            "rpe_p95_abs_z_m": math.nan,
            "rpe_max_abs_z_m": math.nan,
            "rpe_rmse_3d_m": math.nan,
            "rpe_p95_3d_m": math.nan,
            "rpe_max_3d_m": math.nan,
            "score": math.inf,
            "dry_run": True,
        }
        write_json(metrics_path, metrics)
        return TrialResult("", "", trial_dir, params_path, metrics_path, residuals_path, console_path, 0, False, metrics, {})

    env = os.environ.copy()
    ros_log_dir = trial_dir / "roslog"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    env["ROS_LOG_DIR"] = str(ros_log_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    console_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")

    success = proc.returncode == 0 and residuals_path.exists() and nav_path.exists()
    metrics: Dict[str, Any]
    if success:
        metrics = compute_ape_rpe_metrics(gnss_csv=gnss_csv, nav_csv=nav_path, skip_pregate_rejected=True)
        metrics.update(compute_metrics_from_residuals(residuals_path))
        if not save_nav and nav_path.exists():
            nav_path.unlink()
    else:
        metrics = {
            "ape_matched_count": 0,
            "ape_rmse_xy_m": math.inf,
            "ape_p95_xy_m": math.inf,
            "ape_max_xy_m": math.inf,
            "ape_rmse_z_m": math.inf,
            "ape_p95_abs_z_m": math.inf,
            "ape_max_abs_z_m": math.inf,
            "ape_rmse_3d_m": math.inf,
            "ape_p95_3d_m": math.inf,
            "ape_max_3d_m": math.inf,
            "ape_max_match_dt_s": math.inf,
            "rpe_delta_sec": math.inf,
            "rpe_pair_count": 0,
            "rpe_rmse_xy_m": math.inf,
            "rpe_p95_xy_m": math.inf,
            "rpe_max_xy_m": math.inf,
            "rpe_rmse_z_m": math.inf,
            "rpe_p95_abs_z_m": math.inf,
            "rpe_max_abs_z_m": math.inf,
            "rpe_rmse_3d_m": math.inf,
            "rpe_p95_3d_m": math.inf,
            "rpe_max_3d_m": math.inf,
            "residual_matched_count": 0,
            "residual_rmse_xy_m": math.inf,
            "residual_p95_xy_m": math.inf,
            "residual_max_xy_m": math.inf,
            "residual_rmse_z_m": math.inf,
            "residual_p95_abs_z_m": math.inf,
            "residual_max_abs_z_m": math.inf,
            "residual_rmse_3d_m": math.inf,
            "residual_p95_3d_m": math.inf,
            "residual_max_3d_m": math.inf,
            "max_match_dt_s": math.inf,
            "score": math.inf,
        }
    metrics["exit_code"] = proc.returncode
    write_json(metrics_path, metrics)
    return TrialResult("", "", trial_dir, params_path, metrics_path, residuals_path, console_path, proc.returncode, success, metrics, {})


def make_stage_candidates(args: argparse.Namespace, base_params: Dict[str, Any]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    candidates: List[Tuple[str, List[Dict[str, Any]]]] = []

    offset_values = parse_float_grid(args.offset_grid)
    if offset_values:
        candidates.append((
            "time_shift",
            [{"replay_gnss_time_shift_sec": v} for v in offset_values],
        ))

    filter_schemes = parse_text_grid(args.filter_scheme_grid)
    if filter_schemes:
        candidates.append((
            "filter_scheme",
            [{"filter_scheme": scheme} for scheme in filter_schemes],
        ))

    xy_scales = parse_float_grid(args.std_xy_grid)
    z_scales = parse_float_grid(args.std_z_grid)
    if xy_scales and z_scales:
        grid = []
        for xy_scale, z_scale in itertools.product(xy_scales, z_scales):
            grid.append({
                "replay_gnss_std_scale_xy": xy_scale,
                "replay_gnss_std_scale_z": z_scale,
            })
        candidates.append(("gnss_std_scale", grid))

    pos_xy_values = parse_float_grid(args.initpos_xy_grid)
    pos_z_values = parse_float_grid(args.initpos_z_grid)
    vel_values = parse_float_grid(args.initvel_grid)
    if pos_xy_values and pos_z_values and vel_values:
        grid = []
        for pos_xy, pos_z, vel in itertools.product(pos_xy_values, pos_z_values, vel_values):
            grid.append({
                "initposstd": [pos_xy, pos_xy, pos_z],
                "initvelstd": [vel, vel, vel],
            })
        candidates.append(("init_covariance", grid))

    antlever_z_values = parse_float_grid(args.antlever_z_grid)
    if antlever_z_values:
        antlever = base_params.get("antlever", [0.0, 0.0, 0.0])
        if not isinstance(antlever, list) or len(antlever) != 3:
            raise RuntimeError("Base params antlever must be a 3-element list")
        grid = []
        for z in antlever_z_values:
            grid.append({"antlever": [float(antlever[0]), float(antlever[1]), z]})
        candidates.append(("antlever_z", grid))

    auto_init_window_values = parse_float_grid(args.auto_init_window_grid)
    auto_init_speed_values = parse_float_grid(args.auto_init_min_speed_grid)
    if auto_init_window_values or auto_init_speed_values:
        current_window = float(base_params.get("auto_init_gnss_window_sec", 8.0))
        current_speed = float(base_params.get("auto_init_min_speed_for_yaw", 1.0))
        if not auto_init_window_values:
            auto_init_window_values = [current_window]
        if not auto_init_speed_values:
            auto_init_speed_values = [current_speed]
        grid = []
        for window_sec, min_speed in itertools.product(auto_init_window_values, auto_init_speed_values):
            grid.append({
                "auto_init_enable": True,
                "auto_init_pos_only": False,
                "auto_init_gnss_window_sec": window_sec,
                "auto_init_min_speed_for_yaw": min_speed,
            })
        candidates.append(("auto_init_yaw", grid))

    return candidates


def trial_sort_key(result: TrialResult) -> Tuple[float, float, float, float]:
    m = result.metrics
    return (
        float(m.get("score", math.inf)),
        float(m.get("ape_rmse_xy_m", math.inf)),
        float(m.get("rpe_rmse_xy_m", math.inf)),
        float(m.get("ape_p95_xy_m", math.inf)),
    )


def update_summary(summary_rows: List[Dict[str, Any]], result: TrialResult) -> None:
    row = {
        "stage": result.stage,
        "trial_name": result.trial_name,
        "trial_dir": str(result.trial_dir),
        "params_path": str(result.params_path),
        "metrics_path": str(result.metrics_path),
        "exit_code": result.exit_code,
        "success": int(result.success),
        "score": result.metrics.get("score"),
        "ape_rmse_xy_m": result.metrics.get("ape_rmse_xy_m"),
        "ape_p95_xy_m": result.metrics.get("ape_p95_xy_m"),
        "ape_max_xy_m": result.metrics.get("ape_max_xy_m"),
        "ape_rmse_z_m": result.metrics.get("ape_rmse_z_m"),
        "ape_p95_abs_z_m": result.metrics.get("ape_p95_abs_z_m"),
        "ape_max_abs_z_m": result.metrics.get("ape_max_abs_z_m"),
        "ape_rmse_3d_m": result.metrics.get("ape_rmse_3d_m"),
        "ape_p95_3d_m": result.metrics.get("ape_p95_3d_m"),
        "ape_max_3d_m": result.metrics.get("ape_max_3d_m"),
        "ape_max_match_dt_s": result.metrics.get("ape_max_match_dt_s"),
        "rpe_delta_sec": result.metrics.get("rpe_delta_sec"),
        "rpe_pair_count": result.metrics.get("rpe_pair_count"),
        "rpe_rmse_xy_m": result.metrics.get("rpe_rmse_xy_m"),
        "rpe_p95_xy_m": result.metrics.get("rpe_p95_xy_m"),
        "rpe_max_xy_m": result.metrics.get("rpe_max_xy_m"),
        "rpe_rmse_3d_m": result.metrics.get("rpe_rmse_3d_m"),
        "rpe_p95_3d_m": result.metrics.get("rpe_p95_3d_m"),
        "rpe_max_3d_m": result.metrics.get("rpe_max_3d_m"),
        "residual_rmse_xy_m": result.metrics.get("residual_rmse_xy_m"),
        "residual_p95_xy_m": result.metrics.get("residual_p95_xy_m"),
        "residual_max_xy_m": result.metrics.get("residual_max_xy_m"),
        "residual_rmse_3d_m": result.metrics.get("residual_rmse_3d_m"),
        "residual_p95_3d_m": result.metrics.get("residual_p95_3d_m"),
        "residual_max_3d_m": result.metrics.get("residual_max_3d_m"),
        "max_match_dt_s": result.metrics.get("max_match_dt_s"),
        "updates_json": json.dumps(result.updates, ensure_ascii=False),
    }
    summary_rows.append(row)


def finalize_trial(result: TrialResult) -> None:
    write_trial_meta(result.trial_dir / "trial_meta.json", result)


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    base_params_path = Path(args.base_params).resolve()
    imu_csv = Path(args.imu_csv).resolve()
    gnss_csv = Path(args.gnss_csv).resolve()
    replay_exe = Path(args.replay_exe).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not base_params_path.exists():
        raise RuntimeError(f"Base params file not found: {base_params_path}")
    if not imu_csv.exists():
        raise RuntimeError(f"IMU csv not found: {imu_csv}")
    if not gnss_csv.exists():
        raise RuntimeError(f"GNSS csv not found: {gnss_csv}")
    if not replay_exe.exists() and not args.dry_run:
        raise RuntimeError(f"Replay executable not found: {replay_exe}")

    out_dir.mkdir(parents=True, exist_ok=True)
    trials_root = out_dir / "trials"
    trials_root.mkdir(exist_ok=True)

    base_data = load_yaml(base_params_path)
    node_key = detect_node_key(base_data, args.node_name)
    base_params = get_ros_params(base_data, node_key)

    shutil.copy2(base_params_path, out_dir / "base_params.yaml")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_params": str(base_params_path),
        "imu_csv": str(imu_csv),
        "gnss_csv": str(gnss_csv),
        "replay_exe": str(replay_exe),
        "node_key": node_key,
        "dry_run": args.dry_run,
        "stages": [],
    }

    stage_candidates = make_stage_candidates(args, base_params)
    manifest["stages"] = [{"name": name, "trial_count": len(cands)} for name, cands in stage_candidates]
    write_json(out_dir / "manifest.json", manifest)

    summary_rows: List[Dict[str, Any]] = []

    baseline_dir = trials_root / "baseline_000"
    baseline_dir.mkdir(exist_ok=True)
    baseline_params_path = baseline_dir / "params.yaml"
    materialize_trial_yaml(base_data, node_key, {}, baseline_params_path)
    baseline_result = run_replay_trial(
        replay_exe=replay_exe,
        node_name=node_key,
        imu_csv=imu_csv,
        gnss_csv=gnss_csv,
        trial_dir=baseline_dir,
        params_path=baseline_params_path,
        save_nav=args.save_nav_for_all,
        dry_run=args.dry_run,
    )
    baseline_result.stage = "baseline"
    baseline_result.trial_name = "baseline"
    baseline_result.updates = {}
    finalize_trial(baseline_result)
    update_summary(summary_rows, baseline_result)
    write_summary_csv(out_dir / "summary.csv", summary_rows)

    if not args.dry_run and not baseline_result.success:
        raise RuntimeError("Baseline replay trial failed; aborting tuning")

    best_yaml_source = baseline_params_path
    best_result = baseline_result

    for stage_idx, (stage_name, candidates) in enumerate(stage_candidates, start=1):
        stage_dir = trials_root / f"{stage_idx:02d}_{stage_name}"
        stage_dir.mkdir(exist_ok=True)
        stage_results: List[TrialResult] = []

        for trial_idx, updates in enumerate(candidates, start=1):
            trial_name = f"trial_{trial_idx:03d}"
            trial_dir = stage_dir / trial_name
            trial_dir.mkdir(exist_ok=True)

            params_path = trial_dir / "params.yaml"
            current_best_data = load_yaml(best_yaml_source)
            materialize_trial_yaml(current_best_data, node_key, updates, params_path)

            result = run_replay_trial(
                replay_exe=replay_exe,
                node_name=node_key,
                imu_csv=imu_csv,
                gnss_csv=gnss_csv,
                trial_dir=trial_dir,
                params_path=params_path,
                save_nav=args.save_nav_for_all,
                dry_run=args.dry_run,
            )
            result.stage = stage_name
            result.trial_name = trial_name
            result.updates = updates
            finalize_trial(result)
            stage_results.append(result)
            update_summary(summary_rows, result)
            write_summary_csv(out_dir / "summary.csv", summary_rows)

        successful = [r for r in stage_results if r.success]
        if args.dry_run:
            continue
        if not successful:
            raise RuntimeError(f"No successful trials in stage '{stage_name}'")

        stage_best = min(successful, key=trial_sort_key)
        best_yaml_source = stage_best.params_path
        best_result = stage_best
        shutil.copy2(stage_best.params_path, out_dir / f"best_after_{stage_idx:02d}_{stage_name}.yaml")
        write_json(out_dir / f"best_after_{stage_idx:02d}_{stage_name}.json", stage_best.metrics)

    if args.dry_run:
        print(f"[tune_csv_replay] Dry run complete. Planned outputs in {out_dir}")
        return 0

    shutil.copy2(best_yaml_source, out_dir / "best_params.yaml")
    write_json(out_dir / "best_metrics.json", best_result.metrics)
    if best_result.residuals_path.exists():
        shutil.copy2(best_result.residuals_path, out_dir / "best_replay_residuals.csv")

    best_export_dir = out_dir / "best_export"
    best_export_dir.mkdir(exist_ok=True)
    best_export_result = run_replay_trial(
        replay_exe=replay_exe,
        node_name=node_key,
        imu_csv=imu_csv,
        gnss_csv=gnss_csv,
        trial_dir=best_export_dir,
        params_path=best_yaml_source,
        save_nav=True,
        dry_run=False,
    )
    best_export_result.stage = "best_export"
    best_export_result.trial_name = "best_export"
    best_export_result.updates = best_result.updates
    finalize_trial(best_export_result)
    if best_export_result.success:
        if best_export_result.residuals_path.exists():
            shutil.copy2(best_export_result.residuals_path, out_dir / "best_replay_residuals.csv")
        best_nav = best_export_result.trial_dir / "replay_nav.csv"
        if best_nav.exists():
            shutil.copy2(best_nav, out_dir / "best_replay_nav.csv")
            if not args.skip_plots:
                generate_best_plots(
                    out_dir=out_dir,
                    gnss_csv=gnss_csv,
                    nav_csv=best_nav,
                    residual_csv=best_export_result.residuals_path,
                    skip_pregate_rejected=True,
                    metrics=best_export_result.metrics,
                )
    else:
        print("[tune_csv_replay] Warning: failed to export best replay_nav.csv", file=sys.stderr)

    print(f"[tune_csv_replay] Tuning complete. Best score={best_result.metrics['score']:.6f}")
    print(f"[tune_csv_replay] Best params: {out_dir / 'best_params.yaml'}")
    print(f"[tune_csv_replay] Summary: {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[tune_csv_replay] {exc}", file=sys.stderr)
        raise SystemExit(1)
