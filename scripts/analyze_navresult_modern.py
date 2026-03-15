#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013
D2R = math.pi / 180.0


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q))


def radiusmn(lat_rad: float) -> tuple[float, float]:
    s2 = math.sin(lat_rad) ** 2
    t = 1.0 - WGS84_E1 * s2
    st = math.sqrt(t)
    rm = WGS84_RA * (1.0 - WGS84_E1) / (st * t)
    rn = WGS84_RA / st
    return rm, rn


def read_nav_file(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 11:
        raise RuntimeError(f"Expected >=11 columns in {path}, got {arr.shape[1]}")
    keep = np.concatenate(([0], np.where(np.diff(arr[:, 1]) > 0)[0] + 1))
    return arr[keep]


def blh_to_enu(blh_deg_m: np.ndarray, origin_deg_m: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(blh_deg_m[:, 0])
    lon = np.deg2rad(blh_deg_m[:, 1])
    h = blh_deg_m[:, 2]
    lat0 = math.radians(float(origin_deg_m[0]))
    lon0 = math.radians(float(origin_deg_m[1]))
    h0 = float(origin_deg_m[2])
    rm, rn = radiusmn(lat0)
    north = (lat - lat0) * (rm + h0)
    east = (lon - lon0) * (rn + h0) * math.cos(lat0)
    up = h - h0
    return np.column_stack((east, north, up))


def nav_to_enu(nav: np.ndarray, origin_deg_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = nav[:, 1].copy()
    blh = nav[:, 2:5].copy()
    pos = blh_to_enu(blh, origin_deg_m)

    # nav columns are [week, time, lat, lon, h, vn, ve, vd, roll, pitch, yaw]
    vel = np.column_stack((nav[:, 6], nav[:, 5], -nav[:, 7]))
    return t, pos, vel


def align_reference_axis(
    nav_t: np.ndarray,
    nav_pos: np.ndarray,
    nav_vel: np.ndarray,
    ref_t: np.ndarray,
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    time_shift_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nav_t = nav_t + float(time_shift_sec)
    t0 = max(nav_t[0], ref_t[0])
    t1 = min(nav_t[-1], ref_t[-1])
    if t1 <= t0:
        raise RuntimeError("No overlapping time range between navresult and truth")

    keep = (ref_t >= t0) & (ref_t <= t1)
    if np.count_nonzero(keep) < 2:
        raise RuntimeError("Overlapping segment is too short for evaluation")

    t = ref_t[keep]
    nav_pos_i = np.column_stack([np.interp(t, nav_t, nav_pos[:, k]) for k in range(3)])
    nav_vel_i = np.column_stack([np.interp(t, nav_t, nav_vel[:, k]) for k in range(3)])
    ref_pos_a = ref_pos[keep]
    ref_vel_a = ref_vel[keep]
    return t, nav_pos_i, nav_vel_i, ref_pos_a, ref_vel_a


def find_nearest_index(times: np.ndarray, target: float) -> tuple[int, float]:
    idx = np.searchsorted(times, target, side="left")
    idx = int(np.clip(idx, 0, len(times) - 1))
    if idx == 0:
        return 0, abs(times[0] - target)
    prev_idx = idx - 1
    if abs(times[idx] - target) < abs(times[prev_idx] - target):
        return idx, abs(times[idx] - target)
    return prev_idx, abs(times[prev_idx] - target)


def compute_rpe_metrics(t: np.ndarray, ref_xyz: np.ndarray, nav_xyz: np.ndarray, delta_sec: float, tolerance_sec: float) -> dict[str, float]:
    xy_errors = []
    z_errors = []
    d3_errors = []
    for i, t0 in enumerate(t):
        target = t0 + delta_sec
        if target > t[-1]:
            break
        j, dt = find_nearest_index(t, target)
        if j <= i or dt > tolerance_sec:
            continue
        ref_rel = ref_xyz[j] - ref_xyz[i]
        nav_rel = nav_xyz[j] - nav_xyz[i]
        err = nav_rel - ref_rel
        xy_errors.append(float(np.linalg.norm(err[:2])))
        z_errors.append(abs(float(err[2])))
        d3_errors.append(float(np.linalg.norm(err)))
    if not xy_errors:
        raise RuntimeError("No valid RPE pairs found; adjust time shift or RPE delta/tolerance")
    return {
        "rpe_delta_sec": float(delta_sec),
        "rpe_pair_count": len(xy_errors),
        "rpe_rmse_xy_m": rmse(np.asarray(xy_errors)),
        "rpe_p95_xy_m": percentile(np.asarray(xy_errors), 95),
        "rpe_max_xy_m": float(np.max(xy_errors)),
        "rpe_rmse_3d_m": rmse(np.asarray(d3_errors)),
        "rpe_p95_3d_m": percentile(np.asarray(d3_errors), 95),
        "rpe_max_3d_m": float(np.max(d3_errors)),
        "rpe_rmse_z_m": rmse(np.asarray(z_errors)),
    }


def save_metrics_csv(path: Path, metrics: dict[str, float | int | str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def save_xy_error_csv(path: Path, t: np.ndarray, err_xyz: np.ndarray) -> None:
    err_xy = np.linalg.norm(err_xyz[:, :2], axis=1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec", "err_east_m", "err_north_m", "err_up_m", "err_xy_m", "err_3d_m"])
        for i in range(len(t)):
            writer.writerow(
                [
                    f"{t[i]:.9f}",
                    f"{err_xyz[i, 0]:.6f}",
                    f"{err_xyz[i, 1]:.6f}",
                    f"{err_xyz[i, 2]:.6f}",
                    f"{err_xy[i]:.6f}",
                    f"{float(np.linalg.norm(err_xyz[i])):.6f}",
                ]
            )


def save_trajectory_plot(path: Path, ref_pos: np.ndarray, nav_pos: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], label="truth", linewidth=1.8)
    ax.plot(nav_pos[:, 0], nav_pos[:, 1], label="navresult", linewidth=1.3)
    ax.scatter(ref_pos[0, 0], ref_pos[0, 1], c="green", s=30, marker="o", label="start")
    ax.scatter(ref_pos[-1, 0], ref_pos[-1, 1], c="red", s=40, marker="x", label="end")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Trajectory vs Truth")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, format="svg")
    plt.close(fig)


def save_xy_error_plot(path: Path, t: np.ndarray, err_xyz: np.ndarray) -> None:
    err_xy = np.linalg.norm(err_xyz[:, :2], axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t - t[0], err_xy, linewidth=1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("XY error [m]")
    ax.set_title("XY Error vs Truth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KF-GINS navresult.nav with modern APE/RPE metrics")
    parser.add_argument("--navresult", required=True, type=Path, help="KF_GINS_Navresult.nav path")
    parser.add_argument("--truth-nav", required=True, type=Path, help="truth.nav path")
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory")
    parser.add_argument("--time-shift-sec", type=float, default=0.0, help="Additional time shift applied to navresult time")
    parser.add_argument("--rpe-delta-sec", type=float, default=1.0, help="RPE delta interval [s]")
    parser.add_argument("--rpe-tolerance-sec", type=float, default=0.15, help="RPE time matching tolerance [s]")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    nav_raw = read_nav_file(args.navresult)
    truth_raw = read_nav_file(args.truth_nav)
    origin_deg_m = truth_raw[0, 2:5].copy()

    nav_t, nav_pos, nav_vel = nav_to_enu(nav_raw, origin_deg_m)
    truth_t, truth_pos, truth_vel = nav_to_enu(truth_raw, origin_deg_m)

    t, nav_pos_i, nav_vel_i, truth_pos_a, truth_vel_a = align_reference_axis(
        nav_t,
        nav_pos,
        nav_vel,
        truth_t,
        truth_pos,
        truth_vel,
        time_shift_sec=args.time_shift_sec,
    )

    err_pos = nav_pos_i - truth_pos_a
    err_vel = nav_vel_i - truth_vel_a
    err_xy = np.linalg.norm(err_pos[:, :2], axis=1)
    err_3d = np.linalg.norm(err_pos, axis=1)

    metrics: dict[str, float | int | str] = {
        "aligned_count": int(len(t)),
        "time_basis": "truth_axis",
        "time_shift_sec": float(args.time_shift_sec),
        "navresult": str(args.navresult.resolve()),
        "truth_nav": str(args.truth_nav.resolve()),
        "ape_rmse_xy_m": rmse(err_xy),
        "ape_p95_xy_m": percentile(err_xy, 95),
        "ape_max_xy_m": float(np.max(err_xy)),
        "ape_rmse_3d_m": rmse(err_3d),
        "ape_p95_3d_m": percentile(err_3d, 95),
        "ape_max_3d_m": float(np.max(err_3d)),
        "vel_rmse_mps": rmse(np.linalg.norm(err_vel, axis=1)),
        "vel_p95_mps": percentile(np.linalg.norm(err_vel, axis=1), 95),
        "vel_max_mps": float(np.max(np.linalg.norm(err_vel, axis=1))),
    }
    metrics.update(
        compute_rpe_metrics(
            t,
            truth_pos_a,
            nav_pos_i,
            delta_sec=args.rpe_delta_sec,
            tolerance_sec=args.rpe_tolerance_sec,
        )
    )

    save_metrics_csv(args.out_dir / "metrics.csv", metrics)
    (args.out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_trajectory_plot(args.out_dir / "traj_vs_truth.svg", truth_pos_a, nav_pos_i)
    save_xy_error_plot(args.out_dir / "xy_error.svg", t, err_pos)
    save_xy_error_csv(args.out_dir / "xy_error.csv", t, err_pos)

    print(f"[analyze_navresult_modern] aligned_count={len(t)}")
    print(
        "[analyze_navresult_modern] "
        f"APE(xy) rmse={metrics['ape_rmse_xy_m']:.4f} m "
        f"p95={metrics['ape_p95_xy_m']:.4f} m "
        f"max={metrics['ape_max_xy_m']:.4f} m "
        f"RPE(xy,{metrics['rpe_delta_sec']:.1f}s) rmse={metrics['rpe_rmse_xy_m']:.4f} m"
    )
    print(f"[analyze_navresult_modern] wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
