#!/usr/bin/env python3
"""
Compare ROS2 odometry topic against raw GNSS NavSatFix using APE/RPE metrics.
"""

import argparse
import csv
import math

import matplotlib
import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


D2R = math.pi / 180.0
WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013


def radiusmn(lat_rad):
    s2 = math.sin(lat_rad) ** 2
    t = 1.0 - WGS84_E1 * s2
    st = math.sqrt(t)
    rm = WGS84_RA * (1.0 - WGS84_E1) / (st * t)
    rn = WGS84_RA / st
    return rm, rn


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q))


def read_odom_enu(bag_dir, topic):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise RuntimeError(f"Topic not found: {topic}")
    msg_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        rows.append((t, p.x, p.y, p.z))
    if not rows:
        raise RuntimeError(f"No odom messages found on {topic}")
    arr = np.array(rows, dtype=float)
    keep = np.concatenate(([0], np.where(np.diff(arr[:, 0]) > 0)[0] + 1))
    return arr[keep]


def read_gps_to_enu(bag_dir, topic):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise RuntimeError(f"Topic not found: {topic}")
    msg_type = get_message(type_map[topic])
    llh_rows = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        llh_rows.append((t, msg.latitude * D2R, msg.longitude * D2R, msg.altitude))
    if not llh_rows:
        raise RuntimeError(f"No GPS messages found on {topic}")
    arr = np.array(llh_rows, dtype=float)
    lat0 = arr[0, 1]
    lon0 = arr[0, 2]
    h0 = arr[0, 3]
    rm, rn = radiusmn(lat0)
    north = (arr[:, 1] - lat0) * (rm + h0)
    east = (arr[:, 2] - lon0) * (rn + h0) * math.cos(lat0)
    up = arr[:, 3] - h0
    out = np.column_stack((arr[:, 0], east, north, up))
    keep = np.concatenate(([0], np.where(np.diff(out[:, 0]) > 0)[0] + 1))
    return out[keep]


def align_interp(odom, gps):
    to = odom[:, 0]
    tg = gps[:, 0]
    t0 = max(to[0], tg[0])
    t1 = min(to[-1], tg[-1])
    if t1 <= t0:
        raise RuntimeError("No overlapping time range between odom and gps")
    keep = (to >= t0) & (to <= t1)
    od = odom[keep]
    to = od[:, 0]
    gp = np.zeros((len(to), 4), dtype=float)
    gp[:, 0] = to
    for c in range(1, 4):
        gp[:, c] = np.interp(to, tg, gps[:, c])
    return to, od, gp, np.zeros(len(to), dtype=float)


def align_nearest(odom, gps, max_time_diff):
    to = odom[:, 0]
    tg = gps[:, 0]
    t0 = max(to[0], tg[0])
    t1 = min(to[-1], tg[-1])
    if t1 <= t0:
        raise RuntimeError("No overlapping time range between odom and gps")
    keep = (to >= t0) & (to <= t1)
    od = odom[keep]
    to = od[:, 0]
    idx = np.searchsorted(tg, to, side="left")
    idx = np.clip(idx, 0, len(tg) - 1)
    idx_prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(tg[idx_prev] - to) < np.abs(tg[idx] - to)
    idx[use_prev] = idx_prev[use_prev]
    dt = np.abs(tg[idx] - to)
    keep = dt <= max_time_diff
    if not np.any(keep):
        raise RuntimeError("No matched samples under max_time_diff")
    od = od[keep]
    to = to[keep]
    gp = gps[idx[keep]].copy()
    gp[:, 0] = to
    return to, od, gp, dt[keep]


def find_nearest_index(times, target):
    idx = np.searchsorted(times, target, side="left")
    idx = int(np.clip(idx, 0, len(times) - 1))
    if idx == 0:
        return 0, abs(times[0] - target)
    prev_idx = idx - 1
    if abs(times[idx] - target) < abs(times[prev_idx] - target):
        return idx, abs(times[idx] - target)
    return prev_idx, abs(times[prev_idx] - target)


def compute_rpe_metrics(t, gps_xyz, odom_xyz, delta_sec, tolerance_sec):
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
        gps_rel = gps_xyz[j] - gps_xyz[i]
        odom_rel = odom_xyz[j] - odom_xyz[i]
        err = odom_rel - gps_rel
        xy_errors.append(float(np.linalg.norm(err[:2])))
        z_errors.append(abs(float(err[2])))
        d3_errors.append(float(np.linalg.norm(err)))
    if not xy_errors:
        return {
            "rpe_delta_sec": float(delta_sec),
            "rpe_pair_count": 0,
            "rpe_rmse_xy_m": float("inf"),
            "rpe_p95_xy_m": float("inf"),
            "rpe_max_xy_m": float("inf"),
            "rpe_rmse_z_m": float("inf"),
            "rpe_p95_abs_z_m": float("inf"),
            "rpe_max_abs_z_m": float("inf"),
            "rpe_rmse_3d_m": float("inf"),
            "rpe_p95_3d_m": float("inf"),
            "rpe_max_3d_m": float("inf"),
        }
    return {
        "rpe_delta_sec": float(delta_sec),
        "rpe_pair_count": len(xy_errors),
        "rpe_rmse_xy_m": rmse(xy_errors),
        "rpe_p95_xy_m": percentile(xy_errors, 95),
        "rpe_max_xy_m": float(np.max(xy_errors)),
        "rpe_rmse_z_m": rmse(z_errors),
        "rpe_p95_abs_z_m": percentile(z_errors, 95),
        "rpe_max_abs_z_m": float(np.max(z_errors)),
        "rpe_rmse_3d_m": rmse(d3_errors),
        "rpe_p95_3d_m": percentile(d3_errors, 95),
        "rpe_max_3d_m": float(np.max(d3_errors)),
    }


def save_error_csv(path, t, odom, gps, ape_xy, ape_3d):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "odom_x", "odom_y", "odom_z", "gps_x", "gps_y", "gps_z", "ape_xy_m", "ape_3d_m"])
        for i in range(len(t)):
            w.writerow(
                [
                    f"{t[i]:.9f}",
                    f"{odom[i,1]:.6f}",
                    f"{odom[i,2]:.6f}",
                    f"{odom[i,3]:.6f}",
                    f"{gps[i,1]:.6f}",
                    f"{gps[i,2]:.6f}",
                    f"{gps[i,3]:.6f}",
                    f"{ape_xy[i]:.6f}",
                    f"{ape_3d[i]:.6f}",
                ]
            )


def save_metrics_csv(path, metrics):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def main():
    ap = argparse.ArgumentParser(description="Compare odom topic against raw GPS NavSatFix using APE/RPE")
    ap.add_argument("--bag", required=True)
    ap.add_argument("--odom-topic", default="/kf_gins/odom_pred")
    ap.add_argument("--gps-topic", default="/gps/fix")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--time-match", choices=["interp", "nearest"], default="nearest")
    ap.add_argument("--max-time-diff", type=float, default=0.2)
    ap.add_argument("--rpe-delta-sec", type=float, default=1.0)
    ap.add_argument("--rpe-tolerance-sec", type=float, default=0.15)
    args = ap.parse_args()

    odom = read_odom_enu(args.bag, args.odom_topic)
    gps = read_gps_to_enu(args.bag, args.gps_topic)
    if args.time_match == "interp":
        t, od, gp, dt = align_interp(odom, gps)
    else:
        t, od, gp, dt = align_nearest(odom, gps, args.max_time_diff)

    err = od[:, 1:4] - gp[:, 1:4]
    ape_xy = np.linalg.norm(err[:, :2], axis=1)
    ape_z = np.abs(err[:, 2])
    ape_3d = np.linalg.norm(err, axis=1)
    rpe = compute_rpe_metrics(t, gp[:, 1:4], od[:, 1:4], args.rpe_delta_sec, args.rpe_tolerance_sec)

    metrics = {
        "matched_count": int(len(t)),
        "ape_rmse_xy_m": rmse(ape_xy),
        "ape_p95_xy_m": percentile(ape_xy, 95),
        "ape_max_xy_m": float(np.max(ape_xy)),
        "ape_rmse_z_m": rmse(ape_z),
        "ape_p95_abs_z_m": percentile(ape_z, 95),
        "ape_max_abs_z_m": float(np.max(ape_z)),
        "ape_rmse_3d_m": rmse(ape_3d),
        "ape_p95_3d_m": percentile(ape_3d, 95),
        "ape_max_3d_m": float(np.max(ape_3d)),
        "ape_max_match_dt_s": float(np.max(dt)) if len(dt) else 0.0,
        "time_match": args.time_match,
        "max_time_diff": args.max_time_diff,
    }
    metrics.update(rpe)

    out_traj = f"{args.out_prefix}_traj2d.svg"
    out_err = f"{args.out_prefix}_error.svg"
    out_csv = f"{args.out_prefix}_error.csv"
    out_metrics = f"{args.out_prefix}_metrics.csv"

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(gp[:, 1], gp[:, 2], linewidth=1.0, label=f"gps: {args.gps_topic}")
    ax.plot(od[:, 1], od[:, 2], linewidth=1.0, label=f"odom: {args.odom_topic}")
    ax.set_title("2D Trajectory: Odom vs Raw GPS")
    ax.set_xlabel("X/East [m]")
    ax.set_ylabel("Y/North [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_traj, dpi=600)

    t_rel = t - t[0]
    fig2, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(t_rel, ape_xy, label="APE_xy")
    axs[0].plot(t_rel, ape_3d, label="APE_3d")
    axs[0].set_ylabel("APE [m]")
    axs[0].grid(True)
    axs[0].legend(loc="best")
    axs[1].plot(t_rel, err[:, 0], label="err_x")
    axs[1].plot(t_rel, err[:, 1], label="err_y")
    axs[1].plot(t_rel, err[:, 2], label="err_z")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Position Error [m]")
    axs[1].grid(True)
    axs[1].legend(loc="best")
    plt.tight_layout()
    fig2.savefig(out_err, dpi=600)

    save_error_csv(out_csv, t, od, gp, ape_xy, ape_3d)
    save_metrics_csv(out_metrics, metrics)

    print(f"bag={args.bag}")
    print(f"odom_topic={args.odom_topic}")
    print(f"gps_topic={args.gps_topic}")
    print(
        "APE(xy) rmse={:.4f} m, p95={:.4f} m, max={:.4f} m, RPE(xy,{:.1f}s) rmse={:.4f} m, APE(3d) rmse={:.4f} m".format(
            metrics["ape_rmse_xy_m"],
            metrics["ape_p95_xy_m"],
            metrics["ape_max_xy_m"],
            metrics["rpe_delta_sec"],
            metrics["rpe_rmse_xy_m"],
            metrics["ape_rmse_3d_m"],
        )
    )
    print(f"traj2d_svg={out_traj}")
    print(f"err_svg={out_err}")
    print(f"err_csv={out_csv}")
    print(f"metrics_csv={out_metrics}")


if __name__ == "__main__":
    main()
