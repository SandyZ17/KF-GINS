#!/usr/bin/env python3
"""
Compare ROS2 odometry topic against truth.nav.

This script:
1. Loads odom (position + velocity) from rosbag2.
2. Loads truth.nav (BLH + NED velocity).
3. Converts truth position to local ENU w.r.t. first truth sample.
4. Converts truth velocity from NED to ENU.
5. Aligns by relative time and computes position/velocity errors.
6. Optionally aligns truth frame to odom frame (translation / SE3).
7. Saves 3D trajectory plot, error plot and CSV.
"""

import argparse
import csv

import matplotlib
import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D2R = np.pi / 180.0
WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013


def radiusmn(lat_rad):
    s2 = np.sin(lat_rad) ** 2
    t = 1.0 - WGS84_E1 * s2
    st = np.sqrt(t)
    rm = WGS84_RA * (1.0 - WGS84_E1) / (st * t)
    rn = WGS84_RA / st
    return rm, rn


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
        v = msg.twist.twist.linear
        rows.append((t, p.x, p.y, p.z, v.x, v.y, v.z))
    if not rows:
        raise RuntimeError("No odom messages found")
    arr = np.array(rows, dtype=float)
    t = arr[:, 0]
    # Deduplicate and enforce monotonic time.
    keep = np.concatenate(([0], np.where(np.diff(t) > 0)[0] + 1))
    arr = arr[keep]
    return arr


def read_truth_to_enu(truth_nav_path):
    truth = np.loadtxt(truth_nav_path)
    # Columns: [year, sow, lat, lon, h, vn, ve, vd, ...]
    t = truth[:, 1]
    lat = truth[:, 2] * D2R
    lon = truth[:, 3] * D2R
    h = truth[:, 4]
    vn = truth[:, 5]
    ve = truth[:, 6]
    vd = truth[:, 7]

    lat0, lon0, h0 = lat[0], lon[0], h[0]
    rm, rn = radiusmn(lat0)

    north = (lat - lat0) * (rm + h0)
    east = (lon - lon0) * (rn + h0) * np.cos(lat0)
    up = h - h0

    # NED velocity -> ENU velocity
    vx = ve
    vy = vn
    vz = -vd

    # truth.nav time is usually GPS SOW. Convert to relative time so it can
    # align with rosbag messages generated with start-time-zero.
    out = np.column_stack((t - t[0], east, north, up, vx, vy, vz))
    keep = np.concatenate(([0], np.where(np.diff(out[:, 0]) > 0)[0] + 1))
    return out[keep]


def align_and_error(odom, truth):
    to = odom[:, 0]
    tt = truth[:, 0]
    t0 = max(to[0], tt[0])
    t1 = min(to[-1], tt[-1])
    if t1 <= t0:
        raise RuntimeError("No overlap in relative time between odom and truth")

    m = (to >= t0) & (to <= t1)
    to = to[m]
    od = odom[m]

    tr_i = np.zeros((len(to), 7), dtype=float)
    tr_i[:, 0] = to
    for c in range(1, 7):
        tr_i[:, c] = np.interp(to, tt, truth[:, c])

    # Position / velocity errors in ENU
    ep = od[:, 1:4] - tr_i[:, 1:4]
    ev = od[:, 4:7] - tr_i[:, 4:7]
    en_pos = np.linalg.norm(ep, axis=1)
    en_vel = np.linalg.norm(ev, axis=1)
    return to, od, tr_i, ep, ev, en_pos, en_vel


def align_and_error_nearest(odom, truth, max_time_diff):
    to = odom[:, 0]
    tt = truth[:, 0]
    t0 = max(to[0], tt[0])
    t1 = min(to[-1], tt[-1])
    if t1 <= t0:
        raise RuntimeError("No overlap in time between odom and truth")

    m = (to >= t0) & (to <= t1)
    to = to[m]
    od = odom[m]

    idx = np.searchsorted(tt, to, side="left")
    idx = np.clip(idx, 0, len(tt) - 1)
    idx_prev = np.maximum(idx - 1, 0)

    d_curr = np.abs(tt[idx] - to)
    d_prev = np.abs(tt[idx_prev] - to)
    use_prev = d_prev < d_curr
    idx[use_prev] = idx_prev[use_prev]

    dt = np.abs(tt[idx] - to)
    keep = dt <= max_time_diff
    if not np.any(keep):
        raise RuntimeError("No matched samples under max_time_diff")

    od = od[keep]
    to = to[keep]
    tr = truth[idx[keep]]
    tr[:, 0] = to

    ep = od[:, 1:4] - tr[:, 1:4]
    ev = od[:, 4:7] - tr[:, 4:7]
    en_pos = np.linalg.norm(ep, axis=1)
    en_vel = np.linalg.norm(ev, axis=1)
    return to, od, tr, ep, ev, en_pos, en_vel


def fit_translation(src_xyz, dst_xyz):
    t = np.mean(dst_xyz - src_xyz, axis=0)
    r = np.eye(3, dtype=float)
    return r, t


def fit_se3(src_xyz, dst_xyz):
    src_mean = np.mean(src_xyz, axis=0)
    dst_mean = np.mean(dst_xyz, axis=0)
    src_c = src_xyz - src_mean
    dst_c = dst_xyz - dst_mean
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_mean - r @ src_mean
    return r, t


def apply_frame_alignment(od, tr, mode):
    if mode == "none":
        return tr, np.eye(3, dtype=float), np.zeros(3, dtype=float)

    src_p = tr[:, 1:4]
    dst_p = od[:, 1:4]
    src_v = tr[:, 4:7]

    if mode == "translation":
        r, t = fit_translation(src_p, dst_p)
    elif mode == "se3":
        r, t = fit_se3(src_p, dst_p)
    else:
        raise ValueError(f"Unknown frame align mode: {mode}")

    tr_aligned = tr.copy()
    tr_aligned[:, 1:4] = (r @ src_p.T).T + t
    tr_aligned[:, 4:7] = (r @ src_v.T).T
    return tr_aligned, r, t


def save_csv(path, t, ep, ev, en_pos, en_vel):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t_rel_sec",
                "err_px",
                "err_py",
                "err_pz",
                "err_pos_norm",
                "err_vx",
                "err_vy",
                "err_vz",
                "err_vel_norm",
            ]
        )
        for i in range(len(t)):
            w.writerow(
                [
                    f"{t[i]:.6f}",
                    f"{ep[i,0]:.6f}",
                    f"{ep[i,1]:.6f}",
                    f"{ep[i,2]:.6f}",
                    f"{en_pos[i]:.6f}",
                    f"{ev[i,0]:.6f}",
                    f"{ev[i,1]:.6f}",
                    f"{ev[i,2]:.6f}",
                    f"{en_vel[i]:.6f}",
                ]
            )


def main():
    parser = argparse.ArgumentParser(description="Compare odom topic with truth.nav in ENU frame")
    parser.add_argument("--bag", required=True, help="rosbag2 directory")
    parser.add_argument("--odom-topic", default="/kf_gins/odom_pred", help="odom topic")
    parser.add_argument("--truth-nav", default="src/KF-GINS/dataset/truth.nav", help="truth.nav path")
    parser.add_argument("--out-prefix", default="", help="output prefix")
    parser.add_argument(
        "--frame-align",
        choices=["none", "translation", "se3"],
        default="none",
        help="align truth frame to odom frame before error calculation",
    )
    parser.add_argument(
        "--time-match",
        choices=["interp", "nearest"],
        default="interp",
        help="time matching method: linear interpolation or nearest truth sample",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.02,
        help="max |t_odom - t_truth| for nearest matching [s]",
    )
    parser.add_argument(
        "--align-time-origin",
        action="store_true",
        help="shift truth time axis so truth start time equals odom start time",
    )
    args = parser.parse_args()

    if not args.out_prefix:
        args.out_prefix = args.bag.rstrip("/").replace("/", "_")

    odom = read_odom_enu(args.bag, args.odom_topic)
    truth = read_truth_to_enu(args.truth_nav)
    if args.align_time_origin:
        truth = truth.copy()
        truth[:, 0] += odom[0, 0] - truth[0, 0]
    if args.time_match == "interp":
        t, od, tr, _, _, _, _ = align_and_error(odom, truth)
    else:
        t, od, tr, _, _, _, _ = align_and_error_nearest(odom, truth, args.max_time_diff)
    tr, r_align, t_align = apply_frame_alignment(od, tr, args.frame_align)
    ep = od[:, 1:4] - tr[:, 1:4]
    ev = od[:, 4:7] - tr[:, 4:7]
    en_pos = np.linalg.norm(ep, axis=1)
    en_vel = np.linalg.norm(ev, axis=1)

    out_traj = f"{args.out_prefix}_vs_truth_traj3d.png"
    out_err = f"{args.out_prefix}_vs_truth_error.png"
    out_csv = f"{args.out_prefix}_vs_truth_error.csv"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tr[:, 1], tr[:, 2], tr[:, 3], linewidth=0.9, label="truth (ENU)")
    ax.plot(od[:, 1], od[:, 2], od[:, 3], linewidth=0.9, label=f"odom: {args.odom_topic}")
    ax.set_title("3D Trajectory: Odom vs Truth")
    ax.set_xlabel("X/East [m]")
    ax.set_ylabel("Y/North [m]")
    ax.set_zlabel("Z/Up [m]")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_traj, dpi=180)

    trt = t - t[0]
    fig2, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(trt, ep[:, 0], label="pos_err_x")
    axs[0].plot(trt, ep[:, 1], label="pos_err_y")
    axs[0].plot(trt, ep[:, 2], label="pos_err_z")
    axs[0].plot(trt, en_pos, label="pos_err_norm", linewidth=1.2)
    axs[0].set_ylabel("Position Error [m]")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].plot(trt, ev[:, 0], label="vel_err_x")
    axs[1].plot(trt, ev[:, 1], label="vel_err_y")
    axs[1].plot(trt, ev[:, 2], label="vel_err_z")
    axs[1].plot(trt, en_vel, label="vel_err_norm", linewidth=1.2)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Velocity Error [m/s]")
    axs[1].grid(True)
    axs[1].legend(loc="best")
    plt.tight_layout()
    fig2.savefig(out_err, dpi=180)

    save_csv(out_csv, t, ep, ev, en_pos, en_vel)

    print(f"bag={args.bag}")
    print(f"odom_topic={args.odom_topic}")
    print(f"truth_nav={args.truth_nav}")
    print(f"time_match={args.time_match}")
    if args.time_match == "nearest":
        print(f"max_time_diff={args.max_time_diff}")
    print(f"align_time_origin={args.align_time_origin}")
    print(f"frame_align={args.frame_align}")
    print(f"R_truth_to_odom=\n{r_align}")
    print(f"t_truth_to_odom={t_align}")
    print(f"aligned_count={len(t)}")
    print(
        "pos_rmse={:.4f} m, pos_p95={:.4f} m, pos_max={:.4f} m, vel_rmse={:.4f} m/s".format(
            float(np.sqrt(np.mean(en_pos**2))),
            float(np.percentile(en_pos, 95)),
            float(np.max(en_pos)),
            float(np.sqrt(np.mean(en_vel**2))),
        )
    )
    print(f"traj_png={out_traj}")
    print(f"err_png={out_err}")
    print(f"err_csv={out_csv}")


if __name__ == "__main__":
    main()
