#!/usr/bin/env python3
"""
Compare ROS2 odometry topic against truth (truth.nav / UrbanNav GT / TUM).

This script:
1. Loads odom (position + velocity) from rosbag2.
2. Loads truth file (truth.nav / UrbanNav GT / TUM trajectory).
3. Converts truth position to local ENU or local metric frame.
4. Converts / estimates truth velocity if available.
5. Aligns by relative time and computes position/velocity errors.
6. Optionally aligns truth frame to odom frame (translation / SE3).
7. Saves 2D/3D trajectory plots, error plot and CSV.
"""

import argparse
import csv
from pathlib import Path

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


def read_tum_to_local(tum_path):
    # TUM format: timestamp tx ty tz qx qy qz qw
    arr = np.loadtxt(tum_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 8:
        raise RuntimeError(f"TUM file requires >=8 columns, got {arr.shape[1]}")
    t = arr[:, 0]
    xyz = arr[:, 1:4]
    # Normalize to local origin for metric trajectory comparison.
    xyz = xyz - xyz[0]

    # Finite-difference velocity in the same local frame.
    vel = np.zeros_like(xyz)
    if len(t) >= 2:
        dt = np.diff(t)
        dt_safe = np.where(dt > 1e-6, dt, np.nan)
        dxyz = np.diff(xyz, axis=0)
        v_mid = dxyz / dt_safe[:, None]
        vel[1:] = np.where(np.isfinite(v_mid), v_mid, 0.0)
        vel[0] = vel[1]

    out = np.column_stack((t, xyz[:, 0], xyz[:, 1], xyz[:, 2], vel[:, 0], vel[:, 1], vel[:, 2]))
    keep = np.concatenate(([0], np.where(np.diff(out[:, 0]) > 0)[0] + 1))
    return out[keep]


def dms_to_deg(sign_deg: float, minutes: float, seconds: float) -> float:
    sign = -1.0 if sign_deg < 0 else 1.0
    deg_abs = abs(sign_deg) + minutes / 60.0 + seconds / 3600.0
    return sign * deg_abs


def euler_zyx_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def read_urbannav_gt_to_enu(gt_path):
    rows = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("UTCTime") or s.startswith("(sec)"):
                continue
            parts = s.split()
            # Expected columns:
            # UTCTime Week GPSTime lat_d lat_m lat_s lon_d lon_m lon_s h vbx vby vbz abx aby abz roll pitch heading Q
            if len(parts) < 20:
                continue
            try:
                utc_time = float(parts[0])
                gps_time = float(parts[2])
                lat_deg = dms_to_deg(float(parts[3]), float(parts[4]), float(parts[5]))
                lon_deg = dms_to_deg(float(parts[6]), float(parts[7]), float(parts[8]))
                h = float(parts[9])
                vbx = float(parts[10])
                vby = float(parts[11])
                vbz = float(parts[12])
                roll = float(parts[16])
                pitch = float(parts[17])
                heading = float(parts[18])  # azimuth clockwise from North
                yaw_enu = 90.0 - heading  # ENU yaw is CCW from East
                c_b_e = euler_zyx_to_rot(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw_enu))
                v_enu = c_b_e @ np.array([vbx, vby, vbz], dtype=float)
                # Use UTC time (matches many ROS bag header stamps in UrbanNav playback pipelines).
                # GPS SOW is still parsed for debugging/extension if needed.
                _ = gps_time
                rows.append((utc_time, lat_deg, lon_deg, h, v_enu[0], v_enu[1], v_enu[2]))
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No valid UrbanNav GT rows parsed from {gt_path}")
    arr = np.array(rows, dtype=float)
    t = arr[:, 0]
    lat = arr[:, 1] * D2R
    lon = arr[:, 2] * D2R
    h = arr[:, 3]
    vx = arr[:, 4]
    vy = arr[:, 5]
    vz = arr[:, 6]

    lat0, lon0, h0 = lat[0], lon[0], h[0]
    rm, rn = radiusmn(lat0)
    north = (lat - lat0) * (rm + h0)
    east = (lon - lon0) * (rn + h0) * np.cos(lat0)
    up = h - h0
    out = np.column_stack((t, east, north, up, vx, vy, vz))
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
    parser.add_argument("--truth-nav", default="src/KF-GINS/dataset/truth.nav", help="truth file path")
    parser.add_argument(
        "--truth-format",
        choices=["auto", "truth_nav", "urbannav_gt_raw", "tum"],
        default="auto",
        help="truth file format",
    )
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
    parser.add_argument(
        "--segment-start-sec",
        type=float,
        default=None,
        help="keep only samples with (t - t0) >= this value [s]",
    )
    parser.add_argument(
        "--segment-end-sec",
        type=float,
        default=None,
        help="keep only samples with (t - t0) <= this value [s]",
    )
    args = parser.parse_args()

    if not args.out_prefix:
        args.out_prefix = args.bag.rstrip("/").replace("/", "_")

    odom = read_odom_enu(args.bag, args.odom_topic)
    truth_format = args.truth_format
    if truth_format == "auto":
        name = Path(args.truth_nav).name.lower()
        if "urban" in name and name.endswith(".txt"):
            truth_format = "urbannav_gt_raw"
        elif name.endswith(".txt"):
            # Many M2DGR/M3DGR ground-truth trajectories are TUM-format txt.
            try:
                first = np.loadtxt(args.truth_nav, max_rows=1)
                if np.size(first) >= 8:
                    truth_format = "tum"
                else:
                    truth_format = "truth_nav"
            except Exception:
                truth_format = "truth_nav"
        else:
            truth_format = "truth_nav"
    if truth_format == "truth_nav":
        truth = read_truth_to_enu(args.truth_nav)
    elif truth_format == "tum":
        truth = read_tum_to_local(args.truth_nav)
    else:
        truth = read_urbannav_gt_to_enu(args.truth_nav)
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

    # Optional segment filtering (e.g., tunnel-only evaluation) in analysis-relative time.
    trt = t - t[0]
    keep = np.ones_like(trt, dtype=bool)
    if args.segment_start_sec is not None:
        keep &= trt >= args.segment_start_sec
    if args.segment_end_sec is not None:
        keep &= trt <= args.segment_end_sec
    if not np.any(keep):
        raise RuntimeError("No matched samples remain after segment filtering.")
    if not np.all(keep):
        t = t[keep]
        od = od[keep]
        tr = tr[keep]
        ep = ep[keep]
        ev = ev[keep]
        en_pos = en_pos[keep]
        en_vel = en_vel[keep]
    trt = t - t[0]

    out_traj2d = f"{args.out_prefix}_vs_truth_traj2d.png"
    out_traj2d_err = f"{args.out_prefix}_vs_truth_traj2d_error_heatmap.png"
    out_traj = f"{args.out_prefix}_vs_truth_traj3d.png"
    out_err = f"{args.out_prefix}_vs_truth_error.png"
    out_csv = f"{args.out_prefix}_vs_truth_error.csv"

    fig0 = plt.figure(figsize=(8, 8))
    ax0 = fig0.add_subplot(111)
    ax0.plot(tr[:, 1], tr[:, 2], linewidth=1.0, label="truth (ENU)")
    ax0.plot(od[:, 1], od[:, 2], linewidth=1.0, label=f"odom: {args.odom_topic}")
    # Mark start/end points to make trajectory alignment and segment coverage explicit.
    ax0.scatter([tr[0, 1]], [tr[0, 2]], c="green", s=40, marker="o", label="truth start", zorder=3)
    ax0.scatter([tr[-1, 1]], [tr[-1, 2]], c="green", s=50, marker="x", label="truth end", zorder=3)
    ax0.scatter([od[0, 1]], [od[0, 2]], c="red", s=40, marker="o", label="odom start", zorder=3)
    ax0.scatter([od[-1, 1]], [od[-1, 2]], c="red", s=50, marker="x", label="odom end", zorder=3)
    ax0.annotate("T-start", (tr[0, 1], tr[0, 2]), xytext=(5, 5), textcoords="offset points", color="green")
    ax0.annotate("T-end", (tr[-1, 1], tr[-1, 2]), xytext=(5, 5), textcoords="offset points", color="green")
    ax0.annotate("O-start", (od[0, 1], od[0, 2]), xytext=(5, -12), textcoords="offset points", color="red")
    ax0.annotate("O-end", (od[-1, 1], od[-1, 2]), xytext=(5, -12), textcoords="offset points", color="red")
    ax0.set_title("2D Trajectory: Odom vs Truth")
    ax0.set_xlabel("X/East [m]")
    ax0.set_ylabel("Y/North [m]")
    ax0.grid(True)
    ax0.axis("equal")
    ax0.legend(loc="best")
    plt.tight_layout()
    fig0.savefig(out_traj2d, dpi=180)

    traj3d_ok = True
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(tr[:, 1], tr[:, 2], tr[:, 3], linewidth=0.9, label="truth")
        ax.plot(od[:, 1], od[:, 2], od[:, 3], linewidth=0.9, label=f"odom: {args.odom_topic}")
        ax.scatter([tr[0, 1]], [tr[0, 2]], [tr[0, 3]], c="green", s=30, marker="o")
        ax.scatter([tr[-1, 1]], [tr[-1, 2]], [tr[-1, 3]], c="green", s=40, marker="x")
        ax.scatter([od[0, 1]], [od[0, 2]], [od[0, 3]], c="red", s=30, marker="o")
        ax.scatter([od[-1, 1]], [od[-1, 2]], [od[-1, 3]], c="red", s=40, marker="x")
        ax.set_title("3D Trajectory: Odom vs Truth")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.grid(True)
        ax.legend(loc="best")
        plt.tight_layout()
        fig.savefig(out_traj, dpi=180)
    except Exception as e:
        traj3d_ok = False
        print(f"[WARN] 3D trajectory plot skipped: {e}")

    figh = plt.figure(figsize=(8, 8))
    axh = figh.add_subplot(111)
    sc = axh.scatter(od[:, 1], od[:, 2], c=en_pos, s=6, cmap="turbo")
    axh.plot(tr[:, 1], tr[:, 2], linewidth=0.8, color="black", alpha=0.5, label="truth")
    cb = figh.colorbar(sc, ax=axh)
    cb.set_label("Position Error Norm [m]")
    axh.set_title("2D Trajectory Error Heatmap (Odom vs Truth)")
    axh.set_xlabel("X/East [m]")
    axh.set_ylabel("Y/North [m]")
    axh.grid(True)
    axh.axis("equal")
    axh.legend(loc="best")
    plt.tight_layout()
    figh.savefig(out_traj2d_err, dpi=180)

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
    print(f"truth_format={truth_format}")
    if truth_format == "tum" and args.frame_align == "none":
        print(
            "[WARN] TUM truth is usually in a local trajectory frame. "
            "If origins/axes differ from odom ENU, use --frame-align translation or --frame-align se3."
        )
    print(f"time_match={args.time_match}")
    if args.time_match == "nearest":
        print(f"max_time_diff={args.max_time_diff}")
    print(f"align_time_origin={args.align_time_origin}")
    print(f"frame_align={args.frame_align}")
    print(f"segment_start_sec={args.segment_start_sec}")
    print(f"segment_end_sec={args.segment_end_sec}")
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
    print(f"traj_png={out_traj if traj3d_ok else 'SKIPPED'}")
    print(f"traj2d_png={out_traj2d}")
    print(f"traj2d_err_png={out_traj2d_err}")
    print(f"err_png={out_err}")
    print(f"err_csv={out_csv}")


if __name__ == "__main__":
    main()
