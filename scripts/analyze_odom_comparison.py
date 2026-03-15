#!/usr/bin/env python3
"""
Analyze and compare two odometry topics from a rosbag2 dataset.

What this script does:
1. Read two nav_msgs/Odometry topics from rosbag2.
2. Time-align topic A to topic B using linear interpolation.
3. Plot 2D/3D trajectories of both topics.
4. Compute error categories similar to plot_navresult.py:
   - position error (x/y/z and norm)
   - velocity error (vx/vy/vz and norm)
   - attitude error (roll/pitch/yaw in deg)
5. Plot error curves and print summary metrics.
5. Save aligned error samples to CSV.

Default topics are:
- /kf_gins/odom_pred   (predicted/interpolated odometry)
- /kf_gins/odom_fused  (fused odometry)

Example:
python3 src/KF-GINS/scripts/analyze_odom_comparison.py \
  --bag src/KF-GINS/dataset/test_dataset/rosbag2_2026_02_18-13_05_41
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
from matplotlib.ticker import MultipleLocator  # noqa: E402


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q))


def quat_to_rpy_deg(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def read_odom_state(bag_dir, topic):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise RuntimeError(f"Topic not found in bag: {topic}")

    msg_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy_deg(q.x, q.y, q.z, q.w)
        rows.append((stamp, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, roll, pitch, yaw))

    if not rows:
        raise RuntimeError(f"No messages found in topic: {topic}")

    arr = np.array(rows, dtype=float)
    return {
        "t": arr[:, 0],
        "x": arr[:, 1],
        "y": arr[:, 2],
        "z": arr[:, 3],
        "vx": arr[:, 4],
        "vy": arr[:, 5],
        "vz": arr[:, 6],
        "roll": arr[:, 7],
        "pitch": arr[:, 8],
        "yaw": arr[:, 9],
    }


def dedup_monotonic(state):
    # Keep strictly increasing timestamps for interpolation stability.
    t = state["t"]
    idx = np.where(np.diff(t) > 0)[0]
    keep = np.concatenate(([0], idx + 1))
    return {k: v[keep] for k, v in state.items()}


def wrap_deg(diff):
    return (diff + 180.0) % 360.0 - 180.0


def interp_angle_deg(t_src, angle_deg, t_dst):
    # Unwrap before interpolation to avoid 180-deg discontinuity artifacts.
    ang_unwrap = np.unwrap(np.deg2rad(angle_deg))
    ang_i = np.interp(t_dst, t_src, ang_unwrap)
    return np.rad2deg(ang_i)


def align_and_error(ref_state, cmp_state):
    ref_state = dedup_monotonic(ref_state)
    cmp_state = dedup_monotonic(cmp_state)

    t_ref = ref_state["t"]
    t_cmp = cmp_state["t"]
    t0 = max(t_ref[0], t_cmp[0])
    t1 = min(t_ref[-1], t_cmp[-1])
    if t1 <= t0:
        raise RuntimeError("No overlapping time range between the two odom topics.")

    m = (t_cmp >= t0) & (t_cmp <= t1)
    tc = t_cmp[m]

    ref_i = {
        "x": np.interp(tc, t_ref, ref_state["x"]),
        "y": np.interp(tc, t_ref, ref_state["y"]),
        "z": np.interp(tc, t_ref, ref_state["z"]),
        "vx": np.interp(tc, t_ref, ref_state["vx"]),
        "vy": np.interp(tc, t_ref, ref_state["vy"]),
        "vz": np.interp(tc, t_ref, ref_state["vz"]),
        "roll": interp_angle_deg(t_ref, ref_state["roll"], tc),
        "pitch": interp_angle_deg(t_ref, ref_state["pitch"], tc),
        "yaw": interp_angle_deg(t_ref, ref_state["yaw"], tc),
    }

    cmp_a = {k: v[m] for k, v in cmp_state.items() if k != "t"}

    ex = cmp_a["x"] - ref_i["x"]
    ey = cmp_a["y"] - ref_i["y"]
    ez = cmp_a["z"] - ref_i["z"]
    en = np.sqrt(ex * ex + ey * ey + ez * ez)

    evx = cmp_a["vx"] - ref_i["vx"]
    evy = cmp_a["vy"] - ref_i["vy"]
    evz = cmp_a["vz"] - ref_i["vz"]
    evn = np.sqrt(evx * evx + evy * evy + evz * evz)

    eroll = wrap_deg(cmp_a["roll"] - ref_i["roll"])
    epitch = wrap_deg(cmp_a["pitch"] - ref_i["pitch"])
    eyaw = wrap_deg(cmp_a["yaw"] - ref_i["yaw"])

    return tc, ref_i, cmp_a, ex, ey, ez, en, evx, evy, evz, evn, eroll, epitch, eyaw


def save_error_csv(path, t, ex, ey, ez, en, evx, evy, evz, evn, eroll, epitch, eyaw):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stamp_sec",
                "err_x",
                "err_y",
                "err_z",
                "err_pos_norm",
                "err_vx",
                "err_vy",
                "err_vz",
                "err_vel_norm",
                "err_roll_deg",
                "err_pitch_deg",
                "err_yaw_deg",
            ]
        )
        for i in range(len(t)):
            w.writerow(
                [
                    f"{t[i]:.9f}",
                    f"{ex[i]:.6f}",
                    f"{ey[i]:.6f}",
                    f"{ez[i]:.6f}",
                    f"{en[i]:.6f}",
                    f"{evx[i]:.6f}",
                    f"{evy[i]:.6f}",
                    f"{evz[i]:.6f}",
                    f"{evn[i]:.6f}",
                    f"{eroll[i]:.6f}",
                    f"{epitch[i]:.6f}",
                    f"{eyaw[i]:.6f}",
                ]
            )


def find_nearest_index(times, target):
    idx = np.searchsorted(times, target, side="left")
    idx = int(np.clip(idx, 0, len(times) - 1))
    if idx == 0:
        return 0, abs(times[0] - target)
    prev_idx = idx - 1
    if abs(times[idx] - target) < abs(times[prev_idx] - target):
        return idx, abs(times[idx] - target)
    return prev_idx, abs(times[prev_idx] - target)


def compute_rpe_metrics(t, ref_xyz, cmp_xyz, delta_sec, tolerance_sec):
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
        cmp_rel = cmp_xyz[j] - cmp_xyz[i]
        err = cmp_rel - ref_rel
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


def save_metrics_csv(path, metrics):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def main():
    parser = argparse.ArgumentParser(description="Compare two odom topics and plot 2D/3D trajectories + errors.")
    parser.add_argument(
        "--bag",
        default="src/KF-GINS/dataset/test_dataset/rosbag2_2026_02_18-13_05_41",
        help="rosbag2 directory",
    )
    parser.add_argument("--ref-topic", default="/kf_gins/odom_fused", help="reference odom topic")
    parser.add_argument("--cmp-topic", default="/kf_gins/odom_pred", help="compared odom topic")
    parser.add_argument(
        "--out-prefix",
        default="src/KF-GINS/dataset/test_dataset/rosbag2_2026_02_18-13_05_41_odom_compare",
        help="output path prefix",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=10,
        help="plot every N-th aligned sample (metrics still use all samples)",
    )
    parser.add_argument(
        "--x-tick-seconds",
        type=float,
        default=300.0,
        help="x-axis major tick spacing in seconds for error plots",
    )
    parser.add_argument("--rpe-delta-sec", type=float, default=1.0, help="RPE delta interval [s]")
    parser.add_argument("--rpe-tolerance-sec", type=float, default=0.15, help="RPE time matching tolerance [s]")
    args = parser.parse_args()

    ref_state = read_odom_state(args.bag, args.ref_topic)
    cmp_state = read_odom_state(args.bag, args.cmp_topic)
    t, ref_i, cmp_a, ex, ey, ez, en, evx, evy, evz, evn, eroll, epitch, eyaw = align_and_error(
        ref_state, cmp_state
    )

    out_traj_2d = f"{args.out_prefix}_traj2d.svg"
    out_traj_3d = f"{args.out_prefix}_traj3d.svg"
    out_err = f"{args.out_prefix}_error.svg"
    out_csv = f"{args.out_prefix}_error.csv"
    out_metrics = f"{args.out_prefix}_metrics.csv"
    stride = max(1, int(args.plot_stride))

    fig2d = plt.figure(figsize=(8, 8))
    ax2d = fig2d.add_subplot(111)
    ax2d.plot(ref_i["x"][::stride], ref_i["y"][::stride], linewidth=0.9, label=f"ref: {args.ref_topic}")
    ax2d.plot(cmp_a["x"][::stride], cmp_a["y"][::stride], linewidth=0.9, label=f"cmp: {args.cmp_topic}")
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.set_title("2D Trajectory Comparison (XY)")
    ax2d.grid(True)
    ax2d.axis("equal")
    ax2d.legend(loc="best")
    plt.tight_layout()
    fig2d.savefig(out_traj_2d, dpi=600)
    plt.close(fig2d)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref_i["x"][::stride], ref_i["y"][::stride], ref_i["z"][::stride], linewidth=0.8, label=f"ref: {args.ref_topic}")
    ax.plot(cmp_a["x"][::stride], cmp_a["y"][::stride], cmp_a["z"][::stride], linewidth=0.8, label=f"cmp: {args.cmp_topic}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D Trajectory Comparison")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_traj_3d, dpi=600)
    plt.close(fig)

    t_rel = t - t[0]
    fig2, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axs[0].plot(t_rel[::stride], ex[::stride], label="pos_err_x")
    axs[0].plot(t_rel[::stride], ey[::stride], label="pos_err_y")
    axs[0].plot(t_rel[::stride], ez[::stride], label="pos_err_z")
    axs[0].plot(t_rel[::stride], en[::stride], label="pos_err_norm", linewidth=1.3)
    axs[0].set_ylabel("Position Error [m]")
    axs[0].grid(True)
    axs[0].legend(loc="best")

    axs[1].plot(t_rel[::stride], evx[::stride], label="vel_err_x")
    axs[1].plot(t_rel[::stride], evy[::stride], label="vel_err_y")
    axs[1].plot(t_rel[::stride], evz[::stride], label="vel_err_z")
    axs[1].plot(t_rel[::stride], evn[::stride], label="vel_err_norm", linewidth=1.3)
    axs[1].set_ylabel("Velocity Error [m/s]")
    axs[1].grid(True)
    axs[1].legend(loc="best")

    axs[2].plot(t_rel[::stride], eroll[::stride], label="att_err_roll")
    axs[2].plot(t_rel[::stride], epitch[::stride], label="att_err_pitch")
    axs[2].plot(t_rel[::stride], eyaw[::stride], label="att_err_yaw")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Attitude Error [deg]")
    axs[2].grid(True)
    axs[2].legend(loc="best")
    tick_step = max(1.0, float(args.x_tick_seconds))
    axs[2].xaxis.set_major_locator(MultipleLocator(tick_step))
    plt.tight_layout()
    fig2.savefig(out_err, dpi=600)

    save_error_csv(out_csv, t, ex, ey, ez, en, evx, evy, evz, evn, eroll, epitch, eyaw)
    exy = np.sqrt(ex * ex + ey * ey)
    abs_ez = np.abs(ez)
    ref_xyz = np.column_stack((ref_i["x"], ref_i["y"], ref_i["z"]))
    cmp_xyz = np.column_stack((cmp_a["x"], cmp_a["y"], cmp_a["z"]))
    ape_metrics = {
        "aligned_count": int(len(t)),
        "ape_rmse_xy_m": rmse(exy),
        "ape_p95_xy_m": percentile(exy, 95),
        "ape_max_xy_m": float(np.max(exy)),
        "ape_rmse_z_m": rmse(abs_ez),
        "ape_p95_abs_z_m": percentile(abs_ez, 95),
        "ape_max_abs_z_m": float(np.max(abs_ez)),
        "ape_rmse_3d_m": rmse(en),
        "ape_p95_3d_m": percentile(en, 95),
        "ape_max_3d_m": float(np.max(en)),
        "vel_rmse_mps": rmse(evn),
        "yaw_rmse_deg": rmse(eyaw),
    }
    rpe_metrics = compute_rpe_metrics(t, ref_xyz, cmp_xyz, args.rpe_delta_sec, args.rpe_tolerance_sec)
    metrics = {**ape_metrics, **rpe_metrics}
    save_metrics_csv(out_metrics, metrics)

    print(f"bag={args.bag}")
    print(f"ref_topic={args.ref_topic}, count={len(ref_state['t'])}")
    print(f"cmp_topic={args.cmp_topic}, count={len(cmp_state['t'])}")
    print(f"aligned_count={len(t)}")
    print(
        f"APE(xy) rmse={metrics['ape_rmse_xy_m']:.4f} m, p95={metrics['ape_p95_xy_m']:.4f} m, "
        f"max={metrics['ape_max_xy_m']:.4f} m, "
        f"RPE(xy,{metrics['rpe_delta_sec']:.1f}s) rmse={metrics['rpe_rmse_xy_m']:.4f} m, "
        f"APE(3d) rmse={metrics['ape_rmse_3d_m']:.4f} m"
    )
    print(f"traj2d_svg={out_traj_2d}")
    print(f"traj3d_svg={out_traj_3d}")
    print(f"err_svg={out_err}")
    print(f"err_csv={out_csv}")
    print(f"metrics_csv={out_metrics}")


if __name__ == "__main__":
    main()
