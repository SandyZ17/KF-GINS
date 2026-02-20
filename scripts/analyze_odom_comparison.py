#!/usr/bin/env python3
"""
Analyze and compare two odometry topics from a rosbag2 dataset.

What this script does:
1. Read two nav_msgs/Odometry topics from rosbag2.
2. Time-align topic A to topic B using linear interpolation.
3. Plot 3D trajectories of both topics.
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


def main():
    parser = argparse.ArgumentParser(description="Compare two odom topics and plot 3D trajectories + errors.")
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
    args = parser.parse_args()

    ref_state = read_odom_state(args.bag, args.ref_topic)
    cmp_state = read_odom_state(args.bag, args.cmp_topic)
    t, ref_i, cmp_a, ex, ey, ez, en, evx, evy, evz, evn, eroll, epitch, eyaw = align_and_error(
        ref_state, cmp_state
    )

    out_traj = f"{args.out_prefix}_traj3d.png"
    out_err = f"{args.out_prefix}_error.png"
    out_csv = f"{args.out_prefix}_error.csv"
    stride = max(1, int(args.plot_stride))

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
    fig.savefig(out_traj, dpi=180)

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
    fig2.savefig(out_err, dpi=180)

    save_error_csv(out_csv, t, ex, ey, ez, en, evx, evy, evz, evn, eroll, epitch, eyaw)

    pos_rmse = float(np.sqrt(np.mean(en * en)))
    vel_rmse = float(np.sqrt(np.mean(evn * evn)))
    att_yaw_rmse = float(np.sqrt(np.mean(eyaw * eyaw)))
    pos_p95 = float(np.percentile(en, 95))
    pos_max = float(np.max(en))
    print(f"bag={args.bag}")
    print(f"ref_topic={args.ref_topic}, count={len(ref_state['t'])}")
    print(f"cmp_topic={args.cmp_topic}, count={len(cmp_state['t'])}")
    print(f"aligned_count={len(t)}")
    print(
        f"pos_rmse={pos_rmse:.4f} m, pos_p95={pos_p95:.4f} m, pos_max={pos_max:.4f} m, "
        f"vel_rmse={vel_rmse:.4f} m/s, yaw_rmse={att_yaw_rmse:.4f} deg"
    )
    print(f"traj_png={out_traj}")
    print(f"err_png={out_err}")
    print(f"err_csv={out_csv}")


if __name__ == "__main__":
    main()
