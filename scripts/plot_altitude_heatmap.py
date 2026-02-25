#!/usr/bin/env python3
"""
Plot altitude heatmap from pose/odometry in a ROS2 bag.

Input topic should be nav_msgs/msg/Odometry (e.g. /kf_gins/odom_pred).
Output is a 2D XY plot colored by Z (altitude / local up).
"""

import argparse

import matplotlib
import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402


def read_odom_xyz(bag_dir: str, topic: str) -> np.ndarray:
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
        p = msg.pose.pose.position
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        rows.append((t, p.x, p.y, p.z))

    if not rows:
        raise RuntimeError(f"No odometry messages found on {topic}")

    arr = np.array(rows, dtype=float)
    # remove non-increasing timestamps
    keep = np.concatenate(([0], np.where(np.diff(arr[:, 0]) > 0)[0] + 1))
    return arr[keep]


def plot_heatmap(
    data: np.ndarray,
    out_png: str,
    title: str,
    stride: int,
    mode: str,
    gridsize: int,
    overlay_track: bool,
) -> None:
    x = data[::stride, 1]
    y = data[::stride, 2]
    z = data[::stride, 3]

    plt.figure(figsize=(10, 8))
    if mode == "hexbin":
        hb = plt.hexbin(x, y, C=z, gridsize=gridsize, reduce_C_function=np.mean, cmap="turbo")
        cb = plt.colorbar(hb)
    else:
        sc = plt.scatter(x, y, c=z, s=4, cmap="turbo")
        cb = plt.colorbar(sc)

    if overlay_track:
        plt.plot(data[::stride, 1], data[::stride, 2], color="black", linewidth=0.4, alpha=0.35, label="track")
        plt.legend(loc="best")

    cb.set_label("Altitude Z [m]")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)


def plot_surface3d(
    data: np.ndarray,
    out_png: str,
    title: str,
    stride: int,
    elev_deg: float,
    azim_deg: float,
) -> None:
    x = data[::stride, 1]
    y = data[::stride, 2]
    z = data[::stride, 3]

    tri = mtri.Triangulation(x, y)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(tri, z, cmap="turbo", linewidth=0.0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="Altitude Z [m]")
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=elev_deg, azim=azim_deg)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot altitude heatmap from odometry topic")
    parser.add_argument("--bag", required=True, help="rosbag2 directory")
    parser.add_argument("--topic", default="/kf_gins/odom_pred", help="odometry topic")
    parser.add_argument("--out-png", default="", help="output PNG path")
    parser.add_argument("--stride", type=int, default=3, help="point decimation stride")
    parser.add_argument("--mode", choices=["scatter", "hexbin", "surface3d"], default="scatter", help="render mode")
    parser.add_argument("--gridsize", type=int, default=120, help="hexbin grid size")
    parser.add_argument("--overlay-track", action="store_true", help="overlay XY track line")
    parser.add_argument("--elev-deg", type=float, default=28.0, help="3D surface view elevation in degree")
    parser.add_argument("--azim-deg", type=float, default=-58.0, help="3D surface view azimuth in degree")
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    if not args.out_png:
        args.out_png = args.bag.rstrip("/").replace("/", "_") + "_altitude_heatmap.png"

    arr = read_odom_xyz(args.bag, args.topic)
    title = f"Altitude Heatmap ({args.topic})"
    if args.mode == "surface3d":
        plot_surface3d(arr, args.out_png, title, args.stride, args.elev_deg, args.azim_deg)
    else:
        plot_heatmap(arr, args.out_png, title, args.stride, args.mode, args.gridsize, args.overlay_track)

    print(f"bag={args.bag}")
    print(f"topic={args.topic}")
    print(f"count={len(arr)}")
    print(f"z_min={arr[:,3].min():.4f} m, z_max={arr[:,3].max():.4f} m")
    print(f"out_png={args.out_png}")


if __name__ == "__main__":
    main()
