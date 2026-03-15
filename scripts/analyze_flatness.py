#!/usr/bin/env python3
"""
Analyze surface flatness from odometry trajectory and export visualizations.

Outputs:
1) XY residual heatmap (after plane detrending)
2) Along-track residual profile
3) Residual histogram + CDF
4) Spatial PSD (roughness by wavelength)
5) Metrics CSV
"""

import argparse
import csv
import os

import matplotlib
import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
        raise RuntimeError(f"No messages from {topic}")

    arr = np.array(rows, dtype=float)
    keep = np.concatenate(([0], np.where(np.diff(arr[:, 0]) > 0)[0] + 1))
    return arr[keep]


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    return np.concatenate(([0.0], np.cumsum(ds)))


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    a = np.column_stack([x, y, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    return a @ coef


def moving_rms(v: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.abs(v)
    k = np.ones(win, dtype=float) / float(win)
    return np.sqrt(np.convolve(v * v, k, mode="same"))


def save_metrics_csv(path: str, metrics: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatness analysis from odometry")
    parser.add_argument("--bag", required=True, help="rosbag2 directory")
    parser.add_argument("--topic", default="/kf_gins/odom_pred", help="odometry topic")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--plot-stride", type=int, default=2, help="decimation for plotting")
    parser.add_argument("--rms-window", type=int, default=151, help="window size (samples) for moving RMS")
    args = parser.parse_args()

    if args.plot_stride < 1:
        raise ValueError("--plot-stride must be >= 1")
    if args.rms_window < 1:
        raise ValueError("--rms-window must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)

    arr = read_odom_xyz(args.bag, args.topic)
    t = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    z = arr[:, 3]
    s = cumulative_distance(x, y)

    z_plane = fit_plane(x, y, z)
    z_res = z - z_plane
    z_abs = np.abs(z_res)
    z_rms_curve = moving_rms(z_res, args.rms_window)

    metrics = {
        "count": int(len(z_res)),
        "duration_sec": float(t[-1] - t[0]),
        "path_length_m": float(s[-1]),
        "residual_mean_m": float(np.mean(z_res)),
        "residual_std_m": float(np.std(z_res)),
        "residual_rms_m": float(np.sqrt(np.mean(z_res * z_res))),
        "residual_p95_abs_m": float(np.percentile(z_abs, 95)),
        "residual_max_abs_m": float(np.max(z_abs)),
    }

    # 1) XY residual heatmap
    fig = plt.figure(figsize=(10, 8))
    sc = plt.scatter(x[:: args.plot_stride], y[:: args.plot_stride], c=z_res[:: args.plot_stride], s=4, cmap="RdYlBu_r")
    cb = plt.colorbar(sc)
    cb.set_label("Detrended Height Residual [m]")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Flatness Residual Heatmap (XY)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "flatness_heatmap_xy.svg"), dpi=600)
    plt.close(fig)

    # 2) Along-track profile
    fig = plt.figure(figsize=(12, 5))
    plt.plot(s, z_res, linewidth=0.7, label="residual z")
    plt.plot(s, z_rms_curve, linewidth=1.0, label="moving RMS")
    plt.xlabel("Distance Along Track [m]")
    plt.ylabel("Height Residual [m]")
    plt.title("Along-Track Flatness Residual")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "flatness_profile.svg"), dpi=600)
    plt.close(fig)

    # 3) Histogram + CDF
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(z_res, bins=120, density=True, alpha=0.65, color="tab:blue")
    ax1.set_xlabel("Height Residual [m]")
    ax1.set_ylabel("PDF")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    sorted_abs = np.sort(z_abs)
    cdf = np.linspace(0.0, 1.0, len(sorted_abs), endpoint=True)
    ax2.plot(sorted_abs, cdf, color="tab:red", linewidth=1.4)
    ax2.set_ylabel("CDF(|residual|)")
    plt.title("Residual Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "flatness_hist_cdf.svg"), dpi=600)
    plt.close(fig)

    # 4) Spatial PSD (residual vs distance)
    ds = np.diff(s)
    ds_pos = ds[ds > 1e-6]
    if len(ds_pos) > 0:
        ds_uniform = float(np.median(ds_pos))
        su = np.arange(0.0, s[-1], ds_uniform)
        zu = np.interp(su, s, z_res)
        zu = zu - np.mean(zu)
        n = len(zu)
        if n > 8:
            win = np.hanning(n)
            zf = np.fft.rfft(zu * win)
            psd = (np.abs(zf) ** 2) / (np.sum(win * win))
            freq = np.fft.rfftfreq(n, d=ds_uniform)  # cycles/m
            valid = freq > 0
            fig = plt.figure(figsize=(10, 5))
            plt.loglog(freq[valid], psd[valid])
            plt.xlabel("Spatial Frequency [cycles/m]")
            plt.ylabel("PSD")
            plt.title("Spatial PSD of Height Residual")
            plt.grid(True, which="both", alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "flatness_psd.svg"), dpi=600)
            plt.close(fig)

    metrics_path = os.path.join(args.out_dir, "flatness_metrics.csv")
    save_metrics_csv(metrics_path, metrics)

    print(f"bag={args.bag}")
    print(f"topic={args.topic}")
    print(f"out_dir={args.out_dir}")
    print(f"count={metrics['count']}, path_length_m={metrics['path_length_m']:.3f}")
    print(
        "rms={:.6f} m, p95_abs={:.6f} m, max_abs={:.6f} m".format(
            metrics["residual_rms_m"], metrics["residual_p95_abs_m"], metrics["residual_max_abs_m"]
        )
    )
    print(f"metrics_csv={metrics_path}")


if __name__ == "__main__":
    main()
