#!/usr/bin/env python3
"""
Road SHM (Structural Health Monitoring) analysis from ROS2 bag.

This script uses IMU linear acceleration to build a road-health index and
exports:
1) SHM heatmap in XY (if odom topic is available)
2) SHM profile along distance/time
3) IMU spectrogram
4) Event CSV (high-risk samples)
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


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.copy()
    k = np.ones(n, dtype=float) / float(n)
    return np.convolve(x, k, mode="same")


def robust_norm(v: np.ndarray) -> np.ndarray:
    lo = np.percentile(v, 5)
    hi = np.percentile(v, 95)
    if hi <= lo + 1e-12:
        return np.zeros_like(v)
    out = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def read_imu_odom(bag_dir: str, imu_topic: str, odom_topic: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if imu_topic not in type_map:
        raise RuntimeError(f"IMU topic not found: {imu_topic}")
    imu_type = get_message(type_map[imu_topic])
    odom_type = get_message(type_map[odom_topic]) if odom_topic in type_map else None

    imu_rows = []
    odom_rows = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name == imu_topic:
            msg = deserialize_message(data, imu_type)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            a = msg.linear_acceleration
            imu_rows.append((t, a.x, a.y, a.z))
        elif odom_type is not None and name == odom_topic:
            msg = deserialize_message(data, odom_type)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            p = msg.pose.pose.position
            odom_rows.append((t, p.x, p.y))

    if not imu_rows:
        raise RuntimeError("No IMU samples found")

    imu = np.array(imu_rows, dtype=float)
    keep_i = np.concatenate(([0], np.where(np.diff(imu[:, 0]) > 0)[0] + 1))
    imu = imu[keep_i]

    odom = None
    if odom_rows:
        odom = np.array(odom_rows, dtype=float)
        keep_o = np.concatenate(([0], np.where(np.diff(odom[:, 0]) > 0)[0] + 1))
        odom = odom[keep_o]
    return imu, odom


def compute_shm_index(
    imu: np.ndarray, window_sec: float, step_sec: float, trend_sec: float, band_low: float, band_high: float
):
    t = imu[:, 0]
    ax, ay, az = imu[:, 1], imu[:, 2], imu[:, 3]
    amag = np.sqrt(ax * ax + ay * ay + az * az)

    dt = np.diff(t)
    dt = dt[dt > 1e-6]
    fs = float(1.0 / np.median(dt))

    trend_n = max(1, int(round(trend_sec * fs)))
    sig = amag - moving_average(amag, trend_n)

    win_n = max(8, int(round(window_sec * fs)))
    step_n = max(1, int(round(step_sec * fs)))
    if len(sig) < win_n:
        raise RuntimeError("Not enough IMU data for selected window")

    tc = []
    rms_v = []
    peak_v = []
    bp_v = []
    hann = np.hanning(win_n)
    freqs = np.fft.rfftfreq(win_n, d=1.0 / fs)
    band = (freqs >= band_low) & (freqs <= band_high)

    for i0 in range(0, len(sig) - win_n + 1, step_n):
        i1 = i0 + win_n
        seg = sig[i0:i1]
        tc.append(0.5 * (t[i0] + t[i1 - 1]))
        rms_v.append(float(np.sqrt(np.mean(seg * seg))))
        peak_v.append(float(np.max(np.abs(seg))))
        sp = np.fft.rfft(seg * hann)
        p = (np.abs(sp) ** 2) / max(1.0, np.sum(hann * hann))
        bp_v.append(float(np.mean(p[band])) if np.any(band) else 0.0)

    tc = np.array(tc, dtype=float)
    rms_v = np.array(rms_v, dtype=float)
    peak_v = np.array(peak_v, dtype=float)
    bp_v = np.array(bp_v, dtype=float)

    idx01 = 0.45 * robust_norm(rms_v) + 0.35 * robust_norm(peak_v) + 0.20 * robust_norm(bp_v)
    idx100 = 100.0 * idx01
    return tc, idx100, sig, fs


def interpolate_xy(odom: np.ndarray, tq: np.ndarray):
    to = odom[:, 0]
    if tq[0] < to[0] or tq[-1] > to[-1]:
        keep = (tq >= to[0]) & (tq <= to[-1])
    else:
        keep = np.ones_like(tq, dtype=bool)
    tq2 = tq[keep]
    x = np.interp(tq2, to, odom[:, 1])
    y = np.interp(tq2, to, odom[:, 2])
    return tq2, x, y, keep


def cumulative_distance(x: np.ndarray, y: np.ndarray):
    d = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate(([0.0], np.cumsum(d)))


def main():
    ap = argparse.ArgumentParser(description="Road SHM analysis from IMU")
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--imu-topic", default="/imu/data", help="IMU topic")
    ap.add_argument("--odom-topic", default="/kf_gins/odom_pred", help="odom topic for XY mapping")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--window-sec", type=float, default=1.0, help="feature window length")
    ap.add_argument("--step-sec", type=float, default=0.1, help="feature step")
    ap.add_argument("--trend-sec", type=float, default=0.5, help="moving-average detrend length")
    ap.add_argument("--band-low", type=float, default=5.0, help="band-pass low frequency [Hz]")
    ap.add_argument("--band-high", type=float, default=30.0, help="band-pass high frequency [Hz]")
    ap.add_argument("--event-threshold", type=float, default=-1.0, help="fixed event threshold in [0,100], <0 uses p95")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imu, odom = read_imu_odom(args.bag, args.imu_topic, args.odom_topic)
    tc, idx, sig, fs = compute_shm_index(
        imu, args.window_sec, args.step_sec, args.trend_sec, args.band_low, args.band_high
    )

    if args.event_threshold < 0.0:
        thr = float(np.percentile(idx, 95))
    else:
        thr = float(args.event_threshold)
    is_event = idx >= thr

    # XY mapping (if odom exists)
    have_xy = odom is not None and len(odom) > 2
    if have_xy:
        tq, x, y, keep = interpolate_xy(odom, tc)
        idx_xy = idx[keep]
        evt_xy = is_event[keep]
        s = cumulative_distance(x, y)
    else:
        tq = tc
        idx_xy = idx
        evt_xy = is_event
        x = np.full_like(tq, np.nan)
        y = np.full_like(tq, np.nan)
        s = tq - tq[0]

    # 1) Heatmap
    fig = plt.figure(figsize=(10, 8))
    if have_xy:
        sc = plt.scatter(x, y, c=idx_xy, cmap="turbo", s=8)
        cb = plt.colorbar(sc)
        cb.set_label("SHM Index [0-100]")
        if np.any(evt_xy):
            plt.scatter(x[evt_xy], y[evt_xy], s=15, c="red", marker="x", label="events")
            plt.legend(loc="best")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis("equal")
        plt.title("Road SHM Heatmap (XY)")
    else:
        plt.plot(tq - tq[0], idx_xy)
        plt.xlabel("Time [s]")
        plt.ylabel("SHM Index [0-100]")
        plt.title("Road SHM Index (no odom topic)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "shm_heatmap_xy.svg"), dpi=600)
    plt.close(fig)

    # 2) Along-track profile
    fig = plt.figure(figsize=(12, 5))
    plt.plot(s, idx_xy, linewidth=1.0, label="SHM index")
    plt.axhline(thr, color="red", linestyle="--", linewidth=1.0, label=f"threshold={thr:.2f}")
    plt.xlabel("Distance [m]" if have_xy else "Relative Time [s]")
    plt.ylabel("SHM Index [0-100]")
    plt.title("Road SHM Profile")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "shm_profile.svg"), dpi=600)
    plt.close(fig)

    # 3) Spectrogram
    fig = plt.figure(figsize=(12, 5))
    nfft = max(64, int(round(fs * 2.0)))
    noverlap = int(0.75 * nfft)
    plt.specgram(sig, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap="turbo")
    plt.colorbar(label="Power/Frequency [dB]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("IMU Signal Spectrogram (detrended |a|)")
    plt.ylim(0, max(args.band_high * 2.0, 50.0))
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "shm_spectrogram.svg"), dpi=600)
    plt.close(fig)

    # 4) Event CSV
    csv_path = os.path.join(args.out_dir, "shm_events.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "distance", "x", "y", "shm_index"])
        for i in range(len(tq)):
            if not evt_xy[i]:
                continue
            w.writerow([f"{tq[i]:.6f}", f"{s[i]:.3f}", f"{x[i]:.3f}", f"{y[i]:.3f}", f"{idx_xy[i]:.3f}"])

    # metrics
    metrics_path = os.path.join(args.out_dir, "shm_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["sample_count", len(idx_xy)])
        w.writerow(["fs_hz", f"{fs:.4f}"])
        w.writerow(["index_mean", f"{np.mean(idx_xy):.4f}"])
        w.writerow(["index_std", f"{np.std(idx_xy):.4f}"])
        w.writerow(["index_p95", f"{np.percentile(idx_xy, 95):.4f}"])
        w.writerow(["threshold", f"{thr:.4f}"])
        w.writerow(["event_count", int(np.sum(evt_xy))])

    print(f"bag={args.bag}")
    print(f"imu_topic={args.imu_topic}")
    print(f"odom_topic={args.odom_topic} (available={have_xy})")
    print(f"out_dir={args.out_dir}")
    print(f"samples={len(idx_xy)}, fs={fs:.2f} Hz, threshold={thr:.2f}, events={int(np.sum(evt_xy))}")
    print(f"metrics_csv={metrics_path}")
    print(f"events_csv={csv_path}")


if __name__ == "__main__":
    main()
