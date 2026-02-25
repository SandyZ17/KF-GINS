#!/usr/bin/env python3
"""
Unified analysis entry for KF-GINS datasets.

Default behavior runs all available analyses and writes outputs under one folder.
You can also choose specific modes via --modes.

Modes:
- odom_vs_truth
- odom_compare
- flatness
- altitude
- shm
- nis
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def list_topics(bag_dir: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def run_cmd(cmd, desc):
    print(f"[RUN] {desc}")
    print("      " + " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"[WARN] {desc} failed with code {res.returncode}")
        return False
    return True


def analyze_nis(bag_dir: str, topic: str, out_dir: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise RuntimeError(f"NIS topic not found: {topic}")
    msg_type = get_message(type_map[topic])

    rows = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, msg_type)
        # std_msgs/Float64 has no header; use bag read timestamp unavailable here, so index-based time.
        rows.append(float(msg.data))
    if not rows:
        raise RuntimeError(f"No NIS data on {topic}")

    nis = np.array(rows, dtype=float)
    t = np.arange(len(nis), dtype=float)
    os.makedirs(out_dir, exist_ok=True)

    q95 = 7.814727903251179  # chi2 quantile, dof=3, p=0.95
    q99 = 11.344866730144373  # dof=3, p=0.99

    metrics = {
        "count": int(len(nis)),
        "mean": float(np.mean(nis)),
        "median": float(np.median(nis)),
        "std": float(np.std(nis)),
        "p95": float(np.percentile(nis, 95)),
        "p99": float(np.percentile(nis, 99)),
        "max": float(np.max(nis)),
        "exceed_chi2_95_ratio": float(np.mean(nis > q95)),
        "exceed_chi2_99_ratio": float(np.mean(nis > q99)),
    }

    fig = plt.figure(figsize=(12, 4))
    plt.plot(t, nis, linewidth=0.8)
    plt.axhline(q95, color="orange", linestyle="--", label="chi2 dof=3 @95%")
    plt.axhline(q99, color="red", linestyle="--", label="chi2 dof=3 @99%")
    plt.xlabel("GNSS Update Index")
    plt.ylabel("NIS")
    plt.title("NIS Time Series")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "nis_timeseries.png"), dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(nis, bins=80, density=True, alpha=0.7, color="tab:blue")
    ax1.set_xlabel("NIS")
    ax1.set_ylabel("PDF")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    x = np.sort(nis)
    y = np.linspace(0.0, 1.0, len(x), endpoint=True)
    ax2.plot(x, y, color="tab:red", linewidth=1.2)
    ax2.axvline(q95, color="orange", linestyle="--")
    ax2.axvline(q99, color="red", linestyle="--")
    ax2.set_ylabel("CDF")
    plt.title("NIS Histogram + CDF")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "nis_hist_cdf.png"), dpi=220)
    plt.close(fig)

    with open(os.path.join(out_dir, "nis_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])

    print(
        "[OK] NIS: count={}, mean={:.3f}, p95={:.3f}, p99={:.3f}, >7.81={:.2%}".format(
            metrics["count"], metrics["mean"], metrics["p95"], metrics["p99"], metrics["exceed_chi2_95_ratio"]
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Unified KF-GINS analysis script")
    parser.add_argument("--bag", required=True, help="rosbag2 directory")
    parser.add_argument("--out-dir", default="", help="output root directory")
    parser.add_argument("--modes", default="all", help="comma-separated modes or 'all'")
    parser.add_argument("--truth-nav", default="src/KF-GINS/dataset/truth.nav", help="truth file path")
    parser.add_argument(
        "--truth-format",
        choices=["auto", "truth_nav", "urbannav_gt_raw"],
        default="auto",
        help="truth file format passed to analyze_odom_vs_truth.py",
    )
    parser.add_argument("--odom-pred-topic", default="/kf_gins/odom_pred")
    parser.add_argument("--odom-fused-topic", default="/kf_gins/odom_fused")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--nis-topic", default="/kf_gins/nis")
    parser.add_argument("--segment-start-sec", type=float, default=None)
    parser.add_argument("--segment-end-sec", type=float, default=None)
    args = parser.parse_args()

    bag = args.bag
    script_dir = Path(__file__).resolve().parent
    if not args.out_dir:
        args.out_dir = str(Path(bag).with_name(Path(bag).name + "_analysis"))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    topics = list_topics(bag)
    print("[INFO] Topics in bag:")
    for k, v in topics.items():
        print(f"  - {k}: {v}")

    all_modes = ["odom_vs_truth", "odom_compare", "flatness", "altitude", "shm", "nis"]
    modes = all_modes if args.modes == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]

    py = sys.executable

    if "odom_vs_truth" in modes:
        if args.odom_pred_topic in topics and Path(args.truth_nav).exists():
            truth_format = args.truth_format
            if truth_format == "auto":
                truth_name = Path(args.truth_nav).name.lower()
                if "urbannav" in truth_name and truth_name.endswith(".txt"):
                    truth_format = "urbannav_gt_raw"
                else:
                    truth_format = "truth_nav"
            run_cmd(
                [
                    py,
                    str(script_dir / "analyze_odom_vs_truth.py"),
                    "--bag",
                    bag,
                    "--odom-topic",
                    args.odom_pred_topic,
                    "--truth-nav",
                    args.truth_nav,
                    "--truth-format",
                    truth_format,
                    "--time-match",
                    "nearest",
                    "--max-time-diff",
                    "0.2" if truth_format == "urbannav_gt_raw" else "0.005",
                    "--frame-align",
                    "none",
                    "--out-prefix",
                    str(out_root / "odom_pred_vs_truth"),
                ]
                + (
                    ["--segment-start-sec", str(args.segment_start_sec)]
                    if args.segment_start_sec is not None
                    else []
                )
                + (
                    ["--segment-end-sec", str(args.segment_end_sec)]
                    if args.segment_end_sec is not None
                    else []
                ),
                "odom_pred vs truth",
            )
        else:
            print("[SKIP] odom_vs_truth (missing odom_pred topic or truth.nav)")

    if "odom_compare" in modes:
        if args.odom_fused_topic in topics and args.odom_pred_topic in topics:
            run_cmd(
                [
                    py,
                    str(script_dir / "analyze_odom_comparison.py"),
                    "--bag",
                    bag,
                    "--ref-topic",
                    args.odom_fused_topic,
                    "--cmp-topic",
                    args.odom_pred_topic,
                    "--out-prefix",
                    str(out_root / "odom_fused_vs_pred"),
                ],
                "odom_fused vs odom_pred",
            )
        else:
            print("[SKIP] odom_compare (missing odom topics)")

    if "flatness" in modes:
        odom_topic = args.odom_pred_topic if args.odom_pred_topic in topics else args.odom_fused_topic
        if odom_topic in topics:
            run_cmd(
                [
                    py,
                    str(script_dir / "analyze_flatness.py"),
                    "--bag",
                    bag,
                    "--topic",
                    odom_topic,
                    "--out-dir",
                    str(out_root / "flatness"),
                ],
                "flatness",
            )
        else:
            print("[SKIP] flatness (missing odom topic)")

    if "altitude" in modes:
        odom_topic = args.odom_pred_topic if args.odom_pred_topic in topics else args.odom_fused_topic
        if odom_topic in topics:
            run_cmd(
                [
                    py,
                    str(script_dir / "plot_altitude_heatmap.py"),
                    "--bag",
                    bag,
                    "--topic",
                    odom_topic,
                    "--mode",
                    "hexbin",
                    "--overlay-track",
                    "--out-png",
                    str(out_root / "altitude_heatmap.png"),
                ],
                "altitude heatmap",
            )
            run_cmd(
                [
                    py,
                    str(script_dir / "plot_altitude_heatmap.py"),
                    "--bag",
                    bag,
                    "--topic",
                    odom_topic,
                    "--mode",
                    "surface3d",
                    "--stride",
                    "6",
                    "--out-png",
                    str(out_root / "altitude_surface3d.png"),
                ],
                "altitude surface3d",
            )
        else:
            print("[SKIP] altitude (missing odom topic)")

    if "shm" in modes:
        odom_topic = args.odom_fused_topic if args.odom_fused_topic in topics else args.odom_pred_topic
        if args.imu_topic in topics:
            run_cmd(
                [
                    py,
                    str(script_dir / "analyze_shm.py"),
                    "--bag",
                    bag,
                    "--imu-topic",
                    args.imu_topic,
                    "--odom-topic",
                    odom_topic,
                    "--out-dir",
                    str(out_root / "shm"),
                ],
                "shm",
            )
        else:
            print("[SKIP] shm (missing imu topic)")

    if "nis" in modes:
        if args.nis_topic in topics:
            try:
                analyze_nis(bag, args.nis_topic, str(out_root / "nis"))
            except Exception as e:
                print(f"[WARN] NIS analysis failed: {e}")
        else:
            print("[SKIP] nis (missing nis topic)")

    print(f"[DONE] Unified analysis outputs: {out_root}")


if __name__ == "__main__":
    main()
