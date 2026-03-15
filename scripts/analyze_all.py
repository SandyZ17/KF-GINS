#!/usr/bin/env python3
"""
Unified analysis entry for KF-GINS datasets.

Default behavior runs all available analyses and writes outputs under one folder.
You can also choose specific modes via --modes.

Modes:
- odom_vs_gps
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


def read_metrics_csv(path: Path):
    metrics = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 2:
                continue
            key, value = row
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


def write_compare_summary(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def analyze_nis(bag_dir: str, topic: str, out_dir: str, nis_mode: str = "xyz"):
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

    if nis_mode == "xy":
        dof = 2
        q95 = 5.991464547107979  # chi2 quantile, dof=2, p=0.95
        q99 = 9.21034037197618  # dof=2, p=0.99
    else:
        dof = 3
        q95 = 7.814727903251179  # chi2 quantile, dof=3, p=0.95
        q99 = 11.344866730144373  # dof=3, p=0.99

    metrics = {
        "count": int(len(nis)),
        "nis_mode": nis_mode,
        "chi2_dof": dof,
        "chi2_q95": q95,
        "chi2_q99": q99,
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
    plt.axhline(q95, color="orange", linestyle="--", label=f"chi2 dof={dof} @95%")
    plt.axhline(q99, color="red", linestyle="--", label=f"chi2 dof={dof} @99%")
    plt.xlabel("GNSS Update Index")
    plt.ylabel("NIS")
    plt.title("NIS Time Series")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "nis_timeseries.svg"), dpi=600)
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
    fig.savefig(os.path.join(out_dir, "nis_hist_cdf.svg"), dpi=600)
    plt.close(fig)

    with open(os.path.join(out_dir, "nis_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])

    print(
        "[OK] NIS ({mode}, dof={dof}): count={count}, mean={mean:.3f}, p95={p95:.3f}, p99={p99:.3f}, >q95({q95:.2f})={ratio:.2%}".format(
            mode=metrics["nis_mode"],
            dof=metrics["chi2_dof"],
            count=metrics["count"],
            mean=metrics["mean"],
            p95=metrics["p95"],
            p99=metrics["p99"],
            q95=metrics["chi2_q95"],
            ratio=metrics["exceed_chi2_95_ratio"],
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
    parser.add_argument("--gps-topic", default="/gps/fix")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--nis-topic", default="/kf_gins/nis")
    parser.add_argument(
        "--nis-mode",
        choices=["xy", "xyz"],
        default="xyz",
        help="NIS evaluation dimension mode. Use 'xy' for 2D update, 'xyz' for 3D update.",
    )
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

    all_modes = ["odom_vs_gps", "odom_vs_truth", "odom_compare", "flatness", "altitude", "shm", "nis"]
    modes = all_modes if args.modes == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]

    py = sys.executable
    compare_summary_rows = []

    if "odom_vs_gps" in modes:
        gps_topic = args.gps_topic
        if gps_topic in topics:
            for odom_topic, label in (
                (args.odom_pred_topic, "odom_pred_vs_raw_gps"),
                (args.odom_fused_topic, "odom_fused_vs_raw_gps"),
            ):
                if odom_topic not in topics:
                    continue
                out_prefix = out_root / label
                ok = run_cmd(
                    [
                        py,
                        str(script_dir / "analyze_odom_vs_gps.py"),
                        "--bag",
                        bag,
                        "--odom-topic",
                        odom_topic,
                        "--gps-topic",
                        gps_topic,
                        "--time-match",
                        "nearest",
                        "--max-time-diff",
                        "0.2",
                        "--out-prefix",
                        str(out_prefix),
                    ],
                    f"{label} (APE/RPE)",
                )
                if ok:
                    metrics = read_metrics_csv(Path(f"{out_prefix}_metrics.csv"))
                    if metrics:
                        compare_summary_rows.append(
                            {
                                "comparison": label,
                                "odom_topic": odom_topic,
                                "gps_topic": gps_topic,
                                "ape_rmse_xy_m": metrics.get("ape_rmse_xy_m"),
                                "ape_p95_xy_m": metrics.get("ape_p95_xy_m"),
                                "ape_max_xy_m": metrics.get("ape_max_xy_m"),
                                "ape_rmse_3d_m": metrics.get("ape_rmse_3d_m"),
                                "rpe_delta_sec": metrics.get("rpe_delta_sec"),
                                "rpe_rmse_xy_m": metrics.get("rpe_rmse_xy_m"),
                                "rpe_p95_xy_m": metrics.get("rpe_p95_xy_m"),
                                "rpe_max_xy_m": metrics.get("rpe_max_xy_m"),
                                "rpe_rmse_3d_m": metrics.get("rpe_rmse_3d_m"),
                                "matched_count": metrics.get("matched_count"),
                                "ape_max_match_dt_s": metrics.get("ape_max_match_dt_s"),
                            }
                        )
            if compare_summary_rows:
                write_compare_summary(out_root / "compare_metrics_ape_rpe.csv", compare_summary_rows)
        else:
            print("[SKIP] odom_vs_gps (missing /gps/fix topic)")

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
                "odom_pred vs truth (APE/RPE)",
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
                "odom_fused vs odom_pred (APE/RPE)",
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
                    str(out_root / "altitude_heatmap.svg"),
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
                    str(out_root / "altitude_surface3d.svg"),
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
                analyze_nis(bag, args.nis_topic, str(out_root / "nis"), args.nis_mode)
            except Exception as e:
                print(f"[WARN] NIS analysis failed: {e}")
        else:
            print("[SKIP] nis (missing nis topic)")

    if compare_summary_rows:
        print(f"[OK] APE/RPE compare summary: {out_root / 'compare_metrics_ape_rpe.csv'}")

    print(f"[DONE] Unified analysis outputs: {out_root}")


if __name__ == "__main__":
    main()
