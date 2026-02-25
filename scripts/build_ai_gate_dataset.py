#!/usr/bin/env python3
"""
Build a first-stage AI gate training dataset from existing KF-GINS analysis outputs.

This script is intentionally conservative:
- It reuses current analysis scripts (truth alignment + error CSV generation)
- It exports a labeled XY-error dataset for AI-gate prototyping
- It does NOT yet extract full online GNSS/NIS feature vectors from rosbag (phase 2)

Recommended usage:
  python3 src/KF-GINS/scripts/build_ai_gate_dataset.py \
    --source src/KF-GINS/docs/ai_gating_design/dataset_sources.yaml \
    --dataset test_demo \
    --label-profile conservative_xy

  python3 src/KF-GINS/scripts/build_ai_gate_dataset.py \
    --source src/KF-GINS/docs/ai_gating_design/dataset_sources.yaml \
    --dataset urbannav_hk_tunnel1 \
    --label-profile urban_degraded_xy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parents[1]  # src/KF-GINS


@dataclass
class LabelProfile:
    pos_xy_good_m: float
    pos_xy_bad_m: float


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_workspace_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return ROOT.parents[1] / pp  # workspace root / relative


def run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_truth_analysis(ds: Dict[str, Any]) -> Path:
    out_dir = resolve_workspace_path(ds["analyze_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = out_dir / "odom_pred_vs_truth"
    err_csv = Path(str(out_prefix) + "_vs_truth_error.csv")
    if err_csv.exists():
        return err_csv

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_odom_vs_truth.py"),
        "--bag",
        str(resolve_workspace_path(ds["bag"])),
        "--odom-topic",
        ds.get("odom_topic", "/kf_gins/odom_pred"),
        "--truth-nav",
        str(resolve_workspace_path(ds["truth"])),
        "--truth-format",
        ds.get("truth_format", "auto"),
        "--time-match",
        "nearest",
        "--frame-align",
        "se3",
        "--out-prefix",
        str(out_prefix),
    ]

    if ds.get("truth_format") == "urbannav_gt_raw":
        cmd += ["--max-time-diff", "0.2"]
    else:
        cmd += ["--max-time-diff", "0.05"]

    if "segment_start_sec" in ds:
        cmd += ["--segment-start-sec", str(ds["segment_start_sec"])]
    if "segment_end_sec" in ds:
        cmd += ["--segment-end-sec", str(ds["segment_end_sec"])]

    run(cmd)
    return err_csv


def iter_error_rows(err_csv: Path) -> Iterable[Dict[str, float]]:
    with err_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ex = float(row["err_px"])
            ey = float(row["err_py"])
            ez = float(row["err_pz"])
            evx = float(row["err_vx"])
            evy = float(row["err_vy"])
            evz = float(row["err_vz"])
            t = float(row.get("t_rel_sec", row.get("t", "nan")))
            err_xy = math.hypot(ex, ey)
            yield {
                "t_rel_sec": t,
                "err_px": ex,
                "err_py": ey,
                "err_pz": ez,
                "err_xy": err_xy,
                "err_pos_norm": float(row["err_pos_norm"]),
                "err_vx": evx,
                "err_vy": evy,
                "err_vz": evz,
                "err_vel_norm": float(row["err_vel_norm"]),
            }


def label_row(row: Dict[str, float], profile: LabelProfile) -> str:
    if row["err_xy"] <= profile.pos_xy_good_m:
        return "reliable"
    if row["err_xy"] >= profile.pos_xy_bad_m:
        return "unreliable"
    return "uncertain"


def build_dataset(ds_name: str, ds: Dict[str, Any], profile_name: str, profile: LabelProfile) -> Dict[str, Any]:
    err_csv = ensure_truth_analysis(ds)
    out_dir = resolve_workspace_path(ds["build_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_csv = out_dir / "ai_gate_label_samples.csv"
    summary_json = out_dir / "ai_gate_label_summary.json"

    rows = list(iter_error_rows(err_csv))
    labeled: List[Dict[str, Any]] = []
    counts = {"reliable": 0, "uncertain": 0, "unreliable": 0}
    for row in rows:
        label = label_row(row, profile)
        counts[label] += 1
        labeled.append({
            "dataset": ds_name,
            "t_rel_sec": row["t_rel_sec"],
            "err_xy": row["err_xy"],
            "err_pz": row["err_pz"],
            "err_pos_norm": row["err_pos_norm"],
            "err_vel_norm": row["err_vel_norm"],
            "label": label,
        })

    with samples_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "t_rel_sec",
                "err_xy",
                "err_pz",
                "err_pos_norm",
                "err_vel_norm",
                "label",
            ],
        )
        w.writeheader()
        w.writerows(labeled)

    summary = {
        "dataset": ds_name,
        "label_profile": profile_name,
        "source_bag": ds["bag"],
        "source_truth": ds["truth"],
        "truth_format": ds.get("truth_format", "auto"),
        "error_csv": str(err_csv),
        "samples_csv": str(samples_csv),
        "counts": counts,
        "total": len(labeled),
        "notes": [
            "This is a stage-1 label dataset (truth-error based).",
            "Runtime GNSS/NIS feature extraction is not included yet.",
            "Use this output to prototype AI gate labels and train/eval pipeline scaffolding.",
        ],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="src/KF-GINS/docs/ai_gating_design/dataset_sources.yaml",
        help="Dataset source config YAML",
    )
    ap.add_argument("--dataset", required=True, help="Dataset key in source YAML")
    ap.add_argument(
        "--label-profile",
        default="urban_degraded_xy",
        help="Label profile key in labeling_profiles.yaml",
    )
    ap.add_argument(
        "--label-profile-file",
        default="src/KF-GINS/docs/ai_gating_design/labeling_profiles.yaml",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src_cfg = load_yaml(resolve_workspace_path(args.source))
    lbl_cfg = load_yaml(resolve_workspace_path(args.label_profile_file))

    datasets = src_cfg.get("datasets", {})
    profiles = lbl_cfg.get("profiles", {})
    if args.dataset not in datasets:
        raise SystemExit(f"Dataset not found in source config: {args.dataset}")
    if args.label_profile not in profiles:
        raise SystemExit(f"Label profile not found: {args.label_profile}")

    p = profiles[args.label_profile]
    profile = LabelProfile(
        pos_xy_good_m=float(p["pos_xy_good_m"]),
        pos_xy_bad_m=float(p["pos_xy_bad_m"]),
    )

    summary = build_dataset(args.dataset, datasets[args.dataset], args.label_profile, profile)
    print("[DONE] AI gate label dataset generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

