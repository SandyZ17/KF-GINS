#!/usr/bin/env python3
"""
Evaluate a trained XGBoost AI gate model on one or more feature CSV files.

Outputs:
- metrics.json
- threshold_sweep.csv
- prediction_samples.csv (optional compact dump)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    raise SystemExit(f"xgboost is required: {e}")

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


DEFAULT_FEATURES = [
    "gnss_status",
    "hstd_m",
    "vstd_m",
    "gnss_dt_sec",
    "gnss_jump_xy_m",
    "gnss_jump_speed_xy_mps",
    "odom_speed_xy_mps",
    "nis_last",
    "nis_ema",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="XGBoost model JSON path")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "src/KF-GINS/results/ai_gate_dataset/test_demo/ai_gate_features.csv",
            "src/KF-GINS/results/ai_gate_dataset/urbannav_hk_tunnel1/ai_gate_features.csv",
        ],
    )
    ap.add_argument("--out-dir", default="src/KF-GINS/results/ai_gate_model_eval")
    ap.add_argument("--drop-uncertain", action="store_true", default=True)
    ap.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    ap.add_argument("--save-pred-samples", action="store_true", default=True)
    return ap.parse_args()


def load_inputs(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            raise SystemExit(f"Input not found: {pp}")
        df = pd.read_csv(pp)
        df["_source_file"] = str(pp)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def prepare(df: pd.DataFrame, feature_cols: List[str], drop_uncertain: bool = True):
    if drop_uncertain and "label" in df.columns:
        df = df[df["label"] != "uncertain"].copy()
    df["y"] = (df["label"].astype(str) == "reliable").astype(int)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise SystemExit("No requested feature columns found in eval input")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    medians = {c: float(df[c].median(skipna=True)) if not df[c].dropna().empty else 0.0 for c in feature_cols}
    for c in feature_cols:
        df[c] = df[c].fillna(medians[c])
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.int32)
    return df, X, y, feature_cols, medians


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_inputs(args.inputs)
    df, X, y, feature_cols, medians = prepare(df_raw, args.features, drop_uncertain=args.drop_uncertain)

    booster = xgb.Booster()
    booster.load_model(args.model)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    y_prob = booster.predict(dmat)

    # Global metrics at default threshold
    m05 = metrics_at_threshold(y, y_prob, 0.5)
    if HAVE_SKLEARN and len(np.unique(y)) > 1:
        m05["auc"] = float(roc_auc_score(y, y_prob))
    else:
        m05["auc"] = float("nan")

    # Threshold sweep
    sweep = []
    for thr in np.linspace(0.05, 0.95, 19):
        sweep.append(metrics_at_threshold(y, y_prob, float(thr)))
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Per-source metrics
    per_source = {}
    if "_source_file" in df.columns:
        for src, sub in df.groupby("_source_file"):
            idx = sub.index.to_numpy()
            ys = y[idx]
            ps = y_prob[idx]
            ms = metrics_at_threshold(ys, ps, 0.5)
            if HAVE_SKLEARN and len(np.unique(ys)) > 1:
                ms["auc"] = float(roc_auc_score(ys, ps))
            else:
                ms["auc"] = float("nan")
            per_source[src] = ms

    if args.save_pred_samples:
        pred_cols = [c for c in ["dataset", "_source_file", "t_rel_sec", "label", "target_err_xy"] if c in df.columns]
        pred_df = df[pred_cols].copy()
        pred_df["y_true"] = y
        pred_df["y_prob_reliable"] = y_prob
        pred_df["y_pred_0.5"] = (y_prob >= 0.5).astype(int)
        pred_df.to_csv(out_dir / "prediction_samples.csv", index=False)

    result = {
        "model": str(Path(args.model)),
        "inputs": args.inputs,
        "n_total": int(len(df)),
        "feature_cols": feature_cols,
        "medians": medians,
        "metrics_threshold_0.5": m05,
        "per_source_metrics_threshold_0.5": per_source,
        "threshold_sweep_csv": str(out_dir / "threshold_sweep.csv"),
        "prediction_samples_csv": str(out_dir / "prediction_samples.csv") if args.save_pred_samples else None,
        "use_sklearn": HAVE_SKLEARN,
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[DONE] AI gate model evaluation completed")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

