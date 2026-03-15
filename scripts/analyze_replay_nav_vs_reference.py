#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WGS84_A = 6378137.0


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q))


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_replay_nav(path: Path) -> np.ndarray:
    rows = read_csv_dicts(path)
    if not rows:
        raise RuntimeError(f"Replay nav csv is empty: {path}")

    out = np.array(
        [
            [
                float(r["time_sec"]),
                float(r["local_east_m"]),
                float(r["local_north_m"]),
                -float(r["local_down_m"]),
                float(r["vel_e_mps"]),
                float(r["vel_n_mps"]),
                -float(r["vel_d_mps"]),
                float(r["lat_deg"]),
                float(r["lon_deg"]),
                float(r["height_m"]),
            ]
            for r in rows
        ],
        dtype=float,
    )
    keep = np.concatenate(([0], np.where(np.diff(out[:, 0]) > 0)[0] + 1))
    return out[keep]


def read_reference(path: Path) -> np.ndarray:
    rows = read_csv_dicts(path)
    if not rows:
        raise RuntimeError(f"Reference csv is empty: {path}")

    out = np.array(
        [
            [
                float(r["timestamp"]),
                float(r["px"]),
                float(r["py"]),
                float(r["pz"]),
                float(r["vx"]),
                float(r["vy"]),
                float(r["vz"]),
                float(r["latitude_deg"]) if "latitude_deg" in r and r["latitude_deg"] != "" else np.nan,
                float(r["longitude_deg"]) if "longitude_deg" in r and r["longitude_deg"] != "" else np.nan,
                float(r["altitude_m"]) if "altitude_m" in r and r["altitude_m"] != "" else np.nan,
            ]
            for r in rows
        ],
        dtype=float,
    )
    keep = np.concatenate(([0], np.where(np.diff(out[:, 0]) > 0)[0] + 1))
    return out[keep]


def align_interp(
    nav: np.ndarray,
    ref: np.ndarray,
    use_relative_time: bool,
    nav_time_shift_sec: float,
    reference_time_shift_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nav_t = nav[:, 0].copy()
    ref_t = ref[:, 0].copy()
    nav_t += float(nav_time_shift_sec)
    ref_t += float(reference_time_shift_sec)
    if use_relative_time:
        nav_t -= nav_t[0]
        ref_t -= ref_t[0]

    t0 = max(nav_t[0], ref_t[0])
    t1 = min(nav_t[-1], ref_t[-1])
    if t1 <= t0:
        raise RuntimeError("No overlapping time range between replay_nav and reference")

    keep = (nav_t >= t0) & (nav_t <= t1)
    nav_a = nav[keep].copy()
    t = nav_t[keep]
    nav_a[:, 0] = t

    ref_i = np.zeros((len(t), ref.shape[1]), dtype=float)
    ref_i[:, 0] = t
    for c in range(1, ref.shape[1]):
        ref_i[:, c] = np.interp(t, ref_t, ref[:, c])
    return t, nav_a, ref_i


def geodetic_series_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray, origin: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(float(origin[0]))
    lon0 = np.deg2rad(float(origin[1]))
    east = (np.deg2rad(lon_deg) - lon0) * np.cos(lat0) * WGS84_A
    north = (np.deg2rad(lat_deg) - lat0) * WGS84_A
    up = h_m - float(origin[2])
    return np.column_stack([east, north, up])


def use_common_geodetic_frame(nav_a: np.ndarray, ref_i: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    nav_geo = nav_a[:, 7:10]
    ref_geo = ref_i[:, 7:10]
    if not (np.all(np.isfinite(nav_geo)) and np.all(np.isfinite(ref_geo))):
        return nav_a, ref_i, "local_xy_from_inputs"

    origin = ref_geo[0]
    nav_pos_enu = geodetic_series_to_enu(nav_geo[:, 0], nav_geo[:, 1], nav_geo[:, 2], origin)
    ref_pos_enu = geodetic_series_to_enu(ref_geo[:, 0], ref_geo[:, 1], ref_geo[:, 2], origin)

    nav_out = nav_a.copy()
    ref_out = ref_i.copy()
    nav_out[:, 1:4] = nav_pos_enu
    ref_out[:, 1:4] = ref_pos_enu
    return nav_out, ref_out, "common_geodetic_enu"


def compute_metrics(err_pos: np.ndarray, err_vel: np.ndarray) -> dict[str, float]:
    err_xy = np.linalg.norm(err_pos[:, :2], axis=1)
    err_3d = np.linalg.norm(err_pos, axis=1)
    err_vel_norm = np.linalg.norm(err_vel, axis=1)
    return {
        "ape_rmse_xy_m": rmse(err_xy),
        "ape_p95_xy_m": percentile(err_xy, 95),
        "ape_max_xy_m": float(np.max(err_xy)),
        "ape_rmse_3d_m": rmse(err_3d),
        "ape_p95_3d_m": percentile(err_3d, 95),
        "ape_max_3d_m": float(np.max(err_3d)),
        "vel_rmse_mps": rmse(err_vel_norm),
        "vel_p95_mps": percentile(err_vel_norm, 95),
        "vel_max_mps": float(np.max(err_vel_norm)),
    }


def save_error_csv(path: Path, t: np.ndarray, err_pos: np.ndarray, err_vel: np.ndarray) -> None:
    err_xy = np.linalg.norm(err_pos[:, :2], axis=1)
    err_3d = np.linalg.norm(err_pos, axis=1)
    err_vel_norm = np.linalg.norm(err_vel, axis=1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t_sec",
                "err_px_m",
                "err_py_m",
                "err_pz_m",
                "err_xy_m",
                "err_3d_m",
                "err_vx_mps",
                "err_vy_mps",
                "err_vz_mps",
                "err_vel_norm_mps",
            ]
        )
        for i in range(len(t)):
            writer.writerow(
                [
                    f"{t[i]:.9f}",
                    f"{err_pos[i, 0]:.6f}",
                    f"{err_pos[i, 1]:.6f}",
                    f"{err_pos[i, 2]:.6f}",
                    f"{err_xy[i]:.6f}",
                    f"{err_3d[i]:.6f}",
                    f"{err_vel[i, 0]:.6f}",
                    f"{err_vel[i, 1]:.6f}",
                    f"{err_vel[i, 2]:.6f}",
                    f"{err_vel_norm[i]:.6f}",
                ]
            )


def save_trajectory_plot(path: Path, nav: np.ndarray, ref: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ref[:, 1], ref[:, 2], label="Reference", linewidth=1.4)
    ax.plot(nav[:, 1], nav[:, 2], label="Replay", linewidth=1.2)
    ax.scatter(ref[0, 1], ref[0, 2], c="green", marker="o", s=30, label="Start")
    ax.scatter(ref[-1, 1], ref[-1, 2], c="red", marker="x", s=40, label="End")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title("Replay Trajectory vs Reference")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_error_plot(path: Path, t: np.ndarray, err_pos: np.ndarray) -> None:
    err_xy = np.linalg.norm(err_pos[:, :2], axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ["err_east [m]", "err_north [m]", "err_up [m]", "err_xy [m]"]
    series = [err_pos[:, 0], err_pos[:, 1], err_pos[:, 2], err_xy]
    for ax, y, label in zip(axes, series, labels):
        ax.plot(t, y, linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Replay Position Error")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_velocity_error_plot(path: Path, t: np.ndarray, err_vel: np.ndarray) -> None:
    err_vel_norm = np.linalg.norm(err_vel, axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ["err_vx [m/s]", "err_vy [m/s]", "err_vz [m/s]", "err_vnorm [m/s]"]
    series = [err_vel[:, 0], err_vel[:, 1], err_vel[:, 2], err_vel_norm]
    for ax, y, label in zip(axes, series, labels):
        ax.plot(t, y, linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Replay Velocity Error")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def compute_uniform_psd(t: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dt = np.diff(t)
    dt = dt[dt > 1e-6]
    if len(dt) == 0:
        raise RuntimeError("Not enough timestamps to estimate sampling rate")
    dt_uniform = float(np.median(dt))
    fs = 1.0 / dt_uniform
    t_uniform = np.arange(t[0], t[-1] + 0.5 * dt_uniform, dt_uniform)
    sig_uniform = np.interp(t_uniform, t, signal)
    sig_uniform = sig_uniform - np.mean(sig_uniform)

    n = len(sig_uniform)
    if n < 8:
        raise RuntimeError("Signal too short for spectrum analysis")

    win = np.hanning(n)
    spec = np.fft.rfft(sig_uniform * win)
    freq = np.fft.rfftfreq(n, d=dt_uniform)
    psd = (np.abs(spec) ** 2) / (fs * np.sum(win * win))
    if len(psd) > 2:
        psd[1:-1] *= 2.0
    return freq, psd, fs


def save_spectrum(path_svg: Path, path_csv: Path, t: np.ndarray, components: dict[str, np.ndarray], title: str) -> None:
    if not components:
        raise RuntimeError("No components provided for spectrum export")

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fs_used = None
    for name, sig in components.items():
        freq, psd, fs = compute_uniform_psd(t, sig)
        spectra[name] = (freq, psd)
        if fs_used is None:
            fs_used = fs

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, (freq, psd) in spectra.items():
        valid = freq > 0.0
        ax.loglog(freq[valid], psd[valid], linewidth=1.0, label=name)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title(f"{title} (fs={fs_used:.2f} Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_svg, dpi=300)
    plt.close(fig)

    first_key = next(iter(components.keys()))
    freq_ref = spectra[first_key][0]
    with path_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frequency_hz"] + [f"psd_{name}" for name in components.keys()])
        for i in range(len(freq_ref)):
            writer.writerow([f"{freq_ref[i]:.9f}"] + [f"{spectra[name][1][i]:.12e}" for name in components.keys()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KF-GINS replay_nav.csv against reference.csv")
    parser.add_argument("--replay-nav-csv", required=True, type=Path)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--time-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Align by relative time from each sequence start, or absolute timestamps",
    )
    parser.add_argument(
        "--nav-time-shift-sec",
        type=float,
        default=0.0,
        help="Additional time shift applied to replay_nav timestamps before alignment",
    )
    parser.add_argument(
        "--reference-time-shift-sec",
        type=float,
        default=0.0,
        help="Additional time shift applied to reference timestamps before alignment",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    nav = read_replay_nav(args.replay_nav_csv)
    ref = read_reference(args.reference_csv)
    t, nav_a, ref_i = align_interp(
        nav,
        ref,
        use_relative_time=args.time_mode == "relative",
        nav_time_shift_sec=args.nav_time_shift_sec,
        reference_time_shift_sec=args.reference_time_shift_sec,
    )
    nav_eval, ref_eval, position_source = use_common_geodetic_frame(nav_a, ref_i)

    err_pos = nav_eval[:, 1:4] - ref_eval[:, 1:4]
    err_vel = nav_eval[:, 4:7] - ref_eval[:, 4:7]
    metrics = compute_metrics(err_pos, err_vel)
    metrics["time_mode"] = args.time_mode
    metrics["nav_time_shift_sec"] = float(args.nav_time_shift_sec)
    metrics["reference_time_shift_sec"] = float(args.reference_time_shift_sec)
    metrics["position_source"] = position_source
    metrics["replay_nav_csv"] = str(args.replay_nav_csv.resolve())
    metrics["reference_csv"] = str(args.reference_csv.resolve())

    save_trajectory_plot(args.out_dir / "traj_vs_reference.svg", nav_eval, ref_eval)
    save_error_plot(args.out_dir / "position_error.svg", t, err_pos)
    save_velocity_error_plot(args.out_dir / "velocity_error.svg", t, err_vel)
    save_error_csv(args.out_dir / "position_error.csv", t, err_pos, err_vel)
    save_spectrum(
        args.out_dir / "position_error_psd.svg",
        args.out_dir / "position_error_psd.csv",
        t,
        {
            "east": err_pos[:, 0],
            "north": err_pos[:, 1],
            "up": err_pos[:, 2],
            "xy_norm": np.linalg.norm(err_pos[:, :2], axis=1),
        },
        "Replay Position Error Spectrum",
    )
    save_spectrum(
        args.out_dir / "velocity_error_psd.svg",
        args.out_dir / "velocity_error_psd.csv",
        t,
        {
            "vx": err_vel[:, 0],
            "vy": err_vel[:, 1],
            "vz": err_vel[:, 2],
            "vnorm": np.linalg.norm(err_vel, axis=1),
        },
        "Replay Velocity Error Spectrum",
    )

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'traj_vs_reference.svg'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'position_error.svg'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'velocity_error.svg'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'position_error_psd.svg'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'velocity_error_psd.svg'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'position_error.csv'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'position_error_psd.csv'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'velocity_error_psd.csv'}")
    print(f"[analyze_replay_nav_vs_reference] wrote {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
