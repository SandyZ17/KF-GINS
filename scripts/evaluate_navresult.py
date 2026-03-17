#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import site
import warnings
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
warnings.filterwarnings(
    'ignore',
    message='Unable to import Axes3D.*',
    category=UserWarning,
)

import mpl_toolkits

USER_MPL_TOOLKITS = Path(site.getusersitepackages()) / 'mpl_toolkits'
if USER_MPL_TOOLKITS.exists() and hasattr(mpl_toolkits, '__path__'):
    user_mpl_path = str(USER_MPL_TOOLKITS)
    if user_mpl_path not in mpl_toolkits.__path__:
        mpl_toolkits.__path__.insert(0, user_mpl_path)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from plot_navresult import D2R, calcNavresultError, drad2dm, radiusmn


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate KF-GINS navigation result against reference trajectory.'
    )
    parser.add_argument('navresult', type=Path, help='Path to KF_GINS_Navresult.nav')
    parser.add_argument('reference', type=Path, help='Path to reference nav file')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots and metrics. Defaults to <navresult_dir>/evaluation',
    )
    return parser.parse_args()


def compute_metrics(naverror):
    pos_err = naverror[:, 2:5]
    vel_err = naverror[:, 5:8]
    att_err = naverror[:, 8:11]

    horizontal_err = np.linalg.norm(pos_err[:, :2], axis=1)
    position_3d_err = np.linalg.norm(pos_err, axis=1)

    metrics = {
        'sample_count': int(naverror.shape[0]),
        'time_start_s': float(naverror[0, 1]),
        'time_end_s': float(naverror[-1, 1]),
        'position_rmse_m': np.sqrt(np.mean(pos_err ** 2, axis=0)).tolist(),
        'velocity_rmse_mps': np.sqrt(np.mean(vel_err ** 2, axis=0)).tolist(),
        'attitude_rmse_deg': np.sqrt(np.mean(att_err ** 2, axis=0)).tolist(),
        'horizontal_rmse_m': float(np.sqrt(np.mean(horizontal_err ** 2))),
        'vertical_rmse_m': float(np.sqrt(np.mean(pos_err[:, 2] ** 2))),
        'position_3d_rmse_m': float(np.sqrt(np.mean(position_3d_err ** 2))),
        'max_horizontal_error_m': float(np.max(horizontal_err)),
        'max_vertical_error_m': float(np.max(np.abs(pos_err[:, 2]))),
        'max_position_3d_error_m': float(np.max(position_3d_err)),
        'mean_horizontal_error_m': float(np.mean(horizontal_err)),
        'mean_position_3d_error_m': float(np.mean(position_3d_err)),
    }
    return metrics


def save_metrics(metrics, output_dir):
    json_path = output_dir / 'metrics.json'
    txt_path = output_dir / 'metrics.txt'

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + '\n')

    lines = [
        f"sample_count: {metrics['sample_count']}",
        f"time_span_s: {metrics['time_start_s']:.3f} -> {metrics['time_end_s']:.3f}",
        (
            'position_rmse_m [N,E,D]: '
            f"{metrics['position_rmse_m'][0]:.6f}, "
            f"{metrics['position_rmse_m'][1]:.6f}, "
            f"{metrics['position_rmse_m'][2]:.6f}"
        ),
        (
            'velocity_rmse_mps [N,E,D]: '
            f"{metrics['velocity_rmse_mps'][0]:.6f}, "
            f"{metrics['velocity_rmse_mps'][1]:.6f}, "
            f"{metrics['velocity_rmse_mps'][2]:.6f}"
        ),
        (
            'attitude_rmse_deg [R,P,Y]: '
            f"{metrics['attitude_rmse_deg'][0]:.6f}, "
            f"{metrics['attitude_rmse_deg'][1]:.6f}, "
            f"{metrics['attitude_rmse_deg'][2]:.6f}"
        ),
        f"horizontal_rmse_m: {metrics['horizontal_rmse_m']:.6f}",
        f"vertical_rmse_m: {metrics['vertical_rmse_m']:.6f}",
        f"position_3d_rmse_m: {metrics['position_3d_rmse_m']:.6f}",
        f"max_horizontal_error_m: {metrics['max_horizontal_error_m']:.6f}",
        f"max_vertical_error_m: {metrics['max_vertical_error_m']:.6f}",
        f"max_position_3d_error_m: {metrics['max_position_3d_error_m']:.6f}",
        f"mean_horizontal_error_m: {metrics['mean_horizontal_error_m']:.6f}",
        f"mean_position_3d_error_m: {metrics['mean_position_3d_error_m']:.6f}",
    ]
    txt_path.write_text('\n'.join(lines) + '\n')


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def annotate_trajectory(ax, local_xy, label_prefix):
    east = local_xy[:, 1]
    north = local_xy[:, 0]

    ax.scatter(east[0], north[0], marker='o', s=36, label=f'{label_prefix} Start')
    ax.scatter(east[-1], north[-1], marker='s', s=36, label=f'{label_prefix} End')


def annotate_trajectory_3d(ax, local_xyz, label_prefix):
    east = local_xyz[:, 1]
    north = local_xyz[:, 0]
    up = -local_xyz[:, 2]

    ax.scatter(east[0], north[0], up[0], marker='o', s=36, label=f'{label_prefix} Start')
    ax.scatter(east[-1], north[-1], up[-1], marker='s', s=36, label=f'{label_prefix} End')


def load_overlapping_trajectories(navresult_path, reference_path):
    navresult = np.loadtxt(navresult_path)
    reference = np.loadtxt(reference_path)

    start_time = max(navresult[0, 1], reference[0, 1])
    end_time = min(navresult[-1, 1], reference[-1, 1])

    nav_mask = (navresult[:, 1] >= start_time) & (navresult[:, 1] <= end_time)
    ref_mask = (reference[:, 1] >= start_time) & (reference[:, 1] <= end_time)

    nav_overlap = navresult[nav_mask].copy()
    ref_overlap = reference[ref_mask].copy()

    nav_overlap[:, 2:4] *= D2R
    ref_overlap[:, 2:4] *= D2R

    return nav_overlap, ref_overlap


def convert_to_local_ned(nav):
    local = np.zeros((nav.shape[0], 3))
    origin = nav[0, 2:5]
    rm, rn = radiusmn(origin[0])

    for index in range(nav.shape[0]):
        delta_blh = nav[index, 2:5] - origin
        local[index, :] = drad2dm(rm, rn, origin, delta_blh).reshape(3)

    return local


def plot_trajectory_overlay(navresult_path, reference_path, output_dir):
    nav_overlap, ref_overlap = load_overlapping_trajectories(navresult_path, reference_path)
    nav_local = convert_to_local_ned(nav_overlap)
    ref_local = convert_to_local_ned(ref_overlap)

    fig, ax = plt.subplots()
    ax.plot(ref_local[:, 1], ref_local[:, 0], label='Reference', linewidth=1.5, zorder=2)
    ax.plot(
        nav_local[:, 1],
        nav_local[:, 0],
        label='KF-GINS',
        linewidth=1.4,
        linestyle='--',
        zorder=4,
    )
    annotate_trajectory(ax, ref_local, 'Reference')
    annotate_trajectory(ax, nav_local, 'KF-GINS')
    ax.axis('equal')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title('Trajectory Overlay')
    ax.grid()
    ax.legend()
    save_figure(fig, output_dir / 'trajectory_overlay.svg')


def plot_trajectory_overlay_3d(navresult_path, reference_path, output_dir):
    nav_overlap, ref_overlap = load_overlapping_trajectories(navresult_path, reference_path)
    nav_local = convert_to_local_ned(nav_overlap)
    ref_local = convert_to_local_ned(ref_overlap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        ref_local[:, 1],
        ref_local[:, 0],
        -ref_local[:, 2],
        label='Reference',
        linewidth=1.5,
        zorder=2,
    )
    ax.plot(
        nav_local[:, 1],
        nav_local[:, 0],
        -nav_local[:, 2],
        label='KF-GINS',
        linewidth=1.4,
        linestyle='--',
        zorder=4,
    )
    annotate_trajectory_3d(ax, ref_local, 'Reference')
    annotate_trajectory_3d(ax, nav_local, 'KF-GINS')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Trajectory Overlay 3D')
    ax.legend()
    save_figure(fig, output_dir / 'trajectory_overlay_3d.svg')


def plot_position_error(naverror, output_dir):
    fig, ax = plt.subplots()
    ax.plot(naverror[:, 1], naverror[:, 2:5])
    ax.legend(['North', 'East', 'Down'])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m]')
    ax.set_title('Position Error')
    ax.grid()
    save_figure(fig, output_dir / 'position_error.svg')


def plot_velocity_error(naverror, output_dir):
    fig, ax = plt.subplots()
    ax.plot(naverror[:, 1], naverror[:, 5:8])
    ax.legend(['North', 'East', 'Down'])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m/s]')
    ax.set_title('Velocity Error')
    ax.grid()
    save_figure(fig, output_dir / 'velocity_error.svg')


def plot_attitude_error(naverror, output_dir):
    fig, ax = plt.subplots()
    ax.plot(naverror[:, 1], naverror[:, 8:11])
    ax.legend(['Roll', 'Pitch', 'Yaw'])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [deg]')
    ax.set_title('Attitude Error')
    ax.grid()
    save_figure(fig, output_dir / 'attitude_error.svg')


def plot_horizontal_error(naverror, output_dir):
    horizontal_err = np.linalg.norm(naverror[:, 2:4], axis=1)
    fig, ax = plt.subplots()
    ax.plot(naverror[:, 1], horizontal_err)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m]')
    ax.set_title('Horizontal Position Error')
    ax.grid()
    save_figure(fig, output_dir / 'horizontal_error.svg')


def plot_vertical_error(naverror, output_dir):
    fig, ax = plt.subplots()
    ax.plot(naverror[:, 1], naverror[:, 4])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Error [m]')
    ax.set_title('Vertical Position Error')
    ax.grid()
    save_figure(fig, output_dir / 'vertical_error.svg')


def print_metrics(metrics):
    print(f"sample_count: {metrics['sample_count']}")
    print(f"time_span_s: {metrics['time_start_s']:.3f} -> {metrics['time_end_s']:.3f}")
    print(
        'position_rmse_m [N,E,D]: '
        f"{metrics['position_rmse_m'][0]:.6f}, "
        f"{metrics['position_rmse_m'][1]:.6f}, "
        f"{metrics['position_rmse_m'][2]:.6f}"
    )
    print(
        'velocity_rmse_mps [N,E,D]: '
        f"{metrics['velocity_rmse_mps'][0]:.6f}, "
        f"{metrics['velocity_rmse_mps'][1]:.6f}, "
        f"{metrics['velocity_rmse_mps'][2]:.6f}"
    )
    print(
        'attitude_rmse_deg [R,P,Y]: '
        f"{metrics['attitude_rmse_deg'][0]:.6f}, "
        f"{metrics['attitude_rmse_deg'][1]:.6f}, "
        f"{metrics['attitude_rmse_deg'][2]:.6f}"
    )
    print(f"horizontal_rmse_m: {metrics['horizontal_rmse_m']:.6f}")
    print(f"vertical_rmse_m: {metrics['vertical_rmse_m']:.6f}")
    print(f"position_3d_rmse_m: {metrics['position_3d_rmse_m']:.6f}")
    print(f"max_horizontal_error_m: {metrics['max_horizontal_error_m']:.6f}")
    print(f"max_vertical_error_m: {metrics['max_vertical_error_m']:.6f}")
    print(f"max_position_3d_error_m: {metrics['max_position_3d_error_m']:.6f}")


def main():
    args = parse_args()
    navresult = args.navresult.resolve()
    reference = args.reference.resolve()
    output_dir = (args.output_dir or (navresult.parent / 'evaluation')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    naverror = calcNavresultError(str(navresult), str(reference))
    metrics = compute_metrics(naverror)

    plot_trajectory_overlay(navresult, reference, output_dir)
    plot_trajectory_overlay_3d(navresult, reference, output_dir)
    plot_position_error(naverror, output_dir)
    plot_velocity_error(naverror, output_dir)
    plot_attitude_error(naverror, output_dir)
    plot_horizontal_error(naverror, output_dir)
    plot_vertical_error(naverror, output_dir)
    save_metrics(metrics, output_dir)
    print_metrics(metrics)

    print(f'saved plots and metrics to: {output_dir}')


if __name__ == '__main__':
    main()
