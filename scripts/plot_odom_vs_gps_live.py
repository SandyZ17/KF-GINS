#!/usr/bin/env python3
"""
Live compare KF-GINS odom trajectory against GPS trajectory.

Subscribes:
- nav_msgs/Odometry (default: /kf_gins/odom_fused)
- sensor_msgs/NavSatFix (default: /kf_gins/gps/fix)

Shows real-time XY overlay (odom vs gps-local-ENU), and reports matched
position error statistics online.
"""

import argparse
import csv
import math
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


D2R = math.pi / 180.0
WGS84_RA = 6378137.0
WGS84_E1 = 0.00669437999013


def radiusmn(lat_rad: float) -> Tuple[float, float]:
    s2 = math.sin(lat_rad) ** 2
    t = 1.0 - WGS84_E1 * s2
    st = math.sqrt(t)
    rm = WGS84_RA * (1.0 - WGS84_E1) / (st * t)
    rn = WGS84_RA / st
    return rm, rn


class OdomGpsLiveCompare(Node):
    def __init__(
        self,
        odom_topic: str,
        gps_topic: str,
        max_points: int,
        dedup_stamp: bool,
    ) -> None:
        super().__init__("odom_gps_live_compare")
        self.odom_topic = odom_topic
        self.gps_topic = gps_topic
        self.max_points = max_points
        self.dedup_stamp = dedup_stamp

        self.odom_t: List[float] = []
        self.odom_x: List[float] = []
        self.odom_y: List[float] = []
        self.odom_z: List[float] = []

        self.gps_t: List[float] = []
        self.gps_x: List[float] = []
        self.gps_y: List[float] = []
        self.gps_z: List[float] = []

        self._last_odom_stamp: Optional[float] = None
        self._last_gps_stamp: Optional[float] = None

        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._h0: Optional[float] = None
        self._rm0: Optional[float] = None
        self._rn0: Optional[float] = None

        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 100)
        self.create_subscription(NavSatFix, self.gps_topic, self._gps_cb, 100)
        self.get_logger().info(f"Subscribed odom: {self.odom_topic}")
        self.get_logger().info(f"Subscribed gps : {self.gps_topic}")

    def _odom_cb(self, msg: Odometry) -> None:
        ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self.dedup_stamp and self._last_odom_stamp is not None and ts <= self._last_odom_stamp:
            return
        self._last_odom_stamp = ts
        p = msg.pose.pose.position
        self.odom_t.append(ts)
        self.odom_x.append(float(p.x))
        self.odom_y.append(float(p.y))
        self.odom_z.append(float(p.z))
        self._trim(self.odom_t, self.odom_x, self.odom_y, self.odom_z)

    def _gps_cb(self, msg: NavSatFix) -> None:
        ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self.dedup_stamp and self._last_gps_stamp is not None and ts <= self._last_gps_stamp:
            return
        self._last_gps_stamp = ts

        lat = float(msg.latitude) * D2R
        lon = float(msg.longitude) * D2R
        h = float(msg.altitude)

        if self._lat0 is None:
            self._lat0 = lat
            self._lon0 = lon
            self._h0 = h
            self._rm0, self._rn0 = radiusmn(self._lat0)

        north = (lat - self._lat0) * (self._rm0 + self._h0)
        east = (lon - self._lon0) * (self._rn0 + self._h0) * math.cos(self._lat0)
        up = h - self._h0

        self.gps_t.append(ts)
        self.gps_x.append(east)
        self.gps_y.append(north)
        self.gps_z.append(up)
        self._trim(self.gps_t, self.gps_x, self.gps_y, self.gps_z)

    def _trim(self, t: List[float], x: List[float], y: List[float], z: List[float]) -> None:
        if self.max_points <= 0:
            return
        if len(t) <= self.max_points:
            return
        overflow = len(t) - self.max_points
        del t[:overflow]
        del x[:overflow]
        del y[:overflow]
        del z[:overflow]

    def compute_error_stats(self, max_dt: float) -> Optional[Tuple[int, float, float, float]]:
        if len(self.odom_t) < 2 or len(self.gps_t) < 2:
            return None

        to = np.array(self.odom_t, dtype=float)
        ox = np.array(self.odom_x, dtype=float)
        oy = np.array(self.odom_y, dtype=float)
        tg = np.array(self.gps_t, dtype=float)
        gx = np.array(self.gps_x, dtype=float)
        gy = np.array(self.gps_y, dtype=float)

        idx = np.searchsorted(tg, to, side="left")
        idx = np.clip(idx, 0, len(tg) - 1)
        idx_prev = np.maximum(idx - 1, 0)
        use_prev = np.abs(tg[idx_prev] - to) < np.abs(tg[idx] - to)
        idx[use_prev] = idx_prev[use_prev]

        dt = np.abs(tg[idx] - to)
        keep = dt <= max_dt
        if not np.any(keep):
            return None

        ep = np.sqrt((ox[keep] - gx[idx[keep]]) ** 2 + (oy[keep] - gy[idx[keep]]) ** 2)
        rmse = float(np.sqrt(np.mean(ep**2)))
        p95 = float(np.percentile(ep, 95))
        maxe = float(np.max(ep))
        return int(np.sum(keep)), rmse, p95, maxe

    def save_csv(self, out_path: str, max_dt: float) -> None:
        to = np.array(self.odom_t, dtype=float)
        ox = np.array(self.odom_x, dtype=float)
        oy = np.array(self.odom_y, dtype=float)
        oz = np.array(self.odom_z, dtype=float)
        tg = np.array(self.gps_t, dtype=float)
        gx = np.array(self.gps_x, dtype=float)
        gy = np.array(self.gps_y, dtype=float)
        gz = np.array(self.gps_z, dtype=float)

        if len(to) == 0 or len(tg) == 0:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "t_odom",
                        "x_odom",
                        "y_odom",
                        "z_odom",
                        "t_gps",
                        "x_gps",
                        "y_gps",
                        "z_gps",
                        "err_xy",
                        "matched",
                    ]
                )
            return

        idx = np.searchsorted(tg, to, side="left")
        idx = np.clip(idx, 0, len(tg) - 1)
        idx_prev = np.maximum(idx - 1, 0)
        use_prev = np.abs(tg[idx_prev] - to) < np.abs(tg[idx] - to)
        idx[use_prev] = idx_prev[use_prev]
        dt = np.abs(tg[idx] - to)
        keep = dt <= max_dt

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "t_odom",
                    "x_odom",
                    "y_odom",
                    "z_odom",
                    "t_gps",
                    "x_gps",
                    "y_gps",
                    "z_gps",
                    "err_xy",
                    "matched",
                ]
            )
            for i in range(len(to)):
                j = idx[i]
                err = math.hypot(ox[i] - gx[j], oy[i] - gy[j])
                w.writerow(
                    [
                        f"{to[i]:.9f}",
                        f"{ox[i]:.6f}",
                        f"{oy[i]:.6f}",
                        f"{oz[i]:.6f}",
                        f"{tg[j]:.9f}",
                        f"{gx[j]:.6f}",
                        f"{gy[j]:.6f}",
                        f"{gz[j]:.6f}",
                        f"{err:.6f}",
                        int(bool(keep[i])),
                    ]
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Live XY compare: odom vs gps")
    ap.add_argument("--odom-topic", default="/kf_gins/odom_fused")
    ap.add_argument("--gps-topic", default="/kf_gins/gps/fix")
    ap.add_argument("--out-png", default="odom_vs_gps_live.png")
    ap.add_argument("--out-csv", default="", help="Optional matched CSV output path on exit")
    ap.add_argument("--max-points", type=int, default=30000)
    ap.add_argument("--plot-rate", type=float, default=10.0)
    ap.add_argument("--spin-timeout", type=float, default=0.05)
    ap.add_argument("--max-match-dt", type=float, default=0.2)
    ap.add_argument("--no-dedup-stamp", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = OdomGpsLiveCompare(
        odom_topic=args.odom_topic,
        gps_topic=args.gps_topic,
        max_points=args.max_points,
        dedup_stamp=not args.no_dedup_stamp,
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))
    line_o, = ax.plot([], [], linewidth=1.2, label=f"odom: {args.odom_topic}")
    line_g, = ax.plot([], [], linewidth=1.0, label=f"gps : {args.gps_topic}")
    o_start = ax.scatter([], [], c="green", s=35, marker="o", label="odom start", zorder=3)
    o_end = ax.scatter([], [], c="red", s=35, marker="x", label="odom end", zorder=3)
    ax.set_title("Live Trajectory Compare (XY): Odom vs GPS")
    ax.set_xlabel("X/East [m]")
    ax.set_ylabel("Y/North [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="best")

    draw_period = 1.0 / max(args.plot_rate, 0.1)
    last_draw = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=args.spin_timeout)
            now = time.monotonic()
            if now - last_draw < draw_period:
                continue
            last_draw = now

            if node.odom_x:
                line_o.set_data(node.odom_x, node.odom_y)
                o_start.set_offsets([[node.odom_x[0], node.odom_y[0]]])
                o_end.set_offsets([[node.odom_x[-1], node.odom_y[-1]]])
            if node.gps_x:
                line_g.set_data(node.gps_x, node.gps_y)

            stats = node.compute_error_stats(args.max_match_dt)
            if stats is None:
                ax.set_title("Live Trajectory Compare (XY): Odom vs GPS")
            else:
                n, rmse, p95, maxe = stats
                ax.set_title(
                    "Live Trajectory Compare (XY): Odom vs GPS | "
                    f"matched={n}, rmse={rmse:.3f}m, p95={p95:.3f}m, max={maxe:.3f}m"
                )

            ax.relim()
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="box")
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if node.odom_x or node.gps_x:
            if node.odom_x:
                line_o.set_data(node.odom_x, node.odom_y)
                o_start.set_offsets([[node.odom_x[0], node.odom_y[0]]])
                o_end.set_offsets([[node.odom_x[-1], node.odom_y[-1]]])
            if node.gps_x:
                line_g.set_data(node.gps_x, node.gps_y)
            ax.relim()
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="box")
            fig.savefig(args.out_png, dpi=600)
            node.get_logger().info(f"Saved figure: {args.out_png}")
        else:
            node.get_logger().warning("No samples received; figure not saved.")

        if args.out_csv:
            node.save_csv(args.out_csv, args.max_match_dt)
            node.get_logger().info(f"Saved CSV: {args.out_csv}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
