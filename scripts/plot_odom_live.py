#!/usr/bin/env python3
"""
Subscribe to nav_msgs/Odometry and draw trajectory in real time.

Usage:
  python3 src/KF-GINS/scripts/plot_odom_live.py \
    --topic /kf_gins/odom_pred \
    --out-png src/KF-GINS/results/odom_live.png \
    --out-csv src/KF-GINS/results/odom_live.csv
"""

import argparse
import csv
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class OdomLivePlotter(Node):
    def __init__(self, topic: str, max_points: int, dedup_stamp: bool) -> None:
        super().__init__("odom_live_plotter")
        self.topic = topic
        self.max_points = max_points
        self.dedup_stamp = dedup_stamp

        self.t: List[float] = []
        self.x: List[float] = []
        self.y: List[float] = []
        self.z: List[float] = []
        self._last_stamp: Optional[float] = None

        self.create_subscription(Odometry, self.topic, self._cb, 100)
        self.get_logger().info(f"Subscribed topic: {self.topic}")

    def _cb(self, msg: Odometry) -> None:
        ts = float(msg.header.stamp.sec) + \
            float(msg.header.stamp.nanosec) * 1e-9
        if self.dedup_stamp and self._last_stamp is not None and ts <= self._last_stamp:
            return
        self._last_stamp = ts

        p = msg.pose.pose.position
        self.t.append(ts)
        self.x.append(float(p.x))
        self.y.append(float(p.y))
        self.z.append(float(p.z))

        if self.max_points > 0 and len(self.t) > self.max_points:
            overflow = len(self.t) - self.max_points
            del self.t[:overflow]
            del self.x[:overflow]
            del self.y[:overflow]
            del self.z[:overflow]

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "x_m", "y_m", "z_m"])
            for i in range(len(self.t)):
                w.writerow(
                    [
                        f"{self.t[i]:.9f}",
                        f"{self.x[i]:.6f}",
                        f"{self.y[i]:.6f}",
                        f"{self.z[i]:.6f}",
                    ]
                )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real-time trajectory plotter for nav_msgs/Odometry")
    ap.add_argument("--topic", default="/kf_gins/odom_pred",
                    help="Odometry topic")
    ap.add_argument("--out-png", default="odom_live_traj.png",
                    help="Output PNG path (saved on exit)")
    ap.add_argument("--out-csv", default="",
                    help="Optional CSV path (saved on exit)")
    ap.add_argument("--max-points", type=int, default=-1,
                    help="Keep latest N points, <=0 means unlimited")
    ap.add_argument("--plot-rate", type=float, default=10.0,
                    help="Plot refresh rate in Hz")
    ap.add_argument("--spin-timeout", type=float, default=0.05,
                    help="spin_once timeout in seconds")
    ap.add_argument(
        "--no-dedup-stamp",
        action="store_true",
        help="Disable monotonic-stamp deduplication",
    )
    args = ap.parse_args()

    rclpy.init()
    node = OdomLivePlotter(
        topic=args.topic,
        max_points=args.max_points,
        dedup_stamp=not args.no_dedup_stamp,
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    line, = ax.plot([], [], linewidth=1.0, label=args.topic)
    start_pt = ax.scatter([], [], c="green", s=35,
                          marker="o", label="start", zorder=3)
    end_pt = ax.scatter([], [], c="red", s=35,
                        marker="x", label="end", zorder=3)
    ax.set_title("Live Odom Trajectory (XY)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend(loc="best")

    last_draw = 0.0
    draw_period = 1.0 / max(args.plot_rate, 0.1)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=args.spin_timeout)
            now = time.monotonic()
            if now - last_draw < draw_period:
                continue
            last_draw = now

            if not node.x:
                plt.pause(0.001)
                continue

            line.set_data(node.x, node.y)
            start_pt.set_offsets([[node.x[0], node.y[0]]])
            end_pt.set_offsets([[node.x[-1], node.y[-1]]])
            ax.relim()
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="box")
            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if node.x:
            line.set_data(node.x, node.y)
            start_pt.set_offsets([[node.x[0], node.y[0]]])
            end_pt.set_offsets([[node.x[-1], node.y[-1]]])
            ax.relim()
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="box")
            fig.savefig(args.out_png, dpi=600)
            node.get_logger().info(f"Saved trajectory figure: {args.out_png}")
        else:
            node.get_logger().warning("No odom samples received; PNG not saved.")

        if args.out_csv:
            node.save_csv(args.out_csv)
            node.get_logger().info(f"Saved trajectory CSV: {args.out_csv}")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
