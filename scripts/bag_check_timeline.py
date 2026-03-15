#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict

from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def main():
    parser = argparse.ArgumentParser(description="Check rosbag timeline consistency")
    parser.add_argument("--bag", required=True, help="rosbag2 directory")
    parser.add_argument("--imu-topic", default="/imu/data", help="IMU topic")
    parser.add_argument("--gnss-topic", default="/gps/fix", help="GNSS topic")
    parser.add_argument("--print-gaps", action="store_true", help="Print large gaps")
    parser.add_argument("--gap-threshold", type=float, default=0.2, help="Gap threshold (s)")
    args = parser.parse_args()

    storage = StorageOptions(uri=args.bag, storage_id="sqlite3")
    converter = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage, converter)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    for t in (args.imu_topic, args.gnss_topic):
        if t not in type_map:
            raise RuntimeError(f"Topic not found: {t}")

    imu_type = get_message(type_map[args.imu_topic])
    gnss_type = get_message(type_map[args.gnss_topic])

    stats = {
        args.imu_topic: {
            "count": 0,
            "first": None,
            "last": None,
            "min_dt": math.inf,
            "max_dt": -math.inf,
            "sum_dt": 0.0,
            "neg_dt": 0,
            "zero_dt": 0,
        },
        args.gnss_topic: {
            "count": 0,
            "first": None,
            "last": None,
            "min_dt": math.inf,
            "max_dt": -math.inf,
            "sum_dt": 0.0,
            "neg_dt": 0,
            "zero_dt": 0,
        },
    }
    last_time = {args.imu_topic: None, args.gnss_topic: None}
    gaps = defaultdict(list)

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in (args.imu_topic, args.gnss_topic):
            continue
        msg_type = imu_type if topic == args.imu_topic else gnss_type
        msg = deserialize_message(data, msg_type)
        t = to_sec(msg.header.stamp)

        s = stats[topic]
        if s["first"] is None:
            s["first"] = t
        s["last"] = t
        s["count"] += 1

        if last_time[topic] is not None:
            dt = t - last_time[topic]
            s["min_dt"] = min(s["min_dt"], dt)
            s["max_dt"] = max(s["max_dt"], dt)
            s["sum_dt"] += dt
            if dt < 0:
                s["neg_dt"] += 1
            if dt == 0:
                s["zero_dt"] += 1
            if args.print_gaps and dt > args.gap_threshold:
                gaps[topic].append((last_time[topic], t, dt))
        last_time[topic] = t

    print("Bag timeline check")
    for topic, s in stats.items():
        if s["count"] <= 1:
            print(f"- {topic}: count={s['count']} (insufficient data)")
            continue
        mean_dt = s["sum_dt"] / (s["count"] - 1)
        freq = 1.0 / mean_dt if mean_dt > 0 else float("inf")
        print(f"- {topic}")
        print(f"  count: {s['count']}")
        print(f"  time range: {s['first']:.6f} -> {s['last']:.6f} (len {s['last']-s['first']:.3f}s)")
        print(f"  dt min/mean/max: {s['min_dt']:.6f} / {mean_dt:.6f} / {s['max_dt']:.6f}")
        print(f"  approx rate: {freq:.2f} Hz")
        print(f"  negative dt: {s['neg_dt']}, zero dt: {s['zero_dt']}")

    if args.print_gaps:
        for topic, g in gaps.items():
            if not g:
                continue
            print(f"\nLarge gaps for {topic} (> {args.gap_threshold}s):")
            for a, b, dt in g[:20]:
                print(f"  {a:.6f} -> {b:.6f} (dt {dt:.6f})")
            if len(g) > 20:
                print(f"  ... {len(g)-20} more")


if __name__ == "__main__":
    main()
