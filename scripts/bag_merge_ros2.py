#!/usr/bin/env python3
"""
Merge multiple ROS2 bags into one ROS2 bag in chronological order.

This script copies serialized messages directly (no deserialization), so it also
works for custom message types that are not installed in the current ROS2 env.
"""

import argparse
import os

from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter, StorageOptions, TopicMetadata


def open_reader(uri: str) -> SequentialReader:
    r = SequentialReader()
    r.open(
        StorageOptions(uri=uri, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge ROS2 bags")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input ROS2 bag directories")
    ap.add_argument("--output", required=True, help="Output ROS2 bag directory")
    args = ap.parse_args()

    if os.path.exists(args.output):
        raise RuntimeError(f"Output already exists: {args.output}")

    readers = [open_reader(p) for p in args.inputs]
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=args.output, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    # Create union of topics.
    created = set()
    for r in readers:
        for t in r.get_all_topics_and_types():
            key = (t.name, t.type)
            if key in created:
                continue
            writer.create_topic(
                TopicMetadata(name=t.name, type=t.type, serialization_format="cdr")
            )
            created.add(key)

    # Prime each reader with one message.
    heads = []
    for r in readers:
        if r.has_next():
            heads.append(r.read_next())  # (topic, serialized, timestamp)
        else:
            heads.append(None)

    count = 0
    while True:
        best_i = None
        best_t = None
        for i, h in enumerate(heads):
            if h is None:
                continue
            t_ns = h[2]
            if best_i is None or t_ns < best_t:
                best_i = i
                best_t = t_ns
        if best_i is None:
            break

        topic, data, t_ns = heads[best_i]
        writer.write(topic, data, t_ns)
        count += 1
        if readers[best_i].has_next():
            heads[best_i] = readers[best_i].read_next()
        else:
            heads[best_i] = None

    print(f"Merged {len(args.inputs)} bags into {args.output}")
    print(f"topic_count={len(created)}, message_count={count}")


if __name__ == "__main__":
    main()

