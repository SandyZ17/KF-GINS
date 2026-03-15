#!/usr/bin/env python3
"""
Merge ROS2 sqlite3 bags by reading sqlite tables directly and writing a new rosbag2 bag.

Why this exists:
- Some converted bags (e.g. rosbags-generated metadata) may not be readable by rosbag2_py
  due to metadata.yaml format differences, while the underlying sqlite DB is valid.
"""

from __future__ import annotations

import argparse
import heapq
import sqlite3
from pathlib import Path

import yaml
from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata


def first_db_path(bag_dir: Path) -> Path:
    meta = yaml.safe_load((bag_dir / "metadata.yaml").read_text(encoding="utf-8"))
    rel = meta["rosbag2_bagfile_information"]["relative_file_paths"][0]
    return bag_dir / rel


class BagSqlReader:
    def __init__(self, bag_dir: Path):
        self.bag_dir = bag_dir
        self.db_path = first_db_path(bag_dir)
        self.con = sqlite3.connect(str(self.db_path))
        self.con.row_factory = sqlite3.Row
        self.topic_rows = self._load_topics()
        self.topic_by_id = {int(r["id"]): r for r in self.topic_rows}
        self.cur = self.con.cursor()
        self.cur.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp ASC, id ASC")
        self._next = self.cur.fetchone()

    def _load_topics(self):
        cur = self.con.cursor()
        cur.execute("PRAGMA table_info(topics)")
        cols = [r[1] for r in cur.fetchall()]
        select_cols = ["id", "name", "type", "serialization_format"]
        if "offered_qos_profiles" in cols:
            select_cols.append("offered_qos_profiles")
        if "type_description_hash" in cols:
            select_cols.append("type_description_hash")
        cur.execute(f"SELECT {', '.join(select_cols)} FROM topics ORDER BY id ASC")
        return cur.fetchall()

    def peek(self):
        return self._next

    def pop(self):
        row = self._next
        if row is None:
            return None
        self._next = self.cur.fetchone()
        return row

    def close(self):
        self.con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge ROS2 sqlite bags via sqlite direct read")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input rosbag2 directories")
    ap.add_argument("--output", required=True, help="Output rosbag2 directory")
    args = ap.parse_args()

    outdir = Path(args.output)
    if outdir.exists():
        raise RuntimeError(f"Output exists: {outdir}")

    readers = [BagSqlReader(Path(p)) for p in args.inputs]
    try:
        writer = SequentialWriter()
        writer.open(
            StorageOptions(uri=str(outdir), storage_id="sqlite3"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

        # Create union of topics and keep id remap per reader.
        created = {}
        topic_id_map = []
        next_fake_id = 1
        for r in readers:
            local_map = {}
            for tr in r.topic_rows:
                name = tr["name"]
                typ = tr["type"]
                sformat = tr["serialization_format"]
                key = (name, typ, sformat)
                if key not in created:
                    writer.create_topic(
                        TopicMetadata(
                            name=name,
                            type=typ,
                            serialization_format=sformat,
                            offered_qos_profiles=str(tr["offered_qos_profiles"]) if "offered_qos_profiles" in tr.keys() else "",
                        )
                    )
                    created[key] = next_fake_id
                    next_fake_id += 1
                local_map[int(tr["id"])] = key
            topic_id_map.append(local_map)

        heap = []
        for i, r in enumerate(readers):
            row = r.peek()
            if row is not None:
                heapq.heappush(heap, (int(row["timestamp"]), i))

        count = 0
        while heap:
            _, i = heapq.heappop(heap)
            row = readers[i].pop()
            if row is None:
                continue
            topic_id = int(row["topic_id"])
            key = topic_id_map[i][topic_id]
            name = key[0]
            writer.write(name, bytes(row["data"]), int(row["timestamp"]))
            count += 1
            nxt = readers[i].peek()
            if nxt is not None:
                heapq.heappush(heap, (int(nxt["timestamp"]), i))

        print(f"Merged {len(readers)} bags into {outdir}")
        print(f"topic_count={len(created)}, message_count={count}")
    finally:
        for r in readers:
            r.close()


if __name__ == "__main__":
    main()
