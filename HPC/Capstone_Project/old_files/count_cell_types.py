#!/usr/bin/env python3
"""Count CellViT cell types in one or more cells.json files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count how many cells belong to each type in cells.json."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="cellvit_output/raw_cellvit",
        help="Path to cells.json or a directory to scan recursively.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Print CSV format: file,type_id,type_name,count.",
    )
    return parser.parse_args()


def find_cells_json(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("cells.json"))
    raise FileNotFoundError(f"Path does not exist: {path}")


def load_type_map(payload: dict) -> dict[int, str]:
    type_map: dict[int, str] = {}
    for key, value in payload.get("type_map", {}).items():
        try:
            type_map[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return type_map


def count_cell_types(cells_json: Path) -> tuple[Counter[int], dict[int, str]]:
    with cells_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    counts: Counter[int] = Counter()
    for cell in payload.get("cells", []):
        try:
            type_id = int(cell.get("type", 0))
        except (TypeError, ValueError):
            type_id = 0
        counts[type_id] += 1

    return counts, load_type_map(payload)


def print_table(
    cells_json: Path,
    counts: Counter[int],
    type_map: dict[int, str],
    csv: bool,
) -> None:
    if csv:
        for type_id, count in sorted(counts.items()):
            print(f"{cells_json},{type_id},{type_map.get(type_id, f'type_{type_id}')},{count}")
        return

    print(f"\n{cells_json}")
    print("-" * len(str(cells_json)))
    print(f"{'type_id':>7}  {'type_name':<20}  {'count':>10}")
    for type_id, count in sorted(counts.items()):
        type_name = type_map.get(type_id, f"type_{type_id}")
        print(f"{type_id:>7}  {type_name:<20}  {count:>10,}")
    print(f"{'total':>7}  {'':<20}  {sum(counts.values()):>10,}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.path)
    cells_jsons = find_cells_json(input_path)

    if not cells_jsons:
        raise FileNotFoundError(f"No cells.json found under: {input_path}")

    if args.csv:
        print("file,type_id,type_name,count")

    total_counts: Counter[int] = Counter()
    merged_type_map: dict[int, str] = {}

    for cells_json in cells_jsons:
        counts, type_map = count_cell_types(cells_json)
        total_counts.update(counts)
        merged_type_map.update(type_map)
        print_table(cells_json, counts, type_map, args.csv)

    if len(cells_jsons) > 1:
        print_table(Path("TOTAL"), total_counts, merged_type_map, args.csv)


if __name__ == "__main__":
    main()
