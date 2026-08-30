#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Color CellViT cell masks by cell_detection cell type on stitched ROI images.

The script uses CellViT cells.json contours, cell_detection.json types, and the
polyepi_L{level} ROI metadata to redraw each ROI-level mask with one color per
cell type.
It writes:
  * type_mask_<slide>_<region>.tif    RGB mask on black background
  * type_overlay_<slide>_<region>.tif colored mask blended on stitched image,
    with a visual color/type legend
  * type_overlay_no_legend_<slide>_<region>.tif same overlay without legend
  * cell_type_color_legend.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np

cv2 = None
imread = None
imsave = None


MASK_LEVEL = 2
POLY_DIR = f"./polyepi_L{MASK_LEVEL}"
CELLVIT_ROOT = "./cellvit_output"

POLY_RE = re.compile(
    r"^polyepi_(?P<slide>.+?)_(?P<region>.+?)_L(?P<lvl>\d+)"
    r"_(?P<minx>\d+)_(?P<miny>\d+)_(?P<w>\d+)x(?P<h>\d+)\.npy$"
)

DEFAULT_PALETTE = [
    (0, 255, 0),      # bright green
    (0, 0, 255),      # bright blue
    (0, 0, 10),        # black
    (255, 0, 0),      # red
    (255, 255, 0),    # bright yellow
    (255, 0, 255),    # magenta
    (255, 128, 0),    # bright orange
    (80, 255, 160),   # mint
    (80, 220, 255),   # sky blue
    (255, 220, 80),   # gold
    (180, 255, 80),   # chartreuse
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw stitched CellViT masks with colors based on cell type."
    )
    parser.add_argument(
        "--cell-json",
        default=None,
        help="Path to one cells.json/cell.json. If omitted, scan raw CellViT output.",
    )
    parser.add_argument(
        "--detection-json",
        default=None,
        help=(
            "Path to one cell_detection.json. If omitted, use the file next to "
            "each cells.json when available."
        ),
    )
    parser.add_argument("--cellvit-root", default=CELLVIT_ROOT)
    parser.add_argument("--poly-dir", default=POLY_DIR)
    parser.add_argument("--mask-level", type=int, default=MASK_LEVEL)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay opacity.")
    parser.add_argument(
        "--background",
        choices=["stitched", "black"],
        default="stitched",
        help="Background for overlay image when stitched image is available.",
    )
    parser.add_argument(
        "--legend",
        choices=["right", "bottom", "none"],
        default="right",
        help="Add a visual color/type legend to each overlay.",
    )
    parser.add_argument(
        "--legend-scope",
        choices=["present", "all"],
        default="present",
        help="Show only types present in the ROI or all known type colors.",
    )
    return parser.parse_args()


def load_cellvit_payload(json_path: Path) -> tuple[list[dict], dict[int, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_type_map = payload.get("type_map", {})
    type_map = {}
    for key, value in raw_type_map.items():
        try:
            type_map[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    return payload.get("cells", []), type_map


def centroid_key(cell: dict) -> tuple[int, int] | None:
    centroid = cell_centroid_xy(cell)
    if centroid is None:
        return None
    return int(round(centroid[0])), int(round(centroid[1]))


def merge_cell_detection_types(
    cells: list[dict],
    detections: list[dict],
) -> tuple[list[dict], int, str]:
    """
    Return cells with type/type_prob copied from cell_detection.json.

    CellViT's cell_detection.json generally contains the classification results
    but not contours. cells.json contains the contours needed for drawing. In
    normal output both cell lists have the same order; centroid matching is kept
    as a fallback for regenerated or filtered files.
    """
    if not detections:
        return cells, 0, "none"

    merged = [cell.copy() for cell in cells]
    matched = 0

    if len(cells) == len(detections):
        for out_cell, det_cell in zip(merged, detections):
            out_cell["type"] = det_cell.get("type", out_cell.get("type", 0))
            if "type_prob" in det_cell:
                out_cell["type_prob"] = det_cell["type_prob"]
            matched += 1
        return merged, matched, "order"

    detections_by_centroid: dict[tuple[int, int], dict] = {}
    for det_cell in detections:
        key = centroid_key(det_cell)
        if key is not None:
            detections_by_centroid[key] = det_cell

    for out_cell in merged:
        key = centroid_key(out_cell)
        det_cell = detections_by_centroid.get(key) if key is not None else None
        if det_cell is None:
            continue
        out_cell["type"] = det_cell.get("type", out_cell.get("type", 0))
        if "type_prob" in det_cell:
            out_cell["type_prob"] = det_cell["type_prob"]
        matched += 1

    return merged, matched, "centroid"


def find_cell_jsons(cellvit_root: Path) -> list[Path]:
    raw_dir = cellvit_root / "raw_cellvit"
    patterns = [
        str(raw_dir / "*" / "cells.json"),
        str(raw_dir / "*" / "cell.json"),
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    return sorted(paths)


def detection_json_for_cells_json(cells_json: Path) -> Path | None:
    detection_json = cells_json.with_name("cell_detection.json")
    if detection_json.exists():
        return detection_json
    return None


def slide_from_cells_json(cells_json: Path) -> str:
    return cells_json.parent.name


def load_regions_for_slide(slide_base: str, poly_dir: Path, mask_level: int) -> list[dict]:
    regions = []
    for npy_path in sorted(poly_dir.glob(f"polyepi_{slide_base}_*.npy")):
        match = POLY_RE.match(npy_path.name)
        if not match or int(match.group("lvl")) != mask_level:
            continue

        scale = 2**mask_level
        minx0 = int(match.group("minx")) * scale
        miny0 = int(match.group("miny")) * scale

        mask_l = np.load(npy_path)
        if mask_l.ndim != 2:
            mask_l = mask_l.squeeze()
        mask_l = (mask_l > 0).astype(np.uint8)
        h0 = int(mask_l.shape[0] * scale)
        w0 = int(mask_l.shape[1] * scale)
        mask0 = cv2.resize(mask_l, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0

        regions.append(
            {
                "region": match.group("region"),
                "minx0": minx0,
                "miny0": miny0,
                "w0": w0,
                "h0": h0,
                "mask0": mask0,
            }
        )
    return regions


def color_for_type(type_id: int) -> tuple[int, int, int]:
    if type_id <= 0:
        return (128, 128, 128)
    return DEFAULT_PALETTE[(type_id - 1) % len(DEFAULT_PALETTE)]


def cell_centroid_xy(cell: dict) -> tuple[float, float] | None:
    centroid = cell.get("centroid")
    if not centroid or len(centroid) != 2:
        return None
    return float(centroid[0]), float(centroid[1])


def cell_contour_xy(cell: dict) -> np.ndarray | None:
    contour = cell.get("contour")
    if not contour or len(contour) < 3:
        return None
    arr = np.asarray(contour, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return np.round(arr).astype(np.int32)


def draw_type_mask(cells: list[dict], region: dict) -> tuple[np.ndarray, dict[int, int]]:
    h0, w0 = region["h0"], region["w0"]
    minx0, miny0 = region["minx0"], region["miny0"]
    poly_mask = region["mask0"]

    type_mask = np.zeros((h0, w0, 3), dtype=np.uint8)
    type_counts: dict[int, int] = {}

    for cell in cells:
        centroid = cell_centroid_xy(cell)
        if centroid is None:
            continue
        cx, cy = centroid
        rx = int(round(cx - minx0))
        ry = int(round(cy - miny0))
        if not (0 <= rx < w0 and 0 <= ry < h0):
            continue
        if not poly_mask[ry, rx]:
            continue

        contour = cell_contour_xy(cell)
        if contour is None:
            continue
        local = contour.copy()
        local[:, 0] -= minx0
        local[:, 1] -= miny0

        type_id = int(cell.get("type", 0))
        cv2.fillPoly(type_mask, [local.reshape(-1, 1, 2)], color_for_type(type_id))
        type_counts[type_id] = type_counts.get(type_id, 0) + 1

    return type_mask, type_counts


def load_stitched_background(
    cellvit_root: Path,
    slide_base: str,
    region_id: str,
    shape: tuple[int, int, int],
    use_stitched: bool,
) -> np.ndarray:
    if use_stitched:
        stitched_path = cellvit_root / "stitched" / f"stitched_{slide_base}_{region_id}.tif"
        if stitched_path.exists():
            bg = imread(stitched_path)
            if bg.ndim == 2:
                bg = np.repeat(bg[:, :, None], 3, axis=2)
            if bg.shape[:2] == shape[:2]:
                return bg[:, :, :3].astype(np.uint8)

    return np.zeros(shape, dtype=np.uint8)


def blend_overlay(background: np.ndarray, type_mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = background.astype(np.float32).copy()
    mask = np.any(type_mask > 0, axis=2)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * type_mask[mask].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def type_label(type_id: int, type_map: dict[int, str]) -> str:
    return type_map.get(type_id, "background" if type_id == 0 else f"type_{type_id}")


def types_for_legend(
    type_counts: dict[int, int],
    type_map: dict[int, str],
    scope: str,
) -> list[int]:
    if scope == "all":
        type_ids: Iterable[int] = set(type_map.keys()) | set(type_counts.keys())
    else:
        type_ids = type_counts.keys()
    return sorted(int(t) for t in type_ids if int(t) > 0)


def add_legend_to_overlay(
    overlay: np.ndarray,
    type_counts: dict[int, int],
    type_map: dict[int, str],
    legend_position: str,
    legend_scope: str,
) -> np.ndarray:
    if legend_position == "none":
        return overlay

    type_ids = types_for_legend(type_counts, type_map, legend_scope)
    if not type_ids:
        return overlay

    h, w = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    ui_scale = float(np.clip(max(h, w) / 2600.0, 1.0, 3.2))
    title_scale = 0.72 * ui_scale
    label_scale = 0.55 * ui_scale
    thickness = max(1, int(round(ui_scale)))
    title_thickness = max(2, int(round(ui_scale * 1.6)))
    line_h = int(round(30 * ui_scale))
    pad = int(round(18 * ui_scale))
    swatch = int(round(18 * ui_scale))

    labels = [
        f"{type_label(type_id, type_map)} ({type_counts.get(type_id, 0):,})"
        for type_id in type_ids
    ]
    max_label_w = max(
        cv2.getTextSize(label, font, label_scale, thickness)[0][0] for label in labels
    )
    title_w = cv2.getTextSize("Cell type legend", font, title_scale, 2)[0][0]

    if legend_position == "right":
        min_panel_w = int(round(min(max(w * 0.18, 260 * ui_scale), w * 0.35)))
        panel_w = min(max(max_label_w + swatch + 3 * pad, title_w + 2 * pad, min_panel_w), w)
        panel_h = h
        canvas = np.full((h, w + panel_w, 3), 255, dtype=np.uint8)
        canvas[:, :w] = overlay
        x0, y0 = w, 0
    else:
        panel_w = w
        panel_h = pad * 2 + 30 + len(type_ids) * line_h
        canvas = np.full((h + panel_h, w, 3), 255, dtype=np.uint8)
        canvas[:h, :] = overlay
        x0, y0 = 0, h

    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1), (190, 190, 190), 1)
    cv2.putText(
        canvas,
        "Cell type legend",
        (x0 + pad, y0 + pad + int(round(18 * ui_scale))),
        font,
        title_scale,
        (30, 30, 30),
        title_thickness,
        cv2.LINE_AA,
    )

    y = y0 + pad + int(round(52 * ui_scale))
    for type_id, label in zip(type_ids, labels):
        color = color_for_type(type_id)
        cv2.rectangle(
            canvas,
            (x0 + pad, y - swatch + 4),
            (x0 + pad + swatch, y + 4),
            color,
            -1,
        )
        cv2.rectangle(
            canvas,
            (x0 + pad, y - swatch + 4),
            (x0 + pad + swatch, y + 4),
            (60, 60, 60),
            1,
        )
        cv2.putText(
            canvas,
            label,
            (x0 + pad + swatch + 12, y),
            font,
            label_scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    return canvas


def main() -> None:
    args = parse_args()

    global cv2, imread, imsave
    import cv2 as _cv2
    from skimage.io import imread as _imread, imsave as _imsave

    cv2 = _cv2
    imread = _imread
    imsave = _imsave

    cellvit_root = Path(args.cellvit_root)
    poly_dir = Path(args.poly_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cellvit_root / "type_colored"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cell_json:
        cell_jsons = [Path(args.cell_json)]
    else:
        cell_jsons = find_cell_jsons(cellvit_root)

    if not cell_jsons:
        raise FileNotFoundError(
            "No cells.json/cell.json found. Use --cell-json or check --cellvit-root."
        )

    legend: dict[str, dict] = {}
    for cells_json in cell_jsons:
        slide_base = slide_from_cells_json(cells_json)
        cells, cells_type_map = load_cellvit_payload(cells_json)

        detection_json = Path(args.detection_json) if args.detection_json else detection_json_for_cells_json(cells_json)
        if detection_json and detection_json.exists():
            detection_cells, detection_type_map = load_cellvit_payload(detection_json)
            cells, matched_detection_cells, match_mode = merge_cell_detection_types(
                cells,
                detection_cells,
            )
            type_map = detection_type_map or cells_type_map
        else:
            detection_json = None
            matched_detection_cells = 0
            match_mode = "not_found"
            type_map = cells_type_map

        regions = load_regions_for_slide(slide_base, poly_dir, args.mask_level)

        if not regions:
            print(f"No polyepi regions found for slide: {slide_base}")
            continue

        if detection_json is not None:
            print(
                f"Processing {slide_base}: {len(cells):,} cells, {len(regions)} ROI(s); "
                f"cell_detection types matched {matched_detection_cells:,} by {match_mode}"
            )
        else:
            print(
                f"Processing {slide_base}: {len(cells):,} cells, {len(regions)} ROI(s); "
                "cell_detection.json not found, using cells.json types"
            )

        for region in regions:
            region_id = region["region"]
            type_mask, type_counts = draw_type_mask(cells, region)

            bg = load_stitched_background(
                cellvit_root=cellvit_root,
                slide_base=slide_base,
                region_id=region_id,
                shape=type_mask.shape,
                use_stitched=args.background == "stitched",
            )
            overlay = blend_overlay(bg, type_mask, float(args.alpha))
            overlay_with_legend = add_legend_to_overlay(
                overlay=overlay,
                type_counts=type_counts,
                type_map=type_map,
                legend_position=args.legend,
                legend_scope=args.legend_scope,
            )

            safe_tag = f"{slide_base}_{region_id}"
            mask_path = out_dir / f"type_mask_{safe_tag}.tif"
            overlay_path = out_dir / f"type_overlay_{safe_tag}.tif"
            overlay_no_legend_path = out_dir / f"type_overlay_no_legend_{safe_tag}.tif"
            overlay_legend_path = out_dir / f"type_overlay_legend_{safe_tag}.tif"
            imsave(str(mask_path), type_mask, check_contrast=False)
            imsave(str(overlay_path), overlay_with_legend, check_contrast=False)
            imsave(str(overlay_no_legend_path), overlay, check_contrast=False)
            imsave(str(overlay_legend_path), overlay_with_legend, check_contrast=False)

            print(
                f"  {region_id}: {sum(type_counts.values()):,} colored cells "
                f"-> {overlay_legend_path}"
            )

            legend[safe_tag] = {
                "mask": str(mask_path),
                "overlay": str(overlay_path),
                "overlay_no_legend": str(overlay_no_legend_path),
                "overlay_with_legend": str(overlay_legend_path),
                "classification_source": str(detection_json) if detection_json else str(cells_json),
                "classification_match_mode": match_mode,
                "classification_matched_cells": int(matched_detection_cells),
                "counts": {str(k): int(v) for k, v in sorted(type_counts.items())},
            }

        for type_id in sorted({int(c.get("type", 0)) for c in cells} | set(type_map.keys())):
            legend.setdefault("type_colors", {})[str(type_id)] = {
                "name": type_map.get(type_id, f"type_{type_id}"),
                "rgb": list(color_for_type(type_id)),
            }

    legend_path = out_dir / "cell_type_color_legend.json"
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(legend, f, indent=2)
    print(f"Legend saved -> {legend_path}")


if __name__ == "__main__":
    main()
