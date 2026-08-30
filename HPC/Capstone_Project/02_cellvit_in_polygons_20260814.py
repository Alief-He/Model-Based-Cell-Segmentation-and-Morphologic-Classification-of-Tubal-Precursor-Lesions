#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_cellvit_in_polygons_hpc_Huang3_08032026.py

CellViT version of the ROI cell-segmentation step.

This version is not based on project-level ROI tiles. CellViT is run once per
whole slide, then CellViT's WSI-level contours are rasterized directly into one
full label image per epithelial ROI.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import openslide
import pandas as pd
from skimage.io import imsave
from skimage.measure import regionprops


MASK_LEVEL = 2
RUN_LEVEL = 0
MPP = 0.2215
MAGNIFICATION = 40
CELL_MASK_ALPHA = 0.6

IMAGE_DIR = "./NDPI1"
POLY_DIR = f"./polyepi_L{MASK_LEVEL}"
OUT_ROOT = "./cellvit_output"

DEFAULT_PALETTE = [
    (0, 255, 0),      # 1 = Neoplastic (green)
    (0, 0, 255),      # 2 = Inflammatory (blue)
    (128, 128, 128),  # 3 = Connective (gray)
    (255, 0, 0),      # 4 = Dead (red)
    (255, 255, 0),    # 5 = Epithelial (yellow)
]

POLY_RE = re.compile(
    r"^polyepi_(?P<slide>.+?)_(?P<region>.+?)_L(?P<lvl>\d+)"
    r"_(?P<minx>\d+)_(?P<miny>\d+)_(?P<w>\d+)x(?P<h>\d+)\.npy$"
)


def upsample_mask_to_level0(mask_L: np.ndarray, level: int) -> np.ndarray:
    """Nearest-neighbour upsample of epi mask from Script 01 level to L0."""
    scale = 2**level
    h0, w0 = mask_L.shape[0] * scale, mask_L.shape[1] * scale
    return (
        cv2.resize(mask_L.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        > 0
    ).astype(np.uint8)


def load_regions_for_slide(slide_base: str, poly_dir: str) -> list[dict]:
    """Load all epithelial masks for one slide."""
    regions = []
    for path in sorted(glob.glob(os.path.join(poly_dir, f"polyepi_{slide_base}_*.npy"))):
        match = POLY_RE.match(os.path.basename(path))
        if not match or int(match.group("lvl")) != MASK_LEVEL:
            continue

        mask_L = np.load(path)
        if mask_L.ndim != 2:
            mask_L = mask_L.squeeze()
        mask_L = (mask_L > 0).astype(np.uint8)
        mask0 = upsample_mask_to_level0(mask_L, MASK_LEVEL)

        scale = 2**MASK_LEVEL
        minx0 = int(match.group("minx")) * scale
        miny0 = int(match.group("miny")) * scale
        h0, w0 = mask0.shape
        regions.append(
            {
                "region": match.group("region"),
                "mask0": mask0,
                "minx0": minx0,
                "miny0": miny0,
                "w0": w0,
                "h0": h0,
            }
        )
    return regions


def collect_slide_bases(image_dir: str, poly_dir: str) -> list[str]:
    """Collect NDPI slide names that have corresponding polygon masks."""
    ndpi_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".ndpi"))
    slide_bases = [os.path.splitext(f)[0] for f in ndpi_files]

    have_poly = set()
    for path in glob.glob(os.path.join(poly_dir, "polyepi_*.npy")):
        match = POLY_RE.match(os.path.basename(path))
        if match:
            have_poly.add(match.group("slide"))
    return [base for base in slide_bases if base in have_poly]


def run_cellvit_for_slide(
    slide_path: Path,
    raw_outdir: Path,
    model: str,
    nuclei_taxonomy: str,
    batch_size: int,
    gpu: int,
    wsi_mpp: float,
    wsi_magnification: int,
    enforce_amp: bool,
    rerun: bool,
) -> Path:
    """Run CellViT once per whole slide and return its cells.json path."""
    cells_json = raw_outdir / slide_path.stem / "cells.json"
    if cells_json.exists() and not rerun:
        print(f"  CellViT output exists -> {cells_json}")
        return cells_json

    if rerun and (raw_outdir / slide_path.stem).exists():
        shutil.rmtree(raw_outdir / slide_path.stem)

    cmd = [
        sys.executable,
        "-m",
        "cellvit.detect_cells",
        "--model",
        model,
        "--nuclei_taxonomy",
        nuclei_taxonomy,
        "--outdir",
        str(raw_outdir),
        "--gpu",
        str(gpu),
        "--batch_size",
        str(batch_size),
    ]
    if enforce_amp:
        cmd.append("--enforce_amp")
    cmd.extend(
        [
            "process_wsi",
            "--wsi_path",
            str(slide_path),
            "--wsi_mpp",
            str(wsi_mpp),
            "--wsi_magnification",
            str(wsi_magnification),
        ]
    )

    env = os.environ.copy()
    env.setdefault("CELLVIT_RAY_OBJECT_STORE_MEMORY_MB", "100")
    env.setdefault("CELLVIT_RAY_TASK_MEMORY_MB", "1000")

    print("  Running CellViT inference for the whole slide...", flush=True)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "CellViT failed for slide image:\n"
            f"  {slide_path}\n"
            "Run this command manually to see CellViT's full error output:\n"
            f"  {' '.join(str(part) for part in cmd)}"
        ) from exc

    if not cells_json.exists():
        raise FileNotFoundError(f"CellViT finished, but no cells.json found: {cells_json}")
    return cells_json


def load_cellvit_payload(cells_json: Path) -> tuple[list[dict], dict[int, str]]:
    """Load CellViT cells and type map from cells.json/cell_detection.json."""
    with open(cells_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_type_map = payload.get("type_map", {})
    type_map = {}
    for key, value in raw_type_map.items():
        try:
            type_map[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    return payload.get("cells", []), type_map


def cell_centroid_xy(cell: dict) -> tuple[float, float] | None:
    centroid = cell.get("centroid")
    if not centroid or len(centroid) != 2:
        return None
    return float(centroid[0]), float(centroid[1])


def centroid_key(cell: dict) -> tuple[int, int] | None:
    centroid = cell_centroid_xy(cell)
    if centroid is None:
        return None
    return int(round(centroid[0])), int(round(centroid[1]))


def merge_cell_detection_types(
    cells: list[dict],
    detections: list[dict],
) -> tuple[list[dict], int, str]:
    """Copy type/type_prob from cell_detection.json onto cells.json contours."""
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


def detection_json_for_cells_json(cells_json: Path) -> Path | None:
    detection_json = cells_json.with_name("cell_detection.json")
    if detection_json.exists():
        return detection_json
    return None


def cell_contour_xy(cell: dict) -> np.ndarray | None:
    contour = cell.get("contour")
    if not contour or len(contour) < 3:
        return None
    arr = np.asarray(contour, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return np.round(arr).astype(np.int32)


def cell_json_list(value) -> list | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [
            item.tolist() if isinstance(item, np.ndarray) else item
            for item in value
        ]
    return None


def rasterize_cellvit_roi(
    cells: list[dict],
    minx0: int,
    miny0: int,
    width: int,
    height: int,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict]]:
    """
    Rasterize all CellViT contours whose centroids fall inside this ROI mask.
    CellViT coordinates are WSI-level x/y pixels; output is ROI-local row/col.
    """
    labels = np.zeros((height, width), dtype=np.uint32)
    label_to_cell: dict[int, dict] = {}
    next_label = 1

    for cell in cells:
        centroid = cell_centroid_xy(cell)
        if centroid is None:
            continue

        cx_wsi, cy_wsi = centroid
        cx_roi = int(round(cx_wsi - minx0))
        cy_roi = int(round(cy_wsi - miny0))
        if not (0 <= cx_roi < width and 0 <= cy_roi < height):
            continue
        if roi_mask[cy_roi, cx_roi] == 0:
            continue

        contour = cell_contour_xy(cell)
        if contour is None:
            continue

        local = contour.copy()
        local[:, 0] -= minx0
        local[:, 1] -= miny0
        cell_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(cell_mask, [local.reshape(-1, 1, 2)], 1)
        labels[cell_mask > 0] = next_label
        label_to_cell[next_label] = cell
        next_label += 1

    labels[roi_mask == 0] = 0
    return labels, label_to_cell


def color_for_type(type_id: int) -> tuple[int, int, int]:
    if type_id <= 0:
        return (128, 128, 128)
    return DEFAULT_PALETTE[(type_id - 1) % len(DEFAULT_PALETTE)]


def cell_type_id(cell: dict) -> int:
    try:
        return int(cell.get("type", 0))
    except (TypeError, ValueError):
        return 0


def draw_type_mask_from_labels(
    labels: np.ndarray,
    label_to_cell: dict[int, dict],
) -> tuple[np.ndarray, dict[int, int]]:
    type_mask = np.zeros((*labels.shape, 3), dtype=np.uint8)
    type_counts: dict[int, int] = {}

    for label, cell in label_to_cell.items():
        type_id = cell_type_id(cell)
        type_mask[labels == label] = color_for_type(type_id)
        type_counts[type_id] = type_counts.get(type_id, 0) + 1

    return type_mask, type_counts


def blend_overlay(background: np.ndarray, type_mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = background.astype(np.float32).copy()
    mask = np.any(type_mask > 0, axis=2)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * type_mask[mask].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def type_label(type_id: int, type_map: dict[int, str]) -> str:
    return type_map.get(type_id, "background" if type_id == 0 else f"type_{type_id}")


def types_for_legend(type_counts: dict[int, int], type_map: dict[int, str]) -> list[int]:
    type_ids: Iterable[int] = set(type_counts.keys()) | set(type_map.keys())
    return sorted(int(t) for t in type_ids if int(t) > 0 and type_counts.get(int(t), 0) > 0)


def add_legend_to_overlay(
    overlay: np.ndarray,
    type_counts: dict[int, int],
    type_map: dict[int, str],
) -> np.ndarray:
    type_ids = types_for_legend(type_counts, type_map)
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

    labels_for_types = [
        f"{type_label(type_id, type_map)} ({type_counts.get(type_id, 0):,})"
        for type_id in type_ids
    ]
    max_label_w = max(
        cv2.getTextSize(label, font, label_scale, thickness)[0][0]
        for label in labels_for_types
    )
    title_w = cv2.getTextSize("Cell type legend", font, title_scale, 2)[0][0]

    min_panel_w = int(round(min(max(w * 0.18, 260 * ui_scale), w * 0.35)))
    panel_w = min(max(max_label_w + swatch + 3 * pad, title_w + 2 * pad, min_panel_w), w)
    canvas = np.full((h, w + panel_w, 3), 255, dtype=np.uint8)
    canvas[:, :w] = overlay
    x0, y0 = w, 0

    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, h - 1), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, h - 1), (190, 190, 190), 1)
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
    for type_id, label in zip(type_ids, labels_for_types):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CellViT once per slide and rasterize full ROI masks without ROI tiling."
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--poly-dir", default=POLY_DIR)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--model", choices=["HIPT", "SAM"], default="HIPT")
    parser.add_argument("--nuclei-taxonomy", default="pannuke")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wsi-mpp", type=float, default=MPP)
    parser.add_argument("--wsi-magnification", type=int, default=MAGNIFICATION)
    parser.add_argument("--enforce-amp", action="store_true")
    parser.add_argument("--rerun-cellvit", action="store_true")
    parser.add_argument("--slide-name", type=str, default=None, help="Only process this slide")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root)
    raw_outdir = out_root / "raw_cellvit"
    out_mask = out_root / "region_mask"
    out_rgb = out_root / "region_rgb"
    out_stitched = out_root / "stitched"
    out_spatial = out_root / "spatial_data"
    for path in [raw_outdir, out_mask, out_rgb, out_stitched, out_spatial]:
        path.mkdir(parents=True, exist_ok=True)

    slide_bases = collect_slide_bases(args.image_dir, args.poly_dir)

    # If --slide-name is provided (e.g. by a SLURM array task),
    # restrict this process to exactly that one slide.
    if args.slide_name:
        requested_slide = Path(args.slide_name).stem

        if requested_slide not in slide_bases:
            print(
                f"ERROR: Requested slide '{args.slide_name}' was not found "
                "or has no corresponding epithelial polygon masks."
            )
            return

        slide_bases = [requested_slide]

    if not slide_bases:
        print("No slides with epi masks found.")
        return

    for slide_base in slide_bases:
        slide_path = Path(args.image_dir) / f"{slide_base}.ndpi"
        if not slide_path.exists():
            print(f"Slide file not found: {slide_path}")
            continue

        regions = load_regions_for_slide(slide_base, args.poly_dir)
        if not regions:
            print(f"No masks for {slide_base}")
            continue

        print(f"\nProcessing slide: {slide_base} ({len(regions)} ROI(s))")
        slide_start = time.time()
        cells_json = run_cellvit_for_slide(
            slide_path=slide_path,
            raw_outdir=raw_outdir,
            model=args.model,
            nuclei_taxonomy=args.nuclei_taxonomy,
            batch_size=args.batch_size,
            gpu=args.gpu,
            wsi_mpp=args.wsi_mpp,
            wsi_magnification=args.wsi_magnification,
            enforce_amp=args.enforce_amp,
            rerun=args.rerun_cellvit,
        )
        cellvit_cells, cells_type_map = load_cellvit_payload(cells_json)
        detection_json = detection_json_for_cells_json(cells_json)
        if detection_json is not None:
            detection_cells, detection_type_map = load_cellvit_payload(detection_json)
            cellvit_cells, matched_detection_cells, match_mode = merge_cell_detection_types(
                cellvit_cells,
                detection_cells,
            )
            type_map = detection_type_map or cells_type_map
        else:
            matched_detection_cells = 0
            match_mode = "not_found"
            type_map = cells_type_map

        print(f"  Loaded {len(cellvit_cells):,} CellViT cells from {cells_json}", flush=True)
        if detection_json is not None:
            print(
                f"  cell_detection types matched {matched_detection_cells:,} "
                f"by {match_mode}: {detection_json}",
                flush=True,
            )
        else:
            print("  cell_detection.json not found, using cells.json types", flush=True)

        slide = openslide.OpenSlide(str(slide_path))
        all_cells_rows = []

        for region in regions:
            region_id = region["region"]
            minx0, miny0 = region["minx0"], region["miny0"]
            width, height = region["w0"], region["h0"]
            region_mask0 = region["mask0"].copy()

            print(
                f"\n  Region '{region_id}' bbox L0: "
                f"x[{minx0}, {minx0 + width}) y[{miny0}, {miny0 + height})"
            )

            roi_pil = slide.read_region((minx0, miny0), RUN_LEVEL, (width, height)).convert("RGB")
            roi_rgb = np.asarray(roi_pil, dtype=np.uint8)
            labels, label_to_cell = rasterize_cellvit_roi(
                cells=cellvit_cells,
                minx0=minx0,
                miny0=miny0,
                width=width,
                height=height,
                roi_mask=region_mask0,
            )

            type_mask, type_counts = draw_type_mask_from_labels(labels, label_to_cell)
            overlay_u8 = blend_overlay(roi_rgb, type_mask, CELL_MASK_ALPHA)

            stitched_props = regionprops(labels)
            for prop in stitched_props:
                cy_roi, cx_roi = prop.centroid
                cx_wsi = minx0 + cx_roi
                cy_wsi = miny0 + cy_roi
                minr, minc, maxr, maxc = prop.bbox
                source_cell = label_to_cell.get(int(prop.label), {})

                all_cells_rows.append(
                    {
                        "slide": slide_base,
                        "region": region_id,
                        "cell_label": int(prop.label),
                        "type": source_cell.get("type"),
                        "center": cell_json_list(source_cell.get("centroid")),
                        "contour": cell_json_list(source_cell.get("contour")),
                        "cx_roi": round(cx_roi, 2),
                        "cy_roi": round(cy_roi, 2),
                        "cx_wsi": round(cx_wsi, 2),
                        "cy_wsi": round(cy_wsi, 2),
                        "cx_wsi_um": round(cx_wsi * args.wsi_mpp, 3),
                        "cy_wsi_um": round(cy_wsi * args.wsi_mpp, 3),
                        "area_px": int(prop.area),
                        "bbox_minr_roi": int(minr),
                        "bbox_minc_roi": int(minc),
                        "bbox_maxr_roi": int(maxr),
                        "bbox_maxc_roi": int(maxc),
                        "bbox_minr_wsi": int(miny0 + minr),
                        "bbox_minc_wsi": int(minx0 + minc),
                        "bbox_maxr_wsi": int(miny0 + maxr),
                        "bbox_maxc_wsi": int(minx0 + maxc),
                    }
                )

            cnt_img = (region_mask0 * 255).astype(np.uint8)
            contours, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay_u8, contours, -1, (255, 0, 0), thickness=2)
            overlay_u8 = add_legend_to_overlay(overlay_u8, type_counts, type_map)

            tag = f"{slide_base}_{region_id}.tif"
            imsave(str(out_rgb / f"rgb_{tag}"), roi_rgb, check_contrast=False)
            imsave(str(out_mask / f"mask_{tag}"), labels, check_contrast=False)
            stitch_path = out_stitched / f"stitched_{tag}"
            imsave(str(stitch_path), overlay_u8, check_contrast=False)

            print(f"     cells found: {len(stitched_props)}")
            print(f"     type counts: {type_counts}")
            print(f"  Type-colored stitched image with legend saved -> {stitch_path}")

        if all_cells_rows:
            df_cells = pd.DataFrame(all_cells_rows)
            cells_path = out_spatial / f"{slide_base}_cell_coords.csv"
            df_cells.to_csv(cells_path, index=False)
            print(f"\n  Cell coords ({len(df_cells):,} cells) -> {cells_path}")
        else:
            print(f"\n  No cells found for {slide_base}")

        elapsed = time.time() - slide_start
        print(f"  Slide done in {elapsed / 60:.1f} min")
        slide.close()

    print("\nStep 02 CellViT complete.")
    print(f"   Cell coordinate CSVs -> {out_spatial}/ (*_cell_coords.csv)")
    print(f"   Region masks         -> {out_mask}/")
    print(f"   Region RGBs          -> {out_rgb}/")
    print(f"   Stitched overlays    -> {out_stitched}/")
    print(f"   Raw CellViT outputs  -> {raw_outdir}/")


if __name__ == "__main__":
    main()
