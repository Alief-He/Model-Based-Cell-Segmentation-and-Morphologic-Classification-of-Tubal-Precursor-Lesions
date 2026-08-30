#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_cellvit_in_polygons_hpc_Huang3_MODIFIED.py

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
CELL_MASK_ALPHA = 0.4

IMAGE_DIR = "./NDPI1"
POLY_DIR = f"./polyepi_L{MASK_LEVEL}"
OUT_ROOT = "./cellvit_output"

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

    print("  Running CellViT inference for the whole slide...", flush=True)
    try:
        subprocess.run(cmd, check=True)
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


def load_cellvit_cells(cells_json: Path) -> list[dict]:
    """Load CellViT contour detections from cells.json."""
    with open(cells_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("cells", [])


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


def rasterize_cellvit_roi(
    cells: list[dict],
    minx0: int,
    miny0: int,
    width: int,
    height: int,
    roi_mask: np.ndarray,
) -> np.ndarray:
    """
    Rasterize all CellViT contours whose centroids fall inside this ROI mask.
    CellViT coordinates are WSI-level x/y pixels; output is ROI-local row/col.
    """
    labels = np.zeros((height, width), dtype=np.uint32)
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
        next_label += 1

    labels[roi_mask == 0] = 0
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CellViT once per slide and rasterize full ROI masks without ROI tiling."
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--poly-dir", default=POLY_DIR)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--model", choices=["HIPT", "SAM"], default="HIPT")
    parser.add_argument("--nuclei-taxonomy", default="pannuke")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wsi-mpp", type=float, default=MPP)
    parser.add_argument("--wsi-magnification", type=int, default=MAGNIFICATION)
    parser.add_argument("--enforce-amp", action="store_true")
    parser.add_argument("--rerun-cellvit", action="store_true")
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
        cellvit_cells = load_cellvit_cells(cells_json)
        print(f"  Loaded {len(cellvit_cells):,} CellViT cells from {cells_json}", flush=True)

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
            labels = rasterize_cellvit_roi(
                cells=cellvit_cells,
                minx0=minx0,
                miny0=miny0,
                width=width,
                height=height,
                roi_mask=region_mask0,
            )

            overlay = roi_rgb.astype(np.float32)
            if labels.max() > 0:
                mask = labels > 0
                green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
                overlay[mask] = (1.0 - CELL_MASK_ALPHA) * overlay[mask] + CELL_MASK_ALPHA * green
            overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)

            stitched_props = regionprops(labels)
            for prop in stitched_props:
                cy_roi, cx_roi = prop.centroid
                cx_wsi = minx0 + cx_roi
                cy_wsi = miny0 + cy_roi
                minr, minc, maxr, maxc = prop.bbox

                all_cells_rows.append(
                    {
                        "slide": slide_base,
                        "region": region_id,
                        "cell_label": int(prop.label),
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

            tag = f"{slide_base}_{region_id}.tif"
            imsave(str(out_rgb / f"rgb_{tag}"), roi_rgb, check_contrast=False)
            imsave(str(out_mask / f"mask_{tag}"), labels, check_contrast=False)
            stitch_path = out_stitched / f"stitched_{tag}"
            imsave(str(stitch_path), overlay_u8, check_contrast=False)

            print(f"     cells found: {len(stitched_props)}")
            print(f"  Stitched saved -> {stitch_path}")

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
