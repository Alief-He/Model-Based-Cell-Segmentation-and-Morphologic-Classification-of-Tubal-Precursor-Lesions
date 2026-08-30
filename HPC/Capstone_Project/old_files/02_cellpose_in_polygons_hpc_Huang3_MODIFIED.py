#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_cellpose_in_polygons_hpc_Huang3.py

Improvements over Huang2, based on the original hpc.py:

BUG FIXES
  1. Cellpose called on grayscale float32 (not raw RGB uint8)
     → model.eval(gray_full, diameter=None) not model.eval(patch_rgb, diameter=40, channels=[0,0])
  2. Context padding added around each tile before Cellpose, then cropped back
     → prevents cells at tile borders being truncated / missed
  3. Edge-guard: labels touching the tile border are dropped before centroid filtering
     → avoids duplicate counts of the same cell across overlapping tiles
  4. Centroid-in-mask filter replaces pixel-zeroing
     → pixels inside a cell but outside the mask no longer split the cell label;
        instead the whole cell is kept or dropped based on where its centroid falls
  5. Tile skipping uses a minimum polygon-coverage ratio (MIN_POLY_RATIO=2%)
     instead of sum()==0, avoiding processing near-empty tiles
  6. White-tile skip (mean > 0.90) avoids wasting Cellpose on blank background
  7. mask labels are saved as int32 → uint16 (handles >255 cells per tile correctly,
     was already int32 in Huang2 but centroid coords were not recorded at all)

NEW FEATURES (critical for downstream)
  A. Per-cell CSV saved per slide with WSI-absolute centroid (x,y) in pixels at L0
     columns: slide, region, cell_label, tile_x, tile_y,
              cx_tile, cy_tile,   ← centroid within the tile (px, L0)
              cx_wsi,  cy_wsi,    ← centroid in full WSI coordinates (px, L0)
              cx_wsi_um, cy_wsi_um ← centroid in µm (using MPP=0.2215 µm/px)
              area_px, bbox_minr, bbox_minc, bbox_maxr, bbox_maxc
  B. Stitched overlay TIF saved per region (for QC)
  C. Tile-level log CSV (cells per tile) retained from Huang2
"""

import os, re, glob, time
import numpy as np
import pandas as pd
import cv2
import openslide
from tqdm import tqdm
from skimage.io import imsave
from skimage.measure import label as sklabel, regionprops
from cellpose import models
import torch




# ── Parameters ────────────────────────────────────────────────────────────────
MASK_LEVEL    = 2           # level at which Script 01 produced the epi masks
PATCH_SIZE    = 1024        # tile size in L0 pixels
OVERLAP_FRAC  = 0.25        # tile overlap fraction
CONTEXT       = 96          # context padding (px, L0) fed to Cellpose but cropped out after
EDGE_GUARD    = 16          # drop Cellpose labels touching this many px of the tile border
MIN_POLY_RATIO = 0.02       # skip tiles with <2% epi-mask pixels
RUN_LEVEL     = 0           # OpenSlide level for reading image data
MPP           = 0.2215      # µm per pixel at L0 (Hamamatsu 40× scan)

CELL_MASK_ALPHA = 0.4       # green Cellpose mask opacity in stitched/overlay outputs

SAVE_TILE_MASKS   = True    # save per-tile Cellpose label maps
SAVE_TILE_RGB     = True    # save per-tile RGB crops
SAVE_TILE_OVERLAY = False   # per-tile overlay (heavy); stitched overlay always saved

# I/O Folders
IMAGE_DIR = "./NDPIimage"
POLY_DIR  = f"./polyepi_L{MASK_LEVEL}"
OUT_ROOT  = "./cellpose_output"

# ── Filename regex (must match Script 01 output) ──────────────────────────────
POLY_RE = re.compile(
    r"^polyepi_(?P<slide>.+?)_(?P<region>.+?)_L(?P<lvl>\d+)"
    r"_(?P<minx>\d+)_(?P<miny>\d+)_(?P<w>\d+)x(?P<h>\d+)\.npy$"
)


# ── Helper functions ──────────────────────────────────────────────────────────

def upsample_mask_to_level0(mask_L: np.ndarray, level: int) -> np.ndarray:
    """Nearest-neighbour upsample of epi mask from level `level` to L0."""
    scale = 2 ** level
    H0, W0 = mask_L.shape[0] * scale, mask_L.shape[1] * scale
    return (cv2.resize(mask_L.astype(np.uint8), (W0, H0),
                       interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def relabel_after_masking(lbl: np.ndarray) -> np.ndarray:
    """Re-number labels 1..N after some have been zeroed out."""
    if lbl is None or lbl.size == 0 or lbl.max() == 0:
        return lbl.astype(np.uint16)
    return sklabel(lbl > 0, connectivity=1).astype(np.uint16)


def drop_labels_touching_border(lbl: np.ndarray, guard: int) -> np.ndarray:
    """
    Remove any Cellpose label whose footprint touches the tile border by
    `guard` pixels.  This avoids double-counting the same cell in adjacent
    overlapping tiles.
    """
    if guard <= 0 or lbl is None or lbl.max() == 0:
        return lbl.astype(np.uint16)
    H, W = lbl.shape
    g = int(min(guard, H // 2, W // 2))
    band = np.zeros((H, W), dtype=bool)
    band[:g, :] = band[-g:, :] = band[:, :g] = band[:, -g:] = True
    bad = set(np.unique(lbl[band])) - {0}
    if not bad:
        return lbl.astype(np.uint16)
    out = lbl.copy()
    for k in bad:
        out[out == k] = 0
    return relabel_after_masking(out)


def keep_labels_with_centroid_in_mask(lbl: np.ndarray,
                                       poly_mask: np.ndarray) -> np.ndarray:
    """
    Keep a Cellpose label only if its centroid falls inside the epi polygon mask.
    Safer than zeroing pixels, which can split cell labels.
    """
    if lbl is None or lbl.max() == 0:
        return lbl.astype(np.uint16)
    H, W = lbl.shape
    keep_ids = [
        rp.label
        for rp in regionprops(lbl)
        if (0 <= int(round(rp.centroid[0])) < H and
            0 <= int(round(rp.centroid[1])) < W and
            poly_mask[int(round(rp.centroid[0])), int(round(rp.centroid[1]))] > 0)
    ]
    if not keep_ids:
        return np.zeros_like(lbl, dtype=np.uint16)
    kept = np.isin(lbl, np.array(keep_ids, dtype=lbl.dtype))
    return relabel_after_masking(np.where(kept, lbl, 0).astype(np.uint16))


def load_regions_for_slide(slide_base: str, poly_dir: str):
    """Load all epi masks for one slide, return list of region dicts."""
    regions = []
    for p in sorted(glob.glob(os.path.join(poly_dir, f"polyepi_{slide_base}_*.npy"))):
        m = POLY_RE.match(os.path.basename(p))
        if not m or int(m.group("lvl")) != MASK_LEVEL:
            continue
        mask_L = np.load(p)
        if mask_L.ndim != 2:
            mask_L = mask_L.squeeze()
        mask_L  = (mask_L > 0).astype(np.uint8)
        mask0   = upsample_mask_to_level0(mask_L, MASK_LEVEL)
        scale   = 2 ** MASK_LEVEL
        minx0   = int(m.group("minx")) * scale
        miny0   = int(m.group("miny")) * scale
        h0, w0  = mask0.shape
        regions.append(dict(
            region=m.group("region"),
            mask0=mask0, minx0=minx0, miny0=miny0, w0=w0, h0=h0
        ))
    return regions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Output sub-folders
    out_mask      = os.path.join(OUT_ROOT, "mask");         os.makedirs(out_mask,      exist_ok=True)
    out_rgb       = os.path.join(OUT_ROOT, "rgb");          os.makedirs(out_rgb,       exist_ok=True)
    out_overlay   = os.path.join(OUT_ROOT, "overlay");      os.makedirs(out_overlay,   exist_ok=True)
    out_stitched  = os.path.join(OUT_ROOT, "stitched");     os.makedirs(out_stitched,  exist_ok=True)
    out_spatial   = os.path.join(OUT_ROOT, "spatial_data"); os.makedirs(out_spatial,   exist_ok=True)

    # ── Cellpose model ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        model = models.CellposeModel(gpu=True)
        print("🚀 Using GPU for Cellpose")
    else:
        print(f"⚠️  GPU init failed ({e}); falling back to CPU.")
        model = models.CellposeModel(gpu=False)

    '''# Validate context / overlap
    overlap_px = int(PATCH_SIZE * OVERLAP_FRAC)
    assert 2 * CONTEXT <= overlap_px, (
        f"Need 2*CONTEXT ({2*CONTEXT}) <= overlap_px ({overlap_px}). "
        f"Reduce CONTEXT or increase OVERLAP_FRAC."
    )'''

    # ── Collect slides ────────────────────────────────────────────────────────
    ndpi_files  = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".ndpi"))
    slide_bases = [os.path.splitext(f)[0] for f in ndpi_files]

    # Filter to slides that actually have epi masks
    have_poly = set()
    for p in glob.glob(os.path.join(POLY_DIR, "polyepi_*.npy")):
        mm = POLY_RE.match(os.path.basename(p))
        if mm:
            have_poly.add(mm.group("slide"))
    slide_bases = [b for b in slide_bases if b in have_poly] 

    if not slide_bases:
        print("❌ No slides with epi masks found.")
        return

    # ── Per-slide processing ──────────────────────────────────────────────────
    for slide_base in slide_bases:
        slide_path = os.path.join(IMAGE_DIR, f"{slide_base}.ndpi")
        if not os.path.exists(slide_path):
            print(f"⏩ Slide file not found: {slide_path}")
            continue

        regions = load_regions_for_slide(slide_base, POLY_DIR)
        if not regions:
            print(f"⏩ No masks for {slide_base}")
            continue

        print(f"\n📦 Processing slide: {slide_base}  ({len(regions)} ROI(s))")
        slide      = openslide.OpenSlide(slide_path)
        slide_start = time.time()

        # Accumulators for this slide
        all_cells_rows  = []   # one row per cell  → cell_coords CSV
        #tile_log_rows   = []   # one row per tile  → tile_log CSV

        #STRIDE = int(PATCH_SIZE * (1 - OVERLAP_FRAC))

        for R in tqdm(regions):
            region_id             = R["region"]
            minx0, miny0          = R["minx0"], R["miny0"]
            wR, hR                = R["w0"],    R["h0"]
            region_mask0          = R["mask0"].copy()

            print(f"\n  🔷 Region '{region_id}'  bbox L0: "
                  f"x[{minx0}, {minx0+wR})  y[{miny0}, {miny0+hR})")

            # Stitched overlay canvas for this region
            stitched = np.zeros((hR, wR, 3), dtype=np.uint8)

 
            ROI_raw = slide.read_region(
                    (minx0, miny0), RUN_LEVEL, (wR, hR)
                ).convert("RGB")
            
            
            
            ROI_arr = np.asarray(ROI_raw, dtype=np.float32) / 255.0

            ROI_masked = np.where(region_mask0[:, :, None], ROI_arr, 0)
            gray_full = ROI_masked.mean(axis=2).astype(np.float32)
            
            masks_full, _, _ = model.eval(gray_full, diameter=None)
            lbl = masks_full.astype(np.uint16)
            if lbl.max() > 0:
                lbl = keep_labels_with_centroid_in_mask(lbl, region_mask0)

            if lbl.max() > 0:
                for rp in regionprops(lbl):
                    cy_roi, cx_roi = rp.centroid
                    cx_wsi = minx0 + cx_roi
                    cy_wsi = miny0 + cy_roi

                    all_cells_rows.append({
                        "slide":      slide_base,
                        "region":     region_id,
                        "center":     f"({cx_wsi:.2f}, {cy_wsi:.2f})",
                        "area_px":    rp.area,
                        "bbox_minr":  rp.bbox[0],
                        "bbox_minc":  rp.bbox[1],
                        "bbox_maxr":  rp.bbox[2],
                        "bbox_maxc":  rp.bbox[3],
                    })
            
            rgb_tile = np.asarray(ROI_raw, dtype=np.uint8)
            if lbl.max() > 0:
                stitched = rgb_tile.astype(np.float32)
                mask = lbl > 0
                green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
                stitched[mask] = (
                    (1.0 - CELL_MASK_ALPHA) * stitched[mask]
                    + CELL_MASK_ALPHA * green
                )
                stitched = np.clip(stitched, 0, 255).astype(np.uint8)
            else:
                stitched = rgb_tile

            cnt_img = (region_mask0 * 255).astype(np.uint8)
            contours, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(stitched, contours, -1, (255, 0, 0), thickness=2)

            stitch_path = os.path.join(out_stitched, f"stitched_{slide_base}_{region_id}_MOD.tif")
            imsave(stitch_path, stitched, check_contrast=False)
            print(f"  Stitched saved -> {stitch_path}")

        if all_cells_rows:
            df_cells = pd.DataFrame(all_cells_rows)
            cells_path = os.path.join(out_spatial, f"{slide_base}_regionprops.csv")
            df_cells.to_csv(cells_path, index=False)
            print(f"\n  Regionprops ({len(df_cells):,} cells) -> {cells_path}")
        else:
            print(f"\n  No cells found for {slide_base}")

        elapsed = time.time() - slide_start
        print(f"  ⏱️  Slide done in {elapsed/60:.1f} min")
        slide.close()

    print("\n✅ Step 02 complete.")
    print(f"   Regionprops CSVs     -> {out_spatial}/  (*_regionprops.csv)")
    print(f"   Mask tiles           → {out_mask}/")
    print(f"   RGB tiles            → {out_rgb}/")
    print(f"   Stitched overlays    → {out_stitched}/")


if __name__ == "__main__":
    main()
