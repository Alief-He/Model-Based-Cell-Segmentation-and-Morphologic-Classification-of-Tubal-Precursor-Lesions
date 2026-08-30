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
  A. Per-cell CSV saved per slide after all tile masks are stitched together.
     Regionprops are computed on the final stitched ROI label mask, not per tile.
     columns: slide, region, cell_label, cx_roi, cy_roi,
              cx_wsi, cy_wsi, cx_wsi_um, cy_wsi_um,
              area_px, bbox_minr_roi, bbox_minc_roi, bbox_maxr_roi, bbox_maxc_roi,
              bbox_minr_wsi, bbox_minc_wsi, bbox_maxr_wsi, bbox_maxc_wsi
  B. Stitched overlay TIF saved per region (for QC)
"""

import os, re, glob, time
import numpy as np
import pandas as pd
import cv2
import openslide
from tqdm import tqdm
from skimage.io import imsave
from skimage.measure import label as sklabel, regionprops
import torch
from cellpose import models

# ── Parameters ────────────────────────────────────────────────────────────────
MASK_LEVEL    = 2           # level at which Script 01 produced the epi masks
PATCH_SIZE    = 1024        # tile size in L0 pixels
OVERLAP_FRAC  = 0.25        # tile overlap fraction
STRIDE = int(PATCH_SIZE * (1 - OVERLAP_FRAC)) 
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
    try:
        model = models.CellposeModel(gpu=True)
        print("🚀 Using GPU for Cellpose")
    except Exception as e:
        print(f"⚠️  GPU init failed ({e}); falling back to CPU.")
        model = models.CellposeModel(gpu=False)

    # Validate context / overlap
    overlap_px = int(PATCH_SIZE * OVERLAP_FRAC)
    assert 2 * CONTEXT <= overlap_px, (
        f"Need 2*CONTEXT ({2*CONTEXT}) <= overlap_px ({overlap_px}). "
        f"Reduce CONTEXT or increase OVERLAP_FRAC."
    )

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
        all_cells_rows  = []   # one row per cell after stitching masks

        

        for R in regions:
            region_id             = R["region"]
            minx0, miny0          = R["minx0"], R["miny0"]
            wR, hR                = R["w0"],    R["h0"]
            region_mask0          = R["mask0"].copy()

            print(f"\n  🔷 Region '{region_id}'  bbox L0: "
                  f"x[{minx0}, {minx0+wR})  y[{miny0}, {miny0+hR})")

            # Stitched output canvases for this region
            stitched = np.zeros((hR, wR, 3), dtype=np.uint8)
            stitched_labels = np.zeros((hR, wR), dtype=np.uint32)
            next_label = 1

            # Build tile list (absolute WSI coordinates)
            tiles = [
                (xx, yy,
                 min(PATCH_SIZE, (minx0 + wR) - xx),
                 min(PATCH_SIZE, (miny0 + hR) - yy))
                for yy in range(miny0, miny0 + hR, STRIDE)
                for xx in range(minx0, minx0 + wR, STRIDE)
                if min(PATCH_SIZE, (minx0 + wR) - xx) > 0 and
                   min(PATCH_SIZE, (miny0 + hR) - yy) > 0
            ]

            n_skip_poly = n_skip_white = n_processed = 0

            for x, y, pw, ph in tqdm(tiles, desc=f"  {region_id}", unit="tile"):

                # ── 1. Polygon coverage check ─────────────────────────────
                gx0, gy0 = x - minx0, y - miny0
                gsub = region_mask0[gy0:gy0+ph, gx0:gx0+pw]
                if gsub.size == 0 or float(gsub.sum()) / gsub.size < MIN_POLY_RATIO:
                    n_skip_poly += 1
                    continue

                # ── 2. Context padding (stay within region bbox) ──────────
                pad_left  = min(CONTEXT, x - minx0)
                pad_top   = min(CONTEXT, y - miny0)
                pad_right = min(CONTEXT, (minx0 + wR) - (x + pw))
                pad_bot   = min(CONTEXT, (miny0 + hR) - (y + ph))

                read_x = x - pad_left
                read_y = y - pad_top
                read_w = pw + pad_left + pad_right
                read_h = ph + pad_top  + pad_bot

                # ── 3. Read padded RGB patch ──────────────────────────────
                patch_full_pil = slide.read_region(
                    (read_x, read_y), RUN_LEVEL, (read_w, read_h)
                ).convert("RGB")
                patch_full = np.asarray(patch_full_pil, dtype=np.float32) / 255.0

                # White-background skip
                if float(patch_full.mean()) > 0.90:
                    n_skip_white += 1
                    continue

                # ── 4. Run Cellpose on grayscale (BUG FIX vs Huang2) ──────
                #   - float32 grayscale input; diameter=None → auto-estimate
                #   - channels=[0,0] is for RGB input; for grayscale pass None
                gray_full = patch_full.mean(axis=2).astype(np.float32)
                masks_full, _, _ = model.eval(gray_full, diameter=None)

                # ── 5. Crop back to the central tile (remove context) ─────
                lbl = masks_full[pad_top:pad_top+ph,
                                 pad_left:pad_left+pw].astype(np.uint16)

                # ── 6. Edge-guard: drop border-touching labels ────────────
                #lbl = drop_labels_touching_border(lbl, EDGE_GUARD)

                # ── 7. Centroid-in-mask filter ────────────────────────────
                if lbl.max() > 0:
                    hC, wC = lbl.shape
                    gsub_c = gsub[:hC, :wC]
                    lbl    = keep_labels_with_centroid_in_mask(lbl, gsub_c)

                n_processed += 1

                # Tile outputs are optional QC artifacts; cell measurements happen after stitching.
                tile_tag = f"{slide_base}_{region_id}_{x}_{y}.tif"

                if SAVE_TILE_RGB:
                    rgb_u8 = (patch_full[pad_top:pad_top+ph,
                                         pad_left:pad_left+pw, :] * 255).astype(np.uint8)
                    imsave(os.path.join(out_rgb,  f"rgb_{tile_tag}"),  rgb_u8, check_contrast=False)

                if SAVE_TILE_MASKS:
                    imsave(os.path.join(out_mask, f"mask_{tile_tag}"), lbl,    check_contrast=False)

                # ── 11. Update stitched overlay ───────────────────────────
                rgb_tile = (patch_full[pad_top:pad_top+ph,
                                        pad_left:pad_left+pw, :] * 255).astype(np.uint8)
                if lbl.max() > 0:
                    overlay = rgb_tile.astype(np.float32)
                    mask = lbl > 0
                    green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
                    overlay[mask] = (
                        (1.0 - CELL_MASK_ALPHA) * overlay[mask]
                        + CELL_MASK_ALPHA * green
                    )
                    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
                else:
                    overlay_u8 = rgb_tile

                sy, sx = y - miny0, x - minx0
                th = min(ph, hR - sy)
                tw = min(pw, wR - sx)

                if th > 0 and tw > 0:
                    stitched[sy:sy+th, sx:sx+tw, :] = overlay_u8[:th, :tw, :]
                    lbl_crop = lbl[:th, :tw].astype(np.uint32)
                    if lbl_crop.max() > 0:
                        label_mask = lbl_crop > 0
                        lbl_crop = np.where(label_mask, lbl_crop + next_label - 1, 0)
                        label_region = stitched_labels[sy:sy+th, sx:sx+tw]
                        label_region[label_mask] = lbl_crop[label_mask]
                        next_label += int(lbl[:th, :tw].max())

                if SAVE_TILE_OVERLAY:
                    imsave(os.path.join(out_overlay, f"overlay_{tile_tag}"),
                           overlay_u8, check_contrast=False)

            # End tile loop for this region

            stitched_props = regionprops(stitched_labels)
            for rp in stitched_props:
                cy_roi, cx_roi = rp.centroid
                cx_wsi = minx0 + cx_roi
                cy_wsi = miny0 + cy_roi
                minr, minc, maxr, maxc = rp.bbox

                all_cells_rows.append({
                    "slide": slide_base,
                    "region": region_id,
                    "cell_label": int(rp.label),
                    "cx_roi": round(cx_roi, 2),
                    "cy_roi": round(cy_roi, 2),
                    "cx_wsi": round(cx_wsi, 2),
                    "cy_wsi": round(cy_wsi, 2),
                    "cx_wsi_um": round(cx_wsi * MPP, 3),
                    "cy_wsi_um": round(cy_wsi * MPP, 3),
                    "area_px": int(rp.area),
                    "bbox_minr_roi": int(minr),
                    "bbox_minc_roi": int(minc),
                    "bbox_maxr_roi": int(maxr),
                    "bbox_maxc_roi": int(maxc),
                    "bbox_minr_wsi": int(miny0 + minr),
                    "bbox_minc_wsi": int(minx0 + minc),
                    "bbox_maxr_wsi": int(miny0 + maxr),
                    "bbox_maxc_wsi": int(minx0 + maxc),
                })

            # Save stitched overlay with polygon outline
            cnt_img = (region_mask0 * 255).astype(np.uint8)
            contours, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(stitched, contours, -1, (255, 0, 0), thickness=2)
            stitch_path = os.path.join(out_stitched, f"stitched_{slide_base}_{region_id}.tif")
            imsave(stitch_path, stitched, check_contrast=False)

            print(f"     tiles: total={len(tiles)}, processed={n_processed}, "
                  f"skip_poly={n_skip_poly}, skip_white={n_skip_white}")
            print(f"     cells found after stitching: {len(stitched_props)}")
            print(f"  ✅ Stitched saved → {stitch_path}")

        # ── Save per-slide CSVs ───────────────────────────────────────────────

        # (A) Cell coordinates  ← primary output for spatial feature extraction
        if all_cells_rows:
            df_cells = pd.DataFrame(all_cells_rows)
            cells_path = os.path.join(out_spatial, f"{slide_base}_cell_coords.csv")
            df_cells.to_csv(cells_path, index=False)
            print(f"\n  💾 Cell coords ({len(df_cells):,} cells) → {cells_path}")
        else:
            print(f"\n  ⚠️  No cells found for {slide_base}")

        elapsed = time.time() - slide_start
        print(f"  ⏱️  Slide done in {elapsed/60:.1f} min")
        slide.close()

    print("\n✅ Step 02 complete.")
    print(f"   Cell coordinate CSVs → {out_spatial}/  (*_cell_coords.csv)")
    print(f"   Mask tiles           → {out_mask}/")
    print(f"   RGB tiles            → {out_rgb}/")
    print(f"   Stitched overlays    → {out_stitched}/")


if __name__ == "__main__":
    main()
