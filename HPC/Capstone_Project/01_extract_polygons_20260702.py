#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_extract_polygons_v2.py  (derived from 01_extract_polygons20260623.py)

GOAL OF THIS VERSION
This script now targets a "cellularity mask" rather than an "epithelium-only"
mask: it keeps ALL nucleated cells inside the pathologist's coarse polygon
(epithelial cells AND non-epithelial inflammatory cells such as lymphocytes /
macrophages), while removing background, blood vessels / RBC pools, necrosis,
large empty lumina, and pen marks. Epithelial vs. inflammatory typing happens
downstream in Script 02 (Cellpose + classifier) — this script's job is only
to exclude non-cellular / artifact regions.

Pipeline per region:
- Read slide + XML, rasterize Region polygons at a chosen pyramid level
- Convert crop to Hematoxylin/Eosin channels via color deconvolution
- Remove large white gaps (empty lumina) and pen-ink marks (optional)
- Remove blood vessel / RBC pools via an eosin-hue heuristic (optional, NEW)
- Remove low-texture necrotic candidate regions via a local-variance
  heuristic (optional, NEW; approximate — see caveat in detect_necrosis_mask)
- Refine remaining tissue into a cellularity mask using Hematoxylin channel +
  saturation gate + Otsu + size filters
- Grow slightly (geodesic dilation) then re-filter
- Save per-region masks (.npy + .png) and a debug panel (.png)

CAVEATS (please read)
- The blood-vessel and necrosis detectors are color/texture heuristics, not
  trained classifiers. They will not be as reliable as a small patch-level
  classifier trained on pathologist-marked vessel/necrosis examples. Treat
  their outputs as a first pass, spot-check the debug panels on a sample of
  slides, and retune EOSIN_HUE_*/NECROSIS_* thresholds (or replace with a
  trained classifier) if you see systematic false removals/inclusions.
- MPP-aware level picking assumes 'openslide.mpp-x'/'openslide.mpp-y' are
  present in the NDPI metadata (true for Hamamatsu NDPI in virtually all
  cases). If absent, it falls back to PARAMS['level'].

To run it, update the I/O folder parameters in
  IMAGE_DIR =
  XML_DIR   =
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import openslide
import xml.etree.ElementTree as ET
import cv2
import scipy.ndimage as ndi
from skimage.draw import polygon as sk_polygon
from skimage.io import imsave
from skimage.color import rgb2hed

# =========================
# Tunable PARAMETERS
# =========================
# NOTE ON UNITS: all *_um2 parameters below are physical areas in square
# microns, not raw pixel counts. They are converted to pixel-area thresholds
# per-slide inside main(), using the ACTUAL effective MPP of the working
# level (base 'openslide.mpp-x' * level downsample) so thresholds stay
# physically meaningful regardless of which level gets picked or how base
# scan MPP varies across scanners. This matters a lot here: at the default
# target_mpp (~0.885 um/px), a single small lymphocyte nucleus (~7 um dia,
# ~38 um^2) is only ~49 px, and a single epithelial nucleus (~12 um dia,
# ~113 um^2) is only ~144 px -- both smaller than the old min_epi_cell=200 px
# default, meaning isolated single nuclei were being erased as "speckle."
PARAMS = dict(
    level=2,
    intensity_threshold=235,   # "white" cutoff in [0..255] grayscale
    area_threshold_um2=350,    # min connected "white" region (um^2) to erase as lumen
    remove_white=True,         # remove big internal white gaps
    remove_red=True,           # remove red pools/marker inside polygon (HSV bands)
    refine_by_epi=True,        # set True to output filtered epithelium
    S_MIN=30,                  # saturation gate threshold (30-60 typical)
    min_epi_cell_um2=25,       # remove speckles smaller than ~half a small lymphocyte nucleus
    keep_frac=0.90,            # keep largest components until this fraction of epi area
    min_epi_size_um2=25,       # absolute minimum FINAL component size to keep (um^2);
                                # deliberately small so isolated single immune cells survive
    balanced_center=0.50,      # Otsu "balanced split" centre
    balanced_window=0.01,      # trigger when |frac-centre| < window (e.g. 0.49-0.51)
    EXPAND_PX=4,               # geodesic dilation radius (px @ level) to grow epi
    POST_ERODE_PX=1,           # light shrink after expansion to de-jag edges (0=off)
    CLOSE_ITER=1,              # set >0 for light closing on final epi
    OPEN_ITER=2,               # set >0 for light opening on final epi
    debug_max_cols=3,
    debug_figsize=(16, 6),

    # ---- NEW: MPP-aware level selection ----
    # If True, ignore PARAMS['level'] and instead pick the pyramid level whose
    # effective MPP is closest to target_mpp. Keeps physical pixel size
    # consistent across slides even if base scan MPP differs slightly across
    # scanners/sites (helps generalizability). Falls back to PARAMS['level']
    # if MPP metadata is missing from the NDPI.
    # CAUTION: 0.885 um/px is coarse for resolving small lymphocyte nuclei
    # (~7 um dia = only ~8 px across at this MPP). If Script 02 (Cellpose)
    # will segment cells directly on the crop produced at this level, rather
    # than re-cropping from a finer level using this mask as an ROI selector,
    # strongly consider lowering target_mpp (e.g. toward native ~0.2215 or
    # ~0.443 um/px) for better nucleus segmentation fidelity.
    use_mpp_level_selection=True,
    target_mpp=0.885,   # ~ 0.2215 * 2**2, i.e. equivalent to the old level=2 default

    # ---- NEW: blood vessel / RBC pool removal ----
    remove_vessel=True,
    EOSIN_HUE_LOW=0,        # HSV hue lower bound for eosin/RBC orange-red band
    EOSIN_HUE_HIGH=20,
    EOSIN_SAT_MIN=60,       # lower saturation floor than pen ink (RBCs are paler/more orange)
    EOSIN_SAT_MAX=200,      # ceiling to exclude very saturated pure pen ink
    EOSIN_VAL_MIN=120,
    vessel_area_threshold_um2=600,   # min connected RBC-like region (um^2) to remove
    vessel_close_iter=1,             # morphological closing to merge RBC speckle into vessel blobs

    # ---- NEW: necrosis candidate removal (heuristic, approximate) ----
    remove_necrosis=True,
    necrosis_window=9,          # local window (px) for texture/variance estimate -- NOTE: this
                                 # is still a raw pixel window, not MPP-converted; a 9px window
                                 # covers a different physical patch size at different levels, so
                                 # retune if you change target_mpp substantially.
    necrosis_var_thresh=25.0,   # below this local variance (on Hematoxylin channel) = candidate necrosis
    necrosis_min_intensity=40,  # ignore near-background pixels (already removed as white/lumen)
    necrosis_area_threshold_um2=1000,  # min connected low-texture region (um^2) to remove
)

# I/O folders
IMAGE_DIR = "../../../Image20260610"
XML_DIR   = "../../../xml20260610"
SAVE_DIR  = f"./polyepi_L{PARAMS['level']}"
DEBUG_DIR = "./polyepi_debug"


# =========================
# Helpers
# =========================

def setup_folders():
    for d in [SAVE_DIR, DEBUG_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_xml_polygons(xml_path, level):
    """
    Return list of (region_id, x_coords_lvl, y_coords_lvl) at the specified level.
    Uses Huang1's improved name cleaning to handle LRM marks and special characters.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    factor = 2 ** level

    for region in root.iter("Region"):
   #     if region.attrib.get("Type") != "0":
   #         continue

        raw_text = region.attrib.get("Text", "").strip()
        clean_name = "".join(c for c in raw_text if c.isprintable())
        clean_name = re.sub(r'[^\w\s-]', '', clean_name).strip().replace(" ", "_")
        region_id = clean_name if clean_name else f"Region_{len(polygons) + 1}"

        vertices = region.find("Vertices")
        if vertices is None:
            continue

        xs, ys = [], []
        for v in vertices.iter("Vertex"):
            xs.append(float(v.attrib["X"]) / factor)
            ys.append(float(v.attrib["Y"]) / factor)

        if len(xs) >= 3:
            polygons.append((region_id, np.array(xs), np.array(ys)))

    return polygons


def get_slide_mpp_xy(slide):
    """
    NEW: Read both openslide.mpp-x and openslide.mpp-y. Returns (mpp_x, mpp_y)
    as floats, or (None, None) if either is missing/unparseable.
    """
    try:
        mpp_x = float(slide.properties.get("openslide.mpp-x"))
        mpp_y = float(slide.properties.get("openslide.mpp-y"))
        return mpp_x, mpp_y
    except (TypeError, ValueError):
        return None, None


def get_isotropic_base_mpp(slide, tol_frac=0.02):
    """
    NEW: Return a single base MPP value for area-threshold math, which
    assumes square pixels (area = mpp^2). Averages mpp-x and mpp-y and warns
    loudly if they disagree by more than tol_frac (2% default) -- non-square
    pixels would make every um^2-based area threshold in this script subtly
    wrong, so this is worth checking rather than silently using mpp-x alone.
    Returns None if MPP metadata is unavailable.
    """
    mpp_x, mpp_y = get_slide_mpp_xy(slide)
    if mpp_x is None or mpp_y is None:
        return None
    if mpp_y > 0 and abs(mpp_x - mpp_y) / mpp_y > tol_frac:
        print(f"  ⚠️ mpp-x ({mpp_x:.4f}) and mpp-y ({mpp_y:.4f}) differ by more "
              f"than {tol_frac*100:.0f}% -- pixels are non-square on this slide. "
              f"Area-based (um^2) thresholds assume square pixels and will be "
              f"approximate here.")
    return (mpp_x + mpp_y) / 2.0


def get_effective_mpp(slide, level):
    """
    NEW: Effective microns-per-pixel at a given OpenSlide level
    (isotropic base MPP, averaged from mpp-x/mpp-y, * level downsample
    factor). Returns None if MPP metadata is unavailable, so callers can
    fall back to raw pixel thresholds.
    """
    base_mpp = get_isotropic_base_mpp(slide)
    if base_mpp is None:
        return None
    return base_mpp * slide.level_downsamples[level]


def um2_to_px(area_um2, eff_mpp, fallback_px):
    """
    NEW: Convert a physical area (um^2) to a pixel-area threshold at the
    current effective MPP. Falls back to a raw pixel value if MPP is
    unavailable, so the pipeline still runs (with a printed warning).
    """
    if eff_mpp is None or eff_mpp <= 0:
        return fallback_px
    return max(1, int(round(area_um2 / (eff_mpp ** 2))))


def build_pixel_thresholds(params, eff_mpp):
    """
    NEW: Build a per-slide copy of PARAMS with all *_um2 area thresholds
    converted to pixel-area equivalents at the slide's effective MPP. This
    keeps 'how big is a cell/lumen/vessel in pixels' consistent in physical
    terms across slides/scanners, instead of hardcoding pixel counts that
    silently mean different physical sizes depending on the working level.
    """
    p = dict(params)
    conversions = {
        "area_threshold_um2":            "area_threshold",
        "min_epi_cell_um2":              "min_epi_cell",
        "min_epi_size_um2":              "min_epi_size",
        "vessel_area_threshold_um2":     "vessel_area_threshold",
        "necrosis_area_threshold_um2":   "necrosis_area_threshold",
    }
    if eff_mpp is None:
        print("  ⚠️ No MPP metadata; um^2 area thresholds cannot be converted "
              "to pixels accurately. Falling back to a nominal 0.885 um/px assumption.")
        eff_mpp = 0.885
    for um2_key, px_key in conversions.items():
        p[px_key] = um2_to_px(params[um2_key], eff_mpp, fallback_px=200)
    return p


def pick_level_for_mpp(slide, target_mpp, fallback_level):
    """
    NEW: Choose the OpenSlide pyramid level whose effective MPP is closest to
    target_mpp, using NDPI metadata (average of openslide.mpp-x/mpp-y). This
    keeps the working resolution comparable across slides/scanners instead of
    trusting a fixed level index, which can correspond to different physical
    pixel sizes if base-level MPP varies. Falls back to fallback_level if MPP
    is unavailable.
    """
    base_mpp = get_isotropic_base_mpp(slide)
    if base_mpp is None:
        print("  ⚠️ No MPP metadata found; using fallback level", fallback_level)
        return fallback_level

    best_level, best_diff = fallback_level, None
    for lvl in range(slide.level_count):
        downsample = slide.level_downsamples[lvl]
        eff_mpp = base_mpp * downsample
        diff = abs(eff_mpp - target_mpp)
        if best_diff is None or diff < best_diff:
            best_diff, best_level = diff, lvl
    return best_level


def get_hematoxylin_eosin(crop_rgb):
    """
    NEW: Color-deconvolve RGB crop into Hematoxylin and Eosin channels.
    More robust to stain-batch / scanner color variation than plain grayscale,
    since it isolates the nuclear (H) and cytoplasmic/stromal (E) stain
    components directly rather than relying on overall luminance.
    Returns (hem_u8, eos_u8), both uint8 0-255, higher = more stain.
    """
    hed = rgb2hed(crop_rgb)
    hem = hed[..., 0]
    eos = hed[..., 1]

    def _to_u8(chan):
        lo, hi = np.percentile(chan, (1, 99))
        if hi <= lo:
            return np.zeros(chan.shape, dtype=np.uint8)
        out = np.clip((chan - lo) / (hi - lo), 0, 1)
        return (out * 255).astype(np.uint8)

    return _to_u8(hem), _to_u8(eos)


def detect_vessel_rbc_mask(crop_rgb, keep_mask, p):
    """
    NEW: Heuristic blood vessel / RBC pool detector, separate from the
    existing pen-ink 'remove_red' logic. RBCs are eosin-bright orange-red at
    moderate saturation, distinct from typically highly saturated pure-color
    pen ink. Flags packed clusters of RBC-colored pixels above an area
    threshold as vessel/RBC regions to exclude.

    CAVEAT: this is a color heuristic, not a vessel-wall/lumen structural
    detector. It will catch RBC-filled vessels well but may miss empty or
    RBC-sparse vessels, and could occasionally flag strongly eosinophilic
    cytoplasm. Spot-check the debug panel; retune EOSIN_* thresholds as needed.
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    rbc_color = (
        (hue >= p["EOSIN_HUE_LOW"]) & (hue <= p["EOSIN_HUE_HIGH"]) &
        (sat >= p["EOSIN_SAT_MIN"]) & (sat <= p["EOSIN_SAT_MAX"]) &
        (val >= p["EOSIN_VAL_MIN"])
    )
    rbc_color = (rbc_color & (keep_mask == 1)).astype(np.uint8)

    if p.get("vessel_close_iter", 0) > 0:
        se = ndi.generate_binary_structure(2, 2)
        rbc_color = ndi.binary_closing(
            rbc_color, structure=se, iterations=int(p["vessel_close_iter"])
        ).astype(np.uint8)

    vessel_mask = filter_small_regions(rbc_color, int(p["vessel_area_threshold"]))
    return vessel_mask


def detect_necrosis_mask(hem_u8, keep_mask, p):
    """
    NEW: Heuristic necrosis candidate detector using local texture on the
    Hematoxylin channel. Necrotic/karyorrhectic debris tends to lose crisp
    nuclear detail relative to viable, densely-packed nuclei, showing up as
    locally low-variance ("smudged") texture at mid intensity.

    CAVEAT: this is an approximate texture heuristic, not validated against
    pathologist-marked necrosis. Areas of sparse/loosely-packed viable tissue
    can also have low local variance and may be falsely flagged. Recommend
    validating against a sample of pathologist-confirmed necrotic regions
    before trusting this at scale, and consider replacing with a trained
    patch classifier if precision is inadequate.
    """
    win = int(p["necrosis_window"])
    hem_f = hem_u8.astype(np.float32)
    mean = cv2.blur(hem_f, (win, win))
    mean_sq = cv2.blur(hem_f * hem_f, (win, win))
    local_var = np.clip(mean_sq - mean * mean, 0, None)

    candidate = (
        (local_var < p["necrosis_var_thresh"]) &
        (hem_u8 >= p["necrosis_min_intensity"]) &
        (keep_mask == 1)
    ).astype(np.uint8)

    necrosis_mask = filter_small_regions(candidate, int(p["necrosis_area_threshold"]))
    return necrosis_mask


def extract_crop(slide, min_x, min_y, max_x, max_y, level):
    """Crop at target-level coords, reading from base level with OpenSlide."""
    W, H = int(max_x - min_x), int(max_y - min_y)
    crop = slide.read_region(
        (int(min_x * (2 ** level)), int(min_y * (2 ** level))), level, (W, H)
    ).convert("RGB")
    return np.array(crop)


def mask_paths(save_dir, slide_base, region_id, level, min_x, min_y, w, h):
    """Consistent filenames for binary PNG/NPY and JSON meta."""
    tag = f"{slide_base}_{region_id}_L{level}_{min_x}_{min_y}_{w}x{h}"
    png_path  = os.path.join(save_dir, f"polyepi_{tag}.png")
    npy_path  = os.path.join(save_dir, f"polyepi_{tag}.npy")
    meta_path = os.path.join(save_dir, f"polyepi_{tag}.json")
    return png_path, npy_path, meta_path


def save_poly_mask(slide_file, region_id, level, min_x, min_y, poly_mask, save_dir):
    """Save binary epithelium mask (0/1) as PNG + NPY + JSON meta."""
    slide_base = os.path.splitext(slide_file)[0]
    H, W = poly_mask.shape
    png_path, npy_path, meta_path = mask_paths(
        save_dir, slide_base, region_id, level, min_x, min_y, W, H
    )
    imsave(png_path, (poly_mask.astype(np.uint8) * 255), check_contrast=False)
    np.save(npy_path, poly_mask.astype(np.uint8))
    with open(meta_path, "w") as f:
        json.dump(
            {
                "slide":     slide_file,
                "region_id": region_id,
                "level":     level,
                "min_x":     int(min_x),
                "min_y":     int(min_y),
                "width":     int(W),
                "height":    int(H),
            },
            f,
            indent=2,
        )
    print(f"💾 Saved ROI: {region_id} -> {os.path.basename(npy_path)}")


def filter_small_regions(mask, min_size):
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    keep = np.zeros_like(mask, dtype=np.uint8)
    for i, sz in enumerate(sizes, 1):
        if sz >= min_size:
            keep[labeled == i] = 1
    return keep


def keep_largest_until_fraction(mask, frac=0.8):
    """Keep largest components until cumulative area >= frac * total area."""
    if frac is None or frac <= 0 or frac >= 1:
        return mask
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = np.array(ndi.sum(mask, labeled, range(1, num + 1)))
    order = np.argsort(sizes)[::-1]
    total = sizes.sum()
    cutoff = frac * total
    keep = np.zeros_like(mask, dtype=np.uint8)
    acc = 0
    for idx in order:
        lab = idx + 1
        keep[labeled == lab] = 1
        acc += sizes[idx]
        if acc >= cutoff:
            break
    return keep


def compute_epi_threshold(masked_gray, method="otsu"):
    """Compute intensity threshold from non-zero pixels inside the polygon mask."""
    vals = masked_gray[masked_gray > 0]
    if vals.size == 0:
        return 0
    if method == "otsu":
        hist = cv2.calcHist(
            [vals.astype(np.uint8)], [0], None, [256], [0, 256]
        ).ravel()
        total = vals.size
        sumB = wB = maximum = 0.0
        sum1 = np.dot(np.arange(256), hist)
        threshold = 0
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += i * hist[i]
            mB = sumB / wB
            mF = (sum1 - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between >= maximum:
                threshold = i
                maximum = between
        return int(threshold)
    else:
        return int(np.percentile(vals, 30))


def refine_polygon_with_epi(crop_rgb, cleaned_polygon_mask, p, hem_u8=None):
    """
    Refine epithelium inside the (already cleaned) polygon using:
      - saturation gate
      - Otsu with balanced fallback          
      - tiny speckle removal (min_epi_cell)
      - geodesic expansion (EXPAND_PX)       
      - keep_frac then absolute min_epi_size 
      - optional post-erosion                
      - hole fill + optional smoothing      
    Returns epi_final (0/1 uint8) and a debug dict.
    """
    if crop_rgb.shape[0] < 5 or crop_rgb.shape[1] < 5:
        return cleaned_polygon_mask.astype(np.uint8), {}

    hsv  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    keep_mask = cleaned_polygon_mask.astype(np.uint8)

    # NEW: prefer the Hematoxylin channel (color-deconvolved) over plain
    # grayscale when available — more robust to stain/scanner color drift.
    # hem_u8 is "higher = more nuclear stain", so invert to match the
    # existing "masked_gray < thr = cellular" convention below.
    if hem_u8 is not None:
        gray = 255 - hem_u8
    else:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # Masked gray + light denoise
    masked_gray = gray.copy()
    masked_gray[keep_mask == 0] = 0
    if min(masked_gray.shape) > 3:
        masked_gray = cv2.medianBlur(masked_gray, ksize=3)

    # Saturation gate (kept as a separate mask, applied after thresholding)
    sat_mask = ((keep_mask == 1) & (hsv[..., 1] >= p["S_MIN"])).astype(np.uint8)

    # ---------- ADAPTIVE OTSU THRESHOLD with balanced fallback ----------
    vals = masked_gray[keep_mask == 1]
    if vals.size == 0:
        thr   = 0
        frac  = 0.0
        epi_raw = np.zeros_like(masked_gray, dtype=np.uint8)
    else:
        thr  = compute_epi_threshold(masked_gray, method="otsu")
        frac = float((vals < thr).mean())
        balanced = abs(frac - p["balanced_center"]) < p["balanced_window"]
        if balanced:
            # ~50/50 split: Otsu is unreliable here; treat whole polygon as epi
            epi_raw = (keep_mask == 1).astype(np.uint8)
        else:
            epi_raw = ((keep_mask == 1) & (masked_gray < thr)).astype(np.uint8)

    # Apply saturation gate
    epi_raw = (epi_raw & sat_mask).astype(np.uint8)

    # Remove tiny speckles (min_epi_cell)
    epi_clean = filter_small_regions(epi_raw, int(p["min_epi_cell"]))

    # ---------- GEODESIC EXPANSION (recover loosened epi) ----------
    expand_px    = int(p.get("EXPAND_PX", 0))
    post_erode_px = int(p.get("POST_ERODE_PX", 0))

    epi_geo = epi_clean.copy()
    if expand_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1)
        )
        epi_geo = cv2.dilate(epi_geo.astype(np.uint8), k)
        epi_geo = (epi_geo & keep_mask).astype(np.uint8)  # stay inside polygon

    # ---------- RE-FILTER (keep_frac + absolute min size) ----------
    epi_sel = epi_geo
    if p.get("keep_frac") is not None and 0 < float(p["keep_frac"]) < 1:
        epi_sel = keep_largest_until_fraction(epi_sel, float(p["keep_frac"]))
    if int(p.get("min_epi_size", 0)) > 0:
        epi_sel = filter_small_regions(epi_sel, int(p["min_epi_size"]))

    # Optional gentle erosion to de-jag edges after dilation
    if post_erode_px > 0:
        k2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * post_erode_px + 1, 2 * post_erode_px + 1)
        )
        epi_sel = cv2.erode(epi_sel.astype(np.uint8), k2)

    # Fill holes
    epi_final = ndi.binary_fill_holes(epi_sel).astype(np.uint8)

    # Optional morphological smoothing
    if int(p["CLOSE_ITER"]) > 0 or int(p["OPEN_ITER"]) > 0:
        se2 = ndi.generate_binary_structure(2, 2)
        if int(p["CLOSE_ITER"]) > 0:
            epi_final = ndi.binary_closing(
                epi_final, structure=se2, iterations=int(p["CLOSE_ITER"])
            )
        if int(p["OPEN_ITER"]) > 0:
            epi_final = ndi.binary_opening(
                epi_final, structure=se2, iterations=int(p["OPEN_ITER"])
            )
        epi_final = epi_final.astype(np.uint8)

    debug = dict(
        masked_gray=masked_gray,
        sat_mask=sat_mask,
        thr=thr,
        frac=frac,
        epi_raw=epi_raw,
        epi_clean=epi_clean,
        epi_sel=epi_sel,
        epi_final=epi_final,
    )
    return epi_final, debug


def build_debug_panel_and_save(
    slide_base, region_id, crop_rgb, D, save_dir, figsize=(16, 6), max_cols=3
):
    """
    Save a simplified 3-panel debug PNG for QC of the FINAL mask only:
      1. Pathologist ROI polygon outline drawn on the crop RGB
      2. Final binary cellularity mask (what Script 02 will segment cells in)
      3. Final mask overlaid (translucent) on the crop RGB

    All intermediate "what got removed" panels (white/pen/vessel/necrosis
    masks, masked hematoxylin, raw/selected cellularity) are intentionally
    left out of this figure. They're still computed in the pipeline (useful
    if you need to debug a specific removal step later) but not plotted here.
    """
    epi = D["epi_final"]

    # Panel 1: ROI polygon outline on top of the original crop
    roi_overlay = crop_rgb.copy()
    poly_u8 = (D["poly_mask"] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(poly_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi_overlay, contours, -1, (255, 165, 0), thickness=3)  # orange outline

    # Panel 3: final mask overlaid (translucent green) on the original crop
    mask_overlay = crop_rgb.copy()
    mask_overlay[epi == 1] = (
        0.5 * mask_overlay[epi == 1] + np.array([0, 255, 0]) * 0.5
    ).astype(np.uint8)

    panels = [
        ("Pathologist ROI (orange outline)",       roi_overlay),
        ("Final cellularity mask",                 epi * 255),
        ("Final mask overlay (green)",              mask_overlay),
    ]

    rows = int(np.ceil(len(panels) / max_cols))
    plt.figure(figsize=figsize)
    for i, (title, img) in enumerate(panels, 1):
        plt.subplot(rows, max_cols, i)
        plt.imshow(
            img,
            cmap="gray" if (isinstance(img, np.ndarray) and img.ndim == 2) else None,
        )
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"debug_{slide_base}_{region_id}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🖼️  Saved debug panel -> {out_path}")


# =========================
# MAIN
# =========================
def main():
    setup_folders()

    ndpi_files = sorted(
        [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".ndpi")]
    )
    if not ndpi_files:
        print("❌ No .ndpi files found in", IMAGE_DIR)
        return

    for slide_file in ndpi_files:
        slide_path = os.path.join(IMAGE_DIR, slide_file)

        # Match XML to slide (try both naming conventions)
        xml_path = os.path.join(XML_DIR, f"{slide_file}.xml")
        if not os.path.exists(xml_path):
            xml_path = os.path.join(XML_DIR, slide_file.replace(".ndpi", ".xml"))
        if not os.path.exists(xml_path):
            print(f"⏩ Skipping {slide_file}, no XML found.")
            continue

        print(f"\n📦 Processing {slide_file}")
        slide = openslide.OpenSlide(slide_path)

        # NEW: pick working level by target MPP instead of trusting a fixed
        # level index, for consistency across slides/scanners.
        if PARAMS.get("use_mpp_level_selection", False):
            level = pick_level_for_mpp(slide, PARAMS["target_mpp"], PARAMS["level"])
            print(f"  🔎 Using level {level} (target MPP {PARAMS['target_mpp']})")
        else:
            level = PARAMS["level"]

        # NEW: convert um^2 area thresholds to pixel-area thresholds using
        # THIS slide's actual effective MPP at the chosen level, so 'how big
        # is a cell/lumen/vessel' stays physically consistent across slides.
        eff_mpp = get_effective_mpp(slide, level)
        p = build_pixel_thresholds(PARAMS, eff_mpp)
        print(f"  📏 Effective MPP at level {level}: "
              f"{eff_mpp if eff_mpp else 'unknown'} um/px  |  "
              f"min_epi_cell={p['min_epi_cell']}px  min_epi_size={p['min_epi_size']}px  "
              f"area_threshold={p['area_threshold']}px  "
              f"vessel_area_threshold={p['vessel_area_threshold']}px  "
              f"necrosis_area_threshold={p['necrosis_area_threshold']}px")

        polygons = parse_xml_polygons(xml_path, level)

        if not polygons:
            print("  ⚠️ No valid regions found.")
            slide.close()
            continue

        for region_id, xs, ys in polygons:
            try:
                min_x, max_x = int(xs.min()), int(xs.max())
                min_y, max_y = int(ys.min()), int(ys.max())
                W, H = max_x - min_x, max_y - min_y

                if W < 10 or H < 10:
                    print(f"  ⚠️ Skipping tiny region {region_id} ({W}x{H})")
                    continue

                # Read crop at chosen level
                crop_rgb = extract_crop(slide, min_x, min_y, max_x, max_y, level)
                gray     = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
                hem_u8, eos_u8 = get_hematoxylin_eosin(crop_rgb)  # NEW: color deconvolution

                # Base polygon mask
                poly_mask = np.zeros((H, W), dtype=np.uint8)
                rr, cc = sk_polygon(ys - ys.min(), xs - xs.min(), shape=poly_mask.shape)
                poly_mask[rr, cc] = 1

                # ---- Remove large white gaps (inside polygon only) ----
                white_big = np.zeros_like(poly_mask, dtype=np.uint8)
                keep_mask = poly_mask.copy()
                if PARAMS.get("remove_white", True):
                    white_inside = (
                        (poly_mask == 1) & (gray >= PARAMS["intensity_threshold"])
                    ).astype(np.uint8)
                    lbl_w, _ = ndi.label(white_inside)
                    if lbl_w.max() > 0:
                        sizes_w = ndi.sum(
                            white_inside, lbl_w, range(1, lbl_w.max() + 1)
                        )
                        for i, sz in enumerate(sizes_w, 1):
                            if sz >= p["area_threshold"]:
                                white_big[lbl_w == i] = 1
                    keep_mask[white_big == 1] = 0

                # ---- Remove red pools / annotation marker ----
                red_mask = np.zeros_like(poly_mask, dtype=np.uint8)
                if PARAMS.get("remove_red", True):
                    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
                    red1 = cv2.inRange(
                        hsv, np.array([0,   100, 100]), np.array([10,  255, 255])
                    )
                    red2 = cv2.inRange(
                        hsv, np.array([160, 100, 100]), np.array([180, 255, 255])
                    )
                    red_mask = ((red1 > 0) | (red2 > 0)).astype(np.uint8)
                    keep_mask[red_mask == 1] = 0

                # ---- NEW: Remove blood vessel / RBC pools ----
                vessel_mask = np.zeros_like(poly_mask, dtype=np.uint8)
                if PARAMS.get("remove_vessel", True):
                    vessel_mask = detect_vessel_rbc_mask(crop_rgb, keep_mask, p)
                    keep_mask[vessel_mask == 1] = 0

                # ---- NEW: Remove necrosis candidates (heuristic, approximate) ----
                necrosis_mask = np.zeros_like(poly_mask, dtype=np.uint8)
                if PARAMS.get("remove_necrosis", True):
                    necrosis_mask = detect_necrosis_mask(hem_u8, keep_mask, p)
                    keep_mask[necrosis_mask == 1] = 0

                # ---- Cellularity refinement (epithelial + inflammatory cells) ----
                if PARAMS.get("refine_by_epi", True):
                    epi_final, epi_dbg = refine_polygon_with_epi(
                        crop_rgb, keep_mask, p, hem_u8=hem_u8
                    )
                else:
                    epi_final = keep_mask.copy()
                    epi_dbg = dict(
                        masked_gray=gray,
                        sat_mask=np.zeros_like(keep_mask),
                        thr=0,
                        frac=0.0,
                        epi_raw=keep_mask,
                        epi_clean=keep_mask,
                        epi_sel=keep_mask,
                        epi_final=keep_mask,
                    )

                # Save mask + meta
                save_poly_mask(
                    slide_file, region_id, level,
                    min_x, min_y, epi_final, SAVE_DIR
                )

                # Debug panel
                D = dict(
                    poly_mask=poly_mask,
                    white_big=white_big,
                    red_mask=red_mask,
                    vessel_mask=vessel_mask,
                    necrosis_mask=necrosis_mask,
                    masked_gray=epi_dbg["masked_gray"],
                    thr=epi_dbg["thr"],
                    epi_raw=epi_dbg["epi_raw"],
                    epi_sel=epi_dbg["epi_sel"],
                    epi_final=epi_final,
                )
                slide_base = os.path.splitext(slide_file)[0]
                build_debug_panel_and_save(
                    slide_base, region_id, crop_rgb, D,
                    save_dir=DEBUG_DIR,
                    figsize=PARAMS.get("debug_figsize", (18, 12)),
                    max_cols=PARAMS.get("debug_max_cols", 4),
                )

            except Exception as e:
                print(f"❌ Failed to process region {region_id} in {slide_file}: {e}")
                continue

        slide.close()

    print("\n✅ All masks created.")
    print(f"   Epithelium masks -> {SAVE_DIR}")
    print(f"   Debug panels     -> {DEBUG_DIR}")


if __name__ == "__main__":
    main()