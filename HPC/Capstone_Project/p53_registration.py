"""Register p53 to H&E and measure p53 stain in existing H&E nuclear masks.

All saved transforms map H&E physical coordinates (microns) to p53 physical
coordinates. OpenSlide coordinates, including landmarks, use level-0 pixels.
See P53_WORKFLOW.md for the manifest, review and supervised-learning workflow.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from skimage.color import rgb2hed
import tifffile

KEYS = ["slide", "region", "cell_label"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_pairs(path):
    path = Path(path).resolve()
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    seen, slides = set(), set()
    for pair in pairs:
        for key in ("pair_id", "patient_id", "p53_slide"):
            if not pair.get(key):
                raise ValueError(f"Pair is missing {key}")
        pid = pair["pair_id"]
        pair["patient_id"] = str(pair["patient_id"])
        if pid in seen or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in pid):
            raise ValueError("pair_id must be unique and contain only letters, numbers, _ or -")
        seen.add(pid)
        if pair.get("section_relationship", "unknown") not in ("same_section", "serial_section", "unknown"):
            raise ValueError(f"Invalid section_relationship for {pid}")
        for key in ("he_slide", "p53_slide", "landmarks", "qc_landmarks"):
            if pair.get(key):
                pair[key] = str((path.parent / pair[key]).resolve())
        if pair.get("he_slide"):
            slide = Path(pair["he_slide"]).stem
            if slide in slides:
                raise ValueError(f"Duplicate H&E slide {slide}; use one p53 pair per H&E slide")
            slides.add(slide)
    return pairs


def select_pairs(path, pair_id=None):
    pairs = read_pairs(path)
    if pair_id:
        pairs = [p for p in pairs if p["pair_id"] == pair_id]
    if not pairs:
        raise ValueError("No matching pairs in manifest")
    return pairs


def get_mpp(slide, override=None):
    if override is not None:
        values = np.asarray(override if isinstance(override, list) else [override, override], dtype=float)
    else:
        values = np.array([float(slide.properties.get(f"openslide.mpp-{axis}", "nan")) for axis in "xy"])
    if values.shape != (2,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing/invalid MPP metadata: set he_mpp or p53_mpp to [x, y] in microns/pixel")
    return values


def thumbnail(slide, mpp, size):
    rgb = np.asarray(slide.get_thumbnail((size, size)).convert("RGB"))
    spacing = np.asarray(slide.dimensions) / np.array(rgb.shape[1::-1]) * mpp
    # Pixel centers: thumbnail index 0 corresponds to (downsample - 1) / 2 at L0.
    origin = (spacing - mpp) / 2
    return rgb, spacing, origin


def scalar_image(array, spacing, origin):
    img = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    img.SetSpacing(tuple(map(float, spacing)))
    img.SetOrigin(tuple(map(float, origin)))
    return img


def tissue_mask(rgb):
    # Reject off-white glass as well as scanner padding; retain blue/brown tissue.
    minimum, maximum = rgb.min(axis=2).astype(float), rgb.max(axis=2).astype(float)
    mean = rgb.mean(axis=2)
    colored = (minimum < 230) & ((maximum - minimum) > 10)
    return (colored | (mean < 180)) & (mean > 15)


def structure(rgb):
    h = rgb2hed(rgb)[..., 0]
    values = h[tissue_mask(rgb)]
    high = np.percentile(values, 99) if values.size else 0
    if high <= 0:
        raise ValueError("No usable hematoxylin structure in slide preview")
    return np.clip(h / high, 0, 1).astype(np.float32)


def estimate_affine(source_um, target_um, tolerance_um):
    source_um, target_um = np.asarray(source_um), np.asarray(target_um)
    if len(source_um) < 3 or np.linalg.matrix_rank(source_um - source_um.mean(axis=0)) < 2:
        raise ValueError("Need at least three non-collinear matching landmarks")
    matrix, inliers = cv2.estimateAffine2D(
        source_um.astype(np.float32), target_um.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=float(tolerance_um),
        maxIters=5000, confidence=0.999, refineIters=20,
    )
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        raise ValueError("Affine initialization failed or reflected tissue; check slide orientation and landmarks")
    tx = sitk.AffineTransform(2)
    tx.SetMatrix(matrix[:, :2].ravel().tolist())
    tx.SetTranslation(matrix[:, 2].tolist())
    return tx, inliers.ravel().astype(bool)


def landmark_arrays(path, he_mpp, p53_mpp):
    df = pd.read_csv(path)
    a = df[["he_x", "he_y"]].to_numpy(float) * he_mpp
    b = df[["p53_x", "p53_y"]].to_numpy(float) * p53_mpp
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Landmark coordinates must be finite level-0 pixel coordinates")
    return a, b


def initial_transform(he_rgb, p53_rgb, he_geom, p53_geom, pair, he_mpp, p53_mpp):
    if pair.get("landmarks"):
        a, b = landmark_arrays(pair["landmarks"], he_mpp, p53_mpp)
        tx, inliers = estimate_affine(a, b, 50)
        return tx, {"method": "landmarks", "matches": len(a), "inliers": int(inliers.sum())}
    detector = cv2.SIFT_create(nfeatures=12000)
    ka, da = detector.detectAndCompute((structure(he_rgb) * 255).astype("uint8"), None)
    kb, db = detector.detectAndCompute((structure(p53_rgb) * 255).astype("uint8"), None)
    if da is None or db is None or len(db) < 2:
        raise ValueError("No reliable automatic landmarks; provide landmarks CSV in manifest")
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .7 * m[1].distance]
    if len(good) < 12:
        raise ValueError("Too few cross-stain SIFT matches; provide landmarks CSV in manifest")
    a = np.array([ka[m.queryIdx].pt for m in good]) * he_geom[0] + he_geom[1]
    b = np.array([kb[m.trainIdx].pt for m in good]) * p53_geom[0] + p53_geom[1]
    tx, inliers = estimate_affine(a, b, max(p53_geom[0]) * 3)
    if inliers.sum() < 12 or inliers.mean() < .25:
        raise ValueError("Unreliable SIFT initialization; provide landmarks CSV in manifest")
    return tx, {"method": "SIFT_RANSAC", "matches": len(good), "inliers": int(inliers.sum())}


def registration_method(iterations):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(.25, 42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    reg.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=iterations,
        convergenceMinimumValue=1e-5, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    return reg


def refine_transform(fixed, moving, initial, iterations=100, deformable=False):
    reg = registration_method(iterations)
    reg.SetInitialTransform(initial, inPlace=False)
    affine = reg.Execute(fixed, moving)
    diagnostics = {"affine_metric": float(reg.GetMetricValue()), "affine_stop": reg.GetOptimizerStopConditionDescription()}
    if not deformable:
        return affine, diagnostics
    reg = registration_method(iterations)
    bspline = sitk.BSplineTransformInitializer(fixed, [8, 8], order=3)
    reg.SetMovingInitialTransform(affine)
    reg.SetInitialTransform(bspline, inPlace=True)
    reg.Execute(fixed, moving)
    result = sitk.CompositeTransform([affine, bspline])
    result.FlattenTransform()
    diagnostics.update(bspline_metric=float(reg.GetMetricValue()), bspline_stop=reg.GetOptimizerStopConditionDescription())
    return result, diagnostics


def resample_rgb(rgb, spacing, origin, reference, transform):
    return np.stack([
        sitk.GetArrayFromImage(sitk.Resample(
            scalar_image(rgb[..., i], spacing, origin), reference, transform,
            sitk.sitkLinear, 255, sitk.sitkFloat32,
        )) for i in range(3)
    ], axis=-1).clip(0, 255).astype("uint8")


def pair_signature(pair):
    # Bind artifacts to slide identity, geometry overrides and section relationship.
    fields = {k: pair.get(k) for k in ("pair_id", "patient_id", "he_slide", "p53_slide", "he_mpp", "p53_mpp", "section_relationship")}
    for key in ("he_slide", "p53_slide"):
        stat = Path(pair[key]).stat()
        fields[key + "_stat"] = [stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def register_pair(pair, out_root, size=2048, iterations=100, deformable=False):
    if not pair.get("he_slide"):
        raise ValueError(f"Set he_slide for {pair['pair_id']} in the manifest; no matching H&E has been specified")
    out = Path(out_root) / pair["pair_id"]
    out.mkdir(parents=True, exist_ok=True)
    # Reset approval before attempting a new registration, including failed attempts.
    review = {"registration_approved": False, "reviewer": "", "dab_threshold": None,
              "positive_fraction": .6, "negative_fraction": .1,
              "note": "Review overview and region QC; set a calibrated DAB threshold and complete region_review.csv."}
    write_json(out / "review.json", review)
    if (out / "region_review.csv").exists():
        old_review = pd.read_csv(out / "region_review.csv", dtype={"region": str})
        old_review["approved"] = False
        old_review["max_registration_error_um"] = np.nan
        old_review.to_csv(out / "region_review.csv", index=False)
    with openslide.OpenSlide(pair["he_slide"]) as he, openslide.OpenSlide(pair["p53_slide"]) as p53:
        hm, pm = get_mpp(he, pair.get("he_mpp")), get_mpp(p53, pair.get("p53_mpp"))
        hr, hs, ho = thumbnail(he, hm, size)
        pr, ps, po = thumbnail(p53, pm, size)
        initial, diagnostics = initial_transform(hr, pr, (hs, ho), (ps, po), pair, hm, pm)
        fixed = scalar_image(structure(hr), hs, ho)
        moving = scalar_image(structure(pr), ps, po)
        tx, refinement = refine_transform(fixed, moving, initial, iterations, deformable)
        diagnostics.update(refinement)
        warped = resample_rgb(pr, ps, po, fixed, tx)
        ht, pt = tissue_mask(hr), tissue_mask(warped)
        dice = 2 * (ht & pt).sum() / max(1, ht.sum() + pt.sum())
        field = sitk.TransformToDisplacementField(tx, sitk.sitkVectorFloat64, fixed.GetSize(), fixed.GetOrigin(), fixed.GetSpacing())
        jac = sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(field))
        diagnostics.update(tissue_dice=float(dice), nonpositive_jacobian_fraction=float((jac[ht] <= 0).mean()) if ht.any() else 1.)
        if pair.get("qc_landmarks"):
            if pair.get("qc_landmarks") == pair.get("landmarks"):
                raise ValueError("qc_landmarks must be separate from initialization landmarks")
            a, b = landmark_arrays(pair["qc_landmarks"], hm, pm)
            if len(a) == 0:
                raise ValueError("QC landmark table is empty")
            errors = np.linalg.norm(np.array([tx.TransformPoint(tuple(x)) for x in a]) - b, axis=1)
            diagnostics.update(qc_landmark_median_um=float(np.median(errors)), qc_landmark_max_um=float(errors.max()))
            residuals = pd.read_csv(pair["qc_landmarks"])
            residuals["registration_error_um"] = errors
            residuals.to_csv(out / "qc_landmark_residuals.csv", index=False)
        metadata = {"registration_id": str(uuid.uuid4()), "pair_signature": pair_signature(pair),
                    "pair_id": pair["pair_id"], "patient_id": pair["patient_id"],
                    "slide": Path(pair["he_slide"]).stem, "section_relationship": pair.get("section_relationship", "unknown"),
                    "he_mpp": hm.tolist(), "p53_mpp": pm.tolist(),
                    "transform_direction": "HE_microns_to_p53_microns", "deformable": deformable,
                    "qc": diagnostics, "thumbnail_spacing_um": hs.tolist()}
    Image.fromarray(hr).save(out / "he_preview.png")
    Image.fromarray(warped).save(out / "p53_registered_preview.png")
    Image.fromarray(((hr.astype(float) + warped) / 2).astype("uint8")).save(out / "overlay.png")
    yy, xx = np.indices(hr.shape[:2])
    checker = np.where((((xx // 64 + yy // 64) % 2) == 0)[..., None], hr, warped)
    Image.fromarray(checker).save(out / "checkerboard.png")
    sitk.WriteTransform(tx, str(out / "he_to_p53.h5"))
    write_json(out / "registration.json", metadata)
    print(f"Registered {pair['pair_id']}: tissue Dice={dice:.3f}; review required", flush=True)
    return metadata


def mapped_grid(tx, x, y, width, height, he_mpp, p53_mpp):
    origin = (float(x * he_mpp[0]), float(y * he_mpp[1]))
    field = sitk.TransformToDisplacementField(
        tx, sitk.sitkVectorFloat64, [int(width), int(height)], origin, tuple(he_mpp),
    )
    disp = sitk.GetArrayFromImage(field)
    yy, xx = np.indices((height, width))
    return ((xx + x) * he_mpp[0] + disp[..., 0]) / p53_mpp[0], ((yy + y) * he_mpp[1] + disp[..., 1]) / p53_mpp[1]


def warp_patch(slide, tx, x, y, width, height, he_mpp, p53_mpp, max_source_size=8192):
    mx, my = mapped_grid(tx, x, y, width, height, he_mpp, p53_mpp)
    if not np.isfinite(mx).all() or not np.isfinite(my).all():
        raise ValueError("Nonfinite registration mapping")
    valid = (mx >= 0) & (my >= 0) & (mx < slide.dimensions[0] - 1) & (my < slide.dimensions[1] - 1)
    if not valid.any():
        return np.full((height, width, 3), 255, dtype="uint8"), valid
    x0, y0 = int(np.floor(mx[valid].min())), int(np.floor(my[valid].min()))
    x1, y1 = int(np.ceil(mx[valid].max())) + 2, int(np.ceil(my[valid].max())) + 2
    if max(x1 - x0, y1 - y0) > max_source_size:
        raise ValueError("Mapped tile exceeds memory bound; decrease --tile-size or check transform")
    rgba = np.asarray(slide.read_region((x0, y0), 0, (x1 - x0, y1 - y0)))
    mapx, mapy = (mx - x0).astype("float32"), (my - y0).astype("float32")
    rgb = cv2.remap(rgba[..., :3], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    alpha = cv2.remap(rgba[..., 3], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid &= alpha == 255
    rgb[~valid] = 255
    return rgb, valid


def roi_offset(frame):
    # Integer bounding boxes are exact; rounded centroids are only a fallback.
    if {"bbox_minc_wsi", "bbox_minc_roi", "bbox_minr_wsi", "bbox_minr_roi"} <= set(frame.columns):
        offsets = np.column_stack([frame.bbox_minc_wsi - frame.bbox_minc_roi, frame.bbox_minr_wsi - frame.bbox_minr_roi])
    else:
        offsets = frame[["cx_wsi", "cy_wsi"]].to_numpy() - frame[["cx_roi", "cy_roi"]].to_numpy()
    median = np.median(offsets, axis=0)
    if not np.isfinite(offsets).all() or np.max(np.abs(offsets - median)) > .1:
        raise ValueError("Inconsistent ROI origin; expected Script02 level-0 ROI masks")
    return np.rint(median).astype(int)


def validate_review(review):
    threshold = review.get("dab_threshold")
    if threshold is not None and (not np.isfinite(float(threshold)) or float(threshold) <= 0):
        raise ValueError("dab_threshold must be positive in skimage rgb2hed DAB units")
    lo, hi = float(review.get("negative_fraction", .1)), float(review.get("positive_fraction", .6))
    if not 0 <= lo < hi <= 1:
        raise ValueError("Require 0 <= negative_fraction < positive_fraction <= 1")
    if review.get("registration_approved") is True and not str(review.get("reviewer", "")).strip():
        raise ValueError("Set reviewer when approving registration")


def weak_labels(frame, review, region_review, section_relationship, allow_serial=False, max_error_um=2.):
    """Return staining targets, never automatic tumor labels. -1 means unknown."""
    labels = np.full(len(frame), -1, dtype=int)
    if review.get("registration_approved") is not True or review.get("dab_threshold") is None:
        return labels
    if section_relationship != "same_section" and not (section_relationship == "serial_section" and allow_serial):
        return labels
    if region_review is None:
        return labels
    approved = region_review.get("approved", False)
    if str(approved).lower() not in ("true", "1"):
        return labels
    error = float(region_review.get("max_registration_error_um", float("nan")))
    if not np.isfinite(error) or not 0 <= error <= max_error_um:
        return labels
    # Tissue support and geometric coverage are separate; missing tissue is unknown.
    eligible = (frame.p53_mapped_fraction >= .95) & (frame.p53_tissue_fraction >= .5)
    eligible &= frame.p53_equivalent_radius_um > error
    labels[eligible & (frame.p53_positive_fraction >= float(review["positive_fraction"]))] = 1
    labels[eligible & (frame.p53_positive_fraction <= float(review["negative_fraction"]))] = 0
    return labels


def measure_region(slide, tx, mask, offset, he_mpp, p53_mpp, threshold, tile_size, qc_dir, he_slide=None):
    unique = np.unique(mask)
    unique = unique[unique > 0]
    stats = np.zeros((len(unique) + 1, 6), dtype=float)
    # Accumulate whole nuclei even when a nucleus crosses a tile boundary.
    previews = 0
    for y in range(0, mask.shape[0], tile_size):
        for x in range(0, mask.shape[1], tile_size):
            tile = mask[y:y + tile_size, x:x + tile_size]
            if not np.any(tile):
                continue
            rgb, valid = warp_patch(slide, tx, x + offset[0], y + offset[1], tile.shape[1], tile.shape[0], he_mpp, p53_mpp)
            dab = rgb2hed(rgb)[..., 2]
            tissue = tissue_mask(rgb)
            fg = tile > 0
            ids = np.searchsorted(unique, tile[fg]) + 1
            mapped, supported = valid[fg], (valid & tissue)[fg]
            values = dab[fg]
            weights = [np.ones(len(ids)), mapped, supported, values * mapped, values ** 2 * mapped,
                       (values >= threshold) & mapped if threshold is not None else np.zeros(len(ids))]
            for i, w in enumerate(weights):
                stats[:, i] += np.bincount(ids, weights=np.asarray(w, dtype=float), minlength=len(stats))
            # A bounded number of full-resolution QC tiles per ROI.
            if previews < 4:
                boundary = fg & (cv2.erode(fg.astype("uint8"), np.ones((3, 3), "uint8")) == 0)
                display = rgb.copy()
                display[boundary] = (255, 0, 0)
                if he_slide is not None:
                    he_rgb = np.array(he_slide.read_region((int(x + offset[0]), int(y + offset[1])), 0,
                                                          (tile.shape[1], tile.shape[0])).convert("RGB"))
                    he_rgb[boundary] = (255, 0, 0)
                    display = np.concatenate([he_rgb, display], axis=1)
                Image.fromarray(display).save(qc_dir / f"p53_x{x}_y{y}.png")
                previews += 1
    count, mapped, tissue, total, squares, positive = stats[1:].T
    denominator = np.where(mapped > 0, mapped, np.nan)
    mean = total / denominator
    return pd.DataFrame({"cell_label": unique.astype(int), "p53_dab_mean": mean,
                         "p53_dab_std": np.sqrt(np.maximum(0, squares / denominator - mean ** 2)),
                         "p53_positive_fraction": positive / denominator if threshold is not None else np.nan,
                         "p53_mapped_fraction": mapped / count, "p53_tissue_fraction": tissue / np.maximum(mapped, 1),
                         "p53_equivalent_radius_um": np.sqrt(count * np.prod(he_mpp) / np.pi)})


def extract_pair(pair, out_root, cellvit_root, tile_size=512, allow_serial=False, max_error_um=2.):
    out, root = Path(out_root) / pair["pair_id"], Path(cellvit_root)
    meta = json.loads((out / "registration.json").read_text())
    if meta["pair_signature"] != pair_signature(pair):
        raise ValueError("Slide pair or source files changed; register again")
    review = json.loads((out / "review.json").read_text())
    validate_review(review)
    if review.get("registration_approved") is True and (meta["qc"]["tissue_dice"] < .5 or meta["qc"]["nonpositive_jacobian_fraction"] > 0):
        raise ValueError("Registration failed geometric QC; fix the registration before approving")
    slide_name = meta["slide"]
    coords = pd.read_csv(root / "spatial_data" / f"{slide_name}_cell_coords.csv", dtype={"slide": str, "region": str})
    if coords.empty or coords[KEYS].isna().any().any() or coords.duplicated(KEYS).any() or set(coords.slide) != {slide_name}:
        raise ValueError("Coordinate file is empty or has invalid/duplicate cell keys")
    rr_path = out / "region_review.csv"
    if not rr_path.exists():
        region_template = pd.DataFrame({"region": sorted(coords.region.unique()), "approved": False,
                                       "max_registration_error_um": np.nan, "notes": ""})
        if pair.get("qc_landmarks"):
            residuals = pd.read_csv(out / "qc_landmark_residuals.csv", dtype={"region": str})
            if "region" in residuals:
                groups = residuals.groupby("region").registration_error_um.agg(["count", "max"])
                errors = groups.loc[groups["count"] >= 3, "max"]
                region_template["max_registration_error_um"] = region_template.region.map(errors)
        region_template.to_csv(rr_path, index=False)
    rr = pd.read_csv(rr_path, dtype={"region": str})
    if rr.region.duplicated().any():
        raise ValueError("Duplicate regions in region_review.csv")
    rr = rr.set_index("region")
    tx = sitk.ReadTransform(str(out / "he_to_p53.h5"))
    frames = []
    with openslide.OpenSlide(pair["p53_slide"]) as p53, openslide.OpenSlide(pair["he_slide"]) as he:
        for region, frame in coords.groupby("region", sort=True):
            if Path(region).name != region or "/" in region or "\\" in region:
                raise ValueError("Region names cannot contain path separators")
            mask = tifffile.imread(root / "region_mask" / f"mask_{slide_name}_{region}.tif")
            if mask.ndim != 2 or mask.dtype.kind not in "iu" or np.any(mask < 0):
                raise ValueError("Expected a two-dimensional nonnegative integer nuclear mask")
            qc_dir = out / "region_qc" / region
            qc_dir.mkdir(parents=True, exist_ok=True)
            measurements = measure_region(p53, tx, mask, roi_offset(frame), np.array(meta["he_mpp"]),
                                          np.array(meta["p53_mpp"]), review.get("dab_threshold"), tile_size, qc_dir, he)
            measurements = frame[KEYS + ["cx_wsi", "cy_wsi"]].merge(measurements, on="cell_label", how="left", validate="one_to_one")
            measurements["p53_label"] = weak_labels(measurements, review, rr.loc[region].to_dict() if region in rr.index else None,
                                                    meta["section_relationship"], allow_serial, max_error_um)
            measurements["patient_id"] = pair["patient_id"]
            measurements["pair_id"] = pair["pair_id"]
            measurements["registration_id"] = meta["registration_id"]
            measurements["section_relationship"] = meta["section_relationship"]
            frames.append(measurements)
            print(f"Measured {slide_name} / {region}: {len(measurements)} nuclei", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out / "p53_cell_measurements.csv", index=False)
    # Preserve existing human annotations when measurements are regenerated.
    template_path = out / "cell_review_template.csv"
    if not template_path.exists():
        template = result.copy()
        template["tumor"] = -1
        template["p53_positive"] = -1
        template["reviewer"] = ""
        template.to_csv(template_path, index=False)
    write_json(out / "extraction.json", {"registration_id": meta["registration_id"], "review": review,
                                        "region_review_sha256": hashlib.sha256(rr_path.read_bytes()).hexdigest(),
                                        "allow_serial_weak_labels": allow_serial, "max_error_um": max_error_um,
                                        "cells": len(result), "labeled_cells": int((result.p53_label >= 0).sum())})
    return result
