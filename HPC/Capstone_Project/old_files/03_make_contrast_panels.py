from pathlib import Path
import json
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile


ROOT = Path(__file__).resolve().parent
NDPI_DIR = ROOT / "NDPIimage"
POLYEPI_DIR = ROOT / "polyepi_L2"
RGB_TILE_DIR = ROOT / "cellpose_output" / "rgb"
STITCHED_DIR = ROOT / "cellpose_output" / "stitched"
OUTPUT_DIR = ROOT / "cellpose_output" / "contrast"

LABEL_H = 80
PADDING = 24
BG_COLOR = (248, 248, 248)
TEXT_COLOR = (25, 25, 25)


def load_font(size=36):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def normalize_rgb(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    low, high = np.nanpercentile(arr, [1, 99])
    if high <= low:
        high = low + 1
    return (np.clip((arr - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def safe_filename(text):
    return re.sub(r'[<>:"/\\|?*]+', "_", text)


def collect_regions():
    regions = {}
    for path in POLYEPI_DIR.glob("polyepi_*.json"):
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        slide_base = Path(meta["slide"]).stem
        region_id = meta["region_id"]
        key = f"{slide_base}_{region_id}"
        regions[key] = {
            "json": path,
            "slide_base": slide_base,
            "slide_path": NDPI_DIR / meta["slide"],
            "level": int(meta["level"]),
            "min_x": int(meta["min_x"]),
            "min_y": int(meta["min_y"]),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
        }
    return regions


def collect_stitched_pairs():
    huang3 = {}
    modified = {}
    for path in STITCHED_DIR.glob("stitched_*.tif"):
        key = path.stem.removeprefix("stitched_")
        if key.endswith("_MOD"):
            modified[key.removesuffix("_MOD")] = path
        else:
            huang3[key] = path
    return huang3, modified


def read_original_from_ndpi(meta, target_shape):
    try:
        import openslide
    except ImportError:
        return None

    slide_path = meta["slide_path"]
    if not slide_path.exists():
        return None

    scale = 2 ** meta["level"]
    x0 = meta["min_x"] * scale
    y0 = meta["min_y"] * scale
    width = meta["width"] * scale
    height = meta["height"] * scale

    target_h, target_w = target_shape[:2]
    if (height, width) != (target_h, target_w):
        width, height = target_w, target_h

    slide = openslide.OpenSlide(str(slide_path))
    try:
        original = slide.read_region((x0, y0), 0, (width, height)).convert("RGB")
        return np.asarray(original, dtype=np.uint8)
    finally:
        slide.close()


def tile_regex_for_key(key):
    escaped = re.escape(key)
    return re.compile(rf"^rgb_{escaped}_(\d+)_(\d+)\.tif$", re.IGNORECASE)


def read_original_from_rgb_tiles(key, meta, target_shape):
    target_h, target_w = target_shape[:2]
    original = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = 2 ** meta["level"]
    minx0 = meta["min_x"] * scale
    miny0 = meta["min_y"] * scale

    pattern = tile_regex_for_key(key)
    found = False
    for path in RGB_TILE_DIR.glob(f"rgb_{key}_*.tif"):
        match = pattern.match(path.name)
        if not match:
            continue

        x = int(match.group(1))
        y = int(match.group(2))
        sx = x - minx0
        sy = y - miny0
        if sx >= target_w or sy >= target_h:
            continue

        tile = normalize_rgb(tifffile.imread(path))
        th = min(tile.shape[0], target_h - sy)
        tw = min(tile.shape[1], target_w - sx)
        if th <= 0 or tw <= 0:
            continue

        original[sy : sy + th, sx : sx + tw] = tile[:th, :tw]
        found = True

    return original if found else None


def read_original(key, meta, target_shape):
    original = read_original_from_ndpi(meta, target_shape)
    if original is not None:
        return original

    original = read_original_from_rgb_tiles(key, meta, target_shape)
    if original is not None:
        return original

    raise RuntimeError(f"Could not build original image for {key}")


def draw_centered_label(draw, x, y, width, height, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        (x + (width - text_w) // 2, y + (height - text_h) // 2),
        text,
        fill=TEXT_COLOR,
        font=font,
    )


def make_contrast_tif(key, original, huang3, modified):
    h, w = huang3.shape[:2]
    labels = ["Original", "Tile-Based Processing", "Whole Image Processing"]
    panels = [original, huang3, modified]

    canvas_w = PADDING * 4 + w * 3
    canvas_h = PADDING * 2 + LABEL_H + h
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    y0 = PADDING + LABEL_H
    for i, panel in enumerate(panels):
        x0 = PADDING + i * (w + PADDING)
        canvas[y0 : y0 + h, x0 : x0 + w] = panel

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    font = load_font()
    for i, label in enumerate(labels):
        x0 = PADDING + i * (w + PADDING)
        draw_centered_label(draw, x0, PADDING, w, LABEL_H, label, font)

    out_path = OUTPUT_DIR / f"contrast_{safe_filename(key)}.tif"
    tifffile.imwrite(out_path, np.asarray(image), photometric="rgb")
    return out_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    regions = collect_regions()
    huang3_paths, modified_paths = collect_stitched_pairs()
    keys = sorted(set(regions) & set(huang3_paths) & set(modified_paths))
    if not keys:
        raise SystemExit("No complete Original/Huang3/Huang3_MODIFIED triplets found.")

    created = []
    for key in keys:
        huang3 = normalize_rgb(tifffile.imread(huang3_paths[key]))
        modified = normalize_rgb(tifffile.imread(modified_paths[key]))

        if huang3.shape != modified.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: Huang3={huang3.shape}, MOD={modified.shape}"
            )

        original = read_original(key, regions[key], huang3.shape)
        if original.shape != huang3.shape:
            raise RuntimeError(
                f"Original shape mismatch for {key}: original={original.shape}, "
                f"stitched={huang3.shape}"
            )

        created.append(make_contrast_tif(key, original, huang3, modified))

    print(f"Created {len(created)} native-resolution TIFF contrast images:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
