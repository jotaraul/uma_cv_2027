"""Example submission for Part 1: a deliberately naive color-threshold detector.

    python main.py detect --input <input folder> --output <predictions file> --device cuda

It shows the format of a submission (command line, input, output), not a
method: it marks the saturated, warm-colored pixels as fruit, splits them into
connected regions and guesses the species from the mean hue of each region.
Expect a very low score.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage

# Species by mean hue (degrees) of a region: red apples, orange oranges, yellow lemons, yellow-green pears.
HUE_RANGES = [(2, 0, 14), (1, 14, 38), (4, 38, 52), (3, 52, 70)]     # (category_id, from, to)
MIN_AREA = 40


def detect(image_path):
    """Returns a list of (mask, category_id, score) for one image."""
    hsv = np.asarray(Image.open(image_path).convert("HSV"), np.float32) / 255.0
    hue, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    hue = np.where(hue > 340.0, hue - 360.0, hue)                      # reds near 360 -> near 0
    fruit = (sat > 0.5) & (val > 0.25) & (hue > -20.0) & (hue < 70.0)
    fruit = ndimage.binary_opening(fruit, iterations=1)
    labels, n = ndimage.label(fruit)
    detections = []
    for region, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = np.zeros(labels.shape, bool)
        mask[sl] = labels[sl] == region
        if mask.sum() < MIN_AREA:
            continue
        h = float(np.median(np.clip(hue[mask], 0.0, None)))
        category = next((c for c, lo, hi in HUE_RANGES if lo <= h < hi), 1)
        score = float(np.clip(sat[mask].mean(), 0.0, 1.0))
        detections.append((mask, category, score))
    return detections


def progress(action, k, n, item=""):
    """Shows `action k/n: item` on a single line that is updated in place."""
    print(f"\r{action} {k}/{n}" + (f": {item}" if item else "").ljust(40), end="\n" if k == n else "", flush=True)


def to_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--input", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--device", default="cuda", choices=["cuda", "cpu"])   # unused: this method runs on the CPU
    args = p.parse_args()

    images = json.loads((args.input / "images.json").read_text())["images"]
    results = []
    for k, im in enumerate(images, 1):
        progress("Detecting fruits in image", k, len(images), im["file_name"])
        dets = detect(args.input / "images" / im["file_name"])
        dets = sorted(dets, key=lambda x: -x[2])[:300]                    # at most 300 detections per image
        for mask, category_id, score in dets:
            results.append({"image_id": im["id"], "category_id": category_id, "score": score,
                            "segmentation": to_rle(mask)})
    args.output.write_text(json.dumps(results))
    print(f"{len(results)} detections written to {args.output}")


if __name__ == "__main__":
    main()
