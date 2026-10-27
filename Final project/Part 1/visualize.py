"""Draws annotations or predictions of Part 1 over the images.

    python visualize.py --images fruits_part1/train/images --coco fruits_part1/train/annotations.json --out vis/
    python visualize.py --images fruits_part1/train/images --pred predictions.json \
        --images-json fruits_part1/train/images.json --min-score 0.5 --out vis_pred/

Every fruit gets its own color, with its outline and a label (species, and the
score for predictions). Annotations with iscrowd = 1 (ignored fragments) are
drawn in white. Use --ids or --limit to draw only some images, and --no-labels
to hide the labels.
"""

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils


def progress(action, k, n, item=""):
    """Shows `action k/n: item` on a single line that is updated in place."""
    print(f"\r{action} {k}/{n}" + (f": {item}" if item else "").ljust(40), end="\n" if k == n else "", flush=True)


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 1.0)
    return np.array([r, g, b]) * 255


def draw(rgb, items, names, show_labels=True):
    """items: list of (rle, category_id, score or None, ignored)."""
    out = rgb.astype(np.float32) * 0.75
    labels = []
    for k, (rle, cat, score, ignored) in enumerate(items):
        # Work only inside the bounding box of the mask (with a 1-pixel margin): much faster.
        x, y, w, h = (int(v) for v in mask_utils.toBbox(rle))
        if w == 0 or h == 0:
            continue
        x0, y0 = max(x - 1, 0), max(y - 1, 0)
        x1, y1 = min(x + w + 1, out.shape[1]), min(y + h + 1, out.shape[0])
        m = mask_utils.decode(rle)[y0:y1, x0:x1].astype(bool)
        crop = out[y0:y1, x0:x1]
        c = np.array([255.0, 255, 255]) if ignored else color(k + 1)
        crop[m] = 0.4 * crop[m] + 0.6 * c
        inner = np.zeros_like(m)
        inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        crop[m & ~inner] = c
        if show_labels and not ignored:
            ys, xs = np.nonzero(m)
            text = names[cat] + (f" {score:.2f}" if score is not None else "")
            labels.append((x0 + int(xs.mean()), y0 + int(ys.min()), text))
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(13)
    for x, y, text in labels:
        box = d.textbbox((x, y - 3), text, font=font, anchor="mb")
        d.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(255, 255, 255))
        d.text((x, y - 3), text, font=font, fill=(0, 0, 0), anchor="mb")
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--images", type=Path, required=True, help="folder with the images")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--coco", type=Path, help="annotations (annotations.json)")
    src.add_argument("--pred", type=Path, help="predictions file (needs --images-json)")
    p.add_argument("--images-json", type=Path, help="images.json, to know the file of every image_id")
    p.add_argument("--min-score", type=float, default=0.0, help="hide predictions below this score")
    p.add_argument("--ids", type=int, nargs="+", help="draw only these image ids")
    p.add_argument("--limit", type=int, help="draw at most this many images")
    p.add_argument("--no-labels", action="store_true", help="do not draw the labels (species and score)")
    p.add_argument("--out", type=Path, required=True, help="output folder")
    args = p.parse_args()

    if args.coco:
        data = json.loads(args.coco.read_text())
        dets = [dict(a, score=None) for a in data["annotations"]]
    else:
        if not args.images_json:
            p.error("--pred needs --images-json")
        data = json.loads(args.images_json.read_text())
        dets = [dict(d, iscrowd=0) for d in json.loads(args.pred.read_text()) if d["score"] >= args.min_score]
    names = {c["id"]: c["name"] for c in data["categories"]}
    by_image = {}
    for d in dets:
        by_image.setdefault(d["image_id"], []).append(d)
    images = [im for im in data["images"] if args.ids is None or im["id"] in args.ids][:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(images, 1):
        progress("Drawing image", k, len(images), im["file_name"])
        rgb = np.asarray(Image.open(args.images / im["file_name"]).convert("RGB"))
        items = [(d["segmentation"], d["category_id"], d["score"], bool(d["iscrowd"]))
                 for d in sorted(by_image.get(im["id"], []), key=lambda d: -(d["score"] or 0))]
        draw(rgb, items, names, not args.no_labels).save(args.out / im["file_name"], quality=90)
    print(f"{len(images)} images drawn in {args.out}")


if __name__ == "__main__":
    main()
