"""Scores a predictions file of Part 1 with the official metrics.

    python evaluate.py --gt fruits_part1/train/annotations.json --pred predictions.json
    python evaluate.py --gt ... --pred ... --images 3 17 42      # only these images (e.g. a validation split)
    python evaluate.py --gt ... --pred ... --images-file val_ids.txt --out metrics.json

The metrics are those of COCO instance segmentation (pycocotools, masks), with
up to 300 detections per image:

    AP            mask AP averaged over IoU 0.50:0.05:0.95 and over the categories (main metric)
    AP50, AP75    mask AP at IoU 0.50 and 0.75
    AP_<species>  mask AP of each category
    AP_agnostic   mask AP with all categories merged into "fruit"
    accuracy      among the detections that overlap a fruit with IoU >= 0.5, the fraction with its species
    AP_small, AP_medium, AP_large   by visible area: < 32^2, 32^2 - 96^2, > 96^2 pixels

Annotations with iscrowd = 1 (fragments of fewer than 20 visible pixels) are
ignored: detecting them gives no credit and missing them is not penalized.
This is the same script we use to score the test set.
"""

import argparse
import contextlib
import copy
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = 300


class InvalidPredictions(ValueError):
    pass


def progress(action, k, n, item=""):
    """Shows `action k/n: item` on a single line that is updated in place."""
    print(f"\r{action} {k}/{n}" + (f": {item}" if item else "").ljust(40), end="\n" if k == n else "", flush=True)


def load_gt(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def validate(preds, gt):
    """Checks the predictions against the images and categories of the ground truth.

    Returns the predictions to score (at most MAX_DETS per image, the best ones)
    and a list of warnings. Raises InvalidPredictions if the file breaks the format.
    """
    if not isinstance(preds, list):
        raise InvalidPredictions("the file must contain a JSON list of detections")
    images = {i["id"]: i for i in gt.dataset["images"]}
    categories = {c["id"] for c in gt.dataset["categories"]}
    warnings = []
    for k, p in enumerate(preds):
        where = f"detection {k}"
        if not isinstance(p, dict):
            raise InvalidPredictions(f"{where}: must be a JSON object")
        for key in ("image_id", "category_id", "score", "segmentation"):
            if key not in p:
                raise InvalidPredictions(f"{where}: missing '{key}'")
        if p["image_id"] not in images:
            raise InvalidPredictions(f"{where}: unknown image_id {p['image_id']}")
        if p["category_id"] not in categories:
            raise InvalidPredictions(f"{where}: unknown category_id {p['category_id']}")
        if not isinstance(p["score"], (int, float)) or not 0.0 <= p["score"] <= 1.0:
            raise InvalidPredictions(f"{where}: score must be a number in [0, 1]")
        seg = p["segmentation"]
        if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
            raise InvalidPredictions(f"{where}: segmentation must be an RLE dict with 'size' and 'counts'")
        im = images[p["image_id"]]
        if list(seg["size"]) != [im["height"], im["width"]]:
            raise InvalidPredictions(f"{where}: segmentation size {seg['size']} is not [height, width] "
                                     f"= [{im['height']}, {im['width']}] of image {im['id']}")
        if not isinstance(seg["counts"], str):
            raise InvalidPredictions(f"{where}: segmentation counts must be a string (decode the bytes of "
                                     "mask_utils.encode with .decode('ascii'))")
    per_image = defaultdict(list)
    for p in preds:
        per_image[p["image_id"]].append(p)
    kept = []
    for image_id, dets in per_image.items():
        if len(dets) > MAX_DETS:
            warnings.append(f"image {image_id}: {len(dets)} detections, only the best {MAX_DETS} are scored")
            dets = sorted(dets, key=lambda d: -d["score"])[:MAX_DETS]
        kept += dets
    return kept, warnings


def _coco_eval(gt, preds, img_ids, use_cats=True):
    """COCOeval on masks; returns the precision array [T, R, K, A, M] (or None without detections)."""
    if not preds:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(preds))
        ev = COCOeval(gt, dt, "segm")
        ev.params.imgIds = img_ids
        ev.params.maxDets = [1, 100, MAX_DETS]
        ev.params.useCats = int(use_cats)
        ev.evaluate()
        ev.accumulate()
    return ev.eval["precision"]


def _ap(precision, t=None, k=None, a=0):
    """Mean of the valid precision values; t: IoU threshold index, k: category index, a: area range."""
    if precision is None:
        return 0.0
    p = precision[:, :, :, a, -1]           # maxDets = MAX_DETS
    if t is not None:
        p = p[t:t + 1]
    if k is not None:
        p = p[:, :, k:k + 1]
    p = p[p > -1]
    return float(p.mean()) if p.size else float("nan")


def category_accuracy(gt, preds, img_ids):
    """Among the detections whose best IoU with a (non-ignored) fruit is >= 0.5, the fraction with its category."""
    by_image = defaultdict(list)
    for p in preds:
        by_image[p["image_id"]].append(p)
    matched = correct = 0
    for image_id in img_ids:
        dets = by_image.get(image_id, [])
        anns = [a for a in gt.imgToAnns.get(image_id, []) if not a["iscrowd"]]
        if not dets or not anns:
            continue
        ious = mask_utils.iou([d["segmentation"] for d in dets], [a["segmentation"] for a in anns], [0] * len(anns))
        best = np.argmax(ious, axis=1)
        for d, j, iou in zip(dets, best, ious[np.arange(len(dets)), best]):
            if iou >= 0.5:
                matched += 1
                correct += int(d["category_id"] == anns[j]["category_id"])
    return correct / matched if matched else float("nan")


def evaluate(gt, preds, img_ids=None, verbose=False):
    """All the metrics, as a dict. preds must have been validated. verbose: show the progress."""
    img_ids = sorted(img_ids) if img_ids is not None else sorted(gt.getImgIds())
    keep = set(img_ids)
    preds = [p for p in preds if p["image_id"] in keep]
    if verbose:
        progress("Evaluating step", 1, 3, "mask AP")
    precision = _coco_eval(gt, preds, img_ids)
    m = {"AP": _ap(precision), "AP50": _ap(precision, t=0), "AP75": _ap(precision, t=5)}
    for k, c in enumerate(sorted(gt.dataset["categories"], key=lambda c: c["id"])):
        m[f"AP_{c['name']}"] = _ap(precision, k=k)
    if verbose:
        progress("Evaluating step", 2, 3, "class-agnostic AP")
    m["AP_agnostic"] = _ap(_coco_eval(gt, preds, img_ids, use_cats=False))
    if verbose:
        progress("Evaluating step", 3, 3, "category accuracy")
    m["accuracy"] = category_accuracy(gt, preds, img_ids)
    for a, name in [(1, "small"), (2, "medium"), (3, "large")]:
        m[f"AP_{name}"] = _ap(precision, a=a)
    m["images"] = len(img_ids)
    m["detections"] = len(preds)
    m["fruits"] = sum(1 for i in img_ids for a in gt.imgToAnns.get(i, []) if not a["iscrowd"])
    return m


def format_metrics(m, indent=2):
    rows = [(k, f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in m.items()]
    w = max(len(k) for k, _ in rows)
    return "\n".join(f"{' ' * indent}{k:<{w}}  {v}" for k, v in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gt", type=Path, required=True, help="COCO annotations (annotations.json)")
    p.add_argument("--pred", type=Path, required=True, help="predictions file (COCO results format)")
    p.add_argument("--images", type=int, nargs="+", help="score only these image ids")
    p.add_argument("--images-file", type=Path, help="text file with the image ids to score, one per line")
    p.add_argument("--out", type=Path, help="write the metrics to this JSON file")
    args = p.parse_args()

    gt = load_gt(args.gt)
    img_ids = args.images
    if args.images_file:
        img_ids = [int(x) for x in args.images_file.read_text().split()]
    if img_ids is not None:
        unknown = set(img_ids) - set(gt.getImgIds())
        if unknown:
            sys.exit(f"error: unknown image ids {sorted(unknown)[:10]}")
    try:
        preds, warnings = validate(json.loads(args.pred.read_text()), gt)
    except (InvalidPredictions, json.JSONDecodeError) as e:
        sys.exit(f"error: invalid predictions file: {e}")
    for w in warnings:
        print("warning:", w)
    m = evaluate(gt, preds, img_ids, verbose=True)
    print(format_metrics(m))
    if args.out:
        args.out.write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
