"""Score one outside photo (e.g. a colour phone photo) with a White Cap run.

Preprocessing, to make it look like the mono camera's crops the model was trained on:
  1. grayscale (then 3 identical channels, as the camera images are loaded)
  2. find the cap's rim with a Hough circle (DiscLocator needs a dark background)
  3. square crop around the circle with the run's 4% margin, everything outside the circle masked
  4. match the cap's grey levels to the good training caps (histogram matching inside the circle)
  5. resize to 512x512 and ImageNet-normalise, score against the same memory bank (the good caps
     x 8 rotations, coreset 0.1) and the run's threshold

The memory bank is rebuilt from the good images (about 2 minutes), with the run's settings.

Usage (from the repo root):
    python scripts/score_image.py --image path/to/photo.jpg
    python scripts/score_image.py --image photo.jpg --data data/WhiteCap --run results/WhiteCap

Writes <run>/<image name>/preprocessed.png (crop before/after) and heatmap.png.
A photo taken differently from the training images (other camera, lighting, background or
angle) can score high just for looking different; see the README.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patchcore_score import DiscLocator, PatchCore, get_device, list_images, open_image, save_heatmap  # noqa: E402

SIZE, ANGLES, CORESET, MARGIN = (512, 512), [45 * k for k in range(8)], 0.1, 0.04


def circle_mask(n, frac):
    """Disc of radius frac*n/2 in an n x n square (the crop's cap area)."""
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    return (yy - c) ** 2 + (xx - c) ** 2 <= (frac * n / 2) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--data", default="data/WhiteCap", type=Path, help="folder with good/ (the training caps)")
    ap.add_argument("--run", default="results/WhiteCap", type=Path, help="results of patchcore_score.py (threshold)")
    args = ap.parse_args()
    out = args.run / args.image.stem
    out.mkdir(parents=True, exist_ok=True)
    threshold = json.loads((args.run / "run_info.json").read_text())["threshold"]

    # 1-2. grayscale, find the rim
    rgb = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    grey = np.asarray(rgb.convert("L"))
    s = 4  # search at 1/4 size
    small = cv2.medianBlur(cv2.resize(grey, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA), 5)
    short = min(small.shape)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=short, param1=120, param2=40,
                               minRadius=short // 8, maxRadius=short // 2)
    if circles is None:
        raise SystemExit(f"no round cap found in {args.image}")
    cx, cy, r = (circles[0, 0] * s).tolist()
    print(f"cap found: centre ({cx:.0f}, {cy:.0f}), radius {r:.0f}px in a {grey.shape[1]}x{grey.shape[0]} photo")

    # 3. square crop with the run's margin
    half = r * (1 + MARGIN)
    box = tuple(int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    crop = Image.fromarray(grey).crop(box)  # PIL pads outside the photo with black
    n = crop.size[0]
    disc = circle_mask(n, 1 / (1 + MARGIN))

    # 4. match grey levels inside the cap to the good training caps
    loc = DiscLocator(MARGIN)
    train = list_images(args.data / "good")
    ref_vals = []
    for p in train:
        b = loc(open_image(p), p)
        g = np.asarray(open_image(p).convert("L").crop(b).resize((n, n)))
        ref_vals.append(g[disc])
    ref = np.sort(np.concatenate(ref_vals))
    src = np.asarray(crop, dtype=np.uint8)
    vals = src[disc]
    matched_vals = np.empty_like(vals)
    matched_vals[np.argsort(vals)] = ref[np.linspace(0, len(ref) - 1, len(vals)).astype(int)]
    matched = src.copy()
    matched[disc] = matched_vals
    matched[~disc] = 0  # background is masked anyway; keep it neutral
    prep = Image.fromarray(matched).convert("RGB")
    side = np.hstack([np.asarray(crop.convert("RGB").resize((512, 512))), np.asarray(prep.resize((512, 512)))])
    Image.fromarray(side).save(out / "preprocessed.png")
    print(f"cap grey level: mean {vals.mean():.0f} -> {matched_vals.mean():.0f} (good caps {ref.mean():.0f}); "
          f"before/after -> {out / 'preprocessed.png'}")

    # 5. same memory bank as the run, then score
    device = get_device()
    boxes = {}
    model = PatchCore(SIZE, device, CORESET, lambda img, path: boxes[path])

    def bg(grid, mask):
        """Feature-grid mask: a patch is background if (nearly) all of it is."""
        m = torch.from_numpy(mask.astype(np.float32))[None, None]
        return F.adaptive_avg_pool2d(m, tuple(grid))[0, 0].numpy() >= 0.9

    feats, masks = [], []
    for p in train:
        boxes[p] = loc(open_image(p), p)
        for a in ANGLES:
            feats.append(model.features(p, a))
        masks += [bg(feats[-1].shape[1:], loc.background(p, boxes[p]))] * len(ANGLES)
    bank = model.build_bank(feats, masks)
    print(f"memory bank: {bank.shape[0]} patches (same settings as the run)")

    feat = model.extractor(model.tf(prep)[None].to(device))[0].half().cpu()
    outside = ~np.asarray(Image.fromarray(disc).resize(prep.size), dtype=bool)
    score, pmap = model.score(feat, bank, bg(feat.shape[1:], outside))
    run_scores = [float(line.split(",")[2]) for line in (args.run / "scores.csv").read_text().splitlines()[1:]]
    vmax = np.percentile(run_scores, 95)  # same colour scale as the run's heatmaps
    save_heatmap(prep, pmap, score, threshold, f"{args.image.name} (preprocessed)", out / "heatmap.png", vmax)
    print(f"\nscore {score:.3f}   threshold {threshold:.3f}   -> {'BAD' if score > threshold else 'GOOD'}")
    print(f"heatmap -> {out / 'heatmap.png'}")


if __name__ == "__main__":
    main()
