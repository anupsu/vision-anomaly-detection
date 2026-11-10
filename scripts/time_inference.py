"""Time White Cap inference per image, with the run's settings: locate the cap, crop + resize,
features, nearest-patch scoring against the memory bank. The bank is built once, as in the run.

Usage (from the repo root):
    python scripts/time_inference.py                    # data/WhiteCap
    python scripts/time_inference.py --data path/to/WhiteCap
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patchcore_score import DiscLocator, PatchCore, get_device, list_images, open_image  # noqa: E402

SIZE, ANGLES, CORESET, MARGIN = (512, 512), [45 * k for k in range(8)], 0.1, 0.04


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/WhiteCap", type=Path, help="folder with good/ and all/")
    args = ap.parse_args()

    device = get_device()
    sync = (lambda: torch.mps.synchronize()) if device.type == "mps" else (
        (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None))
    loc = DiscLocator(MARGIN)
    boxes = {}
    model = PatchCore(SIZE, device, CORESET, lambda img, path: boxes[path])

    def bg_mask(p, grid):
        bg = loc.background(p, boxes[p])
        m = torch.from_numpy(bg.astype(np.float32))[None, None]
        return F.adaptive_avg_pool2d(m, tuple(grid))[0, 0].numpy() >= 0.9

    train = list_images(args.data / "good")
    for p in train:
        boxes[p] = loc(open_image(p), p)
    feats, masks = [], []
    for p in train:
        for a in ANGLES:
            feats.append(model.features(p, a))
        masks += [bg_mask(p, feats[-1].shape[1:])] * len(ANGLES)
    t = time.time()
    bank = model.build_bank(feats, masks)
    print(f"bank {bank.shape[0]} patches built in {time.time() - t:.0f}s (one-off, not part of inference)")

    train_names = {q.name for q in train}
    test = [p for p in list_images(args.data / "all") if p.name not in train_names]
    rows = []
    for i, p in enumerate(test):
        sync()
        t0 = time.perf_counter()
        img = open_image(p)
        t1 = time.perf_counter()
        boxes[p] = loc(img, p)
        t2 = time.perf_counter()
        f = model.features(p)  # crop + resize + normalise + backbone (re-reads the file, as the run does)
        sync()
        t3 = time.perf_counter()
        model.score(f, bank, bg_mask(p, f.shape[1:]))
        sync()
        t4 = time.perf_counter()
        rows.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))
        if i == 0:
            print(f"first image (warm-up): {1000 * (t4 - t0):.0f} ms")

    r = np.array(rows[1:]) * 1000  # skip the warm-up image
    names = ["read image", "locate cap", "crop+resize+features", "score vs bank", "TOTAL"]
    print(f"\n{len(r)} images after warm-up, device {device}, bank {bank.shape[0]} patches")
    for n, col in zip(names, r.T):
        print(f"  {n:22s} mean {col.mean():6.0f} ms   median {np.median(col):6.0f}   "
              f"min {col.min():6.0f}   max {col.max():6.0f}")


if __name__ == "__main__":
    main()
