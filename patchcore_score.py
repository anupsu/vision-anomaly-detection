"""Train PatchCore on the good images, then score every image in all/.

PatchCore (Roth et al., 2022) stores patch features of good images from a
frozen ImageNet backbone in a memory bank. A test patch's anomaly score is its
distance to the nearest stored patch; the image score is the worst patch.

Threshold: with only a handful of good images there is no validation set, so
each good image is scored leave-one-out (bank built from the others) and the
highest of those scores is the threshold (override with --threshold). Images
above it are predicted bad.
Good images that are also in all/ are reported with their leave-one-out score,
since scoring them against a bank containing themselves would give ~0.

Usage:
    python patchcore_score.py                                  # Model1 defaults
    python patchcore_score.py --region ring                    # top ring only, near full resolution
    python patchcore_score.py --data image_data_patchcore/Model1 --size 768 1024

Part crop (--region part): only the part is scored, without the background, and
parts that sit at different places in the frame end up aligned. Draw the part's box
once in a reference good image (original pixels); in every image the part's top-left
corner is found and that box is placed relative to it. Check with --preview-crops:
    python patchcore_score.py --data image_data_patchcore/Model2 --region part \\
        --part-box 750 465 1935 1355 --out results/Model2_part --preview-crops
    python patchcore_score.py --data image_data_patchcore/Model2 --region part \\
        --part-box 750 465 1935 1355 --out results/Model2_part
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes, binary_opening, gaussian_filter, label, sobel,
                           uniform_filter1d)
from torchvision import models, transforms

from ring_unwrap import HEIGHT, OUTER, R_FROM, R_TO, SEAM, WIDTH, face_rows, find_hole, suppress_white, unwrap

IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def open_image(path):
    """Open an image the way it is meant to be seen: phone photos are often stored sideways with
    an EXIF rotation flag (ignoring it would score some photos on their side). No-op for BMPs."""
    return ImageOps.exif_transpose(Image.open(path))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FeatureExtractor(torch.nn.Module):
    """Wide-ResNet50 layer2 + layer3, 3x3 neighbourhood pooled, at layer2 resolution (stride 8)."""

    def __init__(self):
        super().__init__()
        net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # (B, 1536, H/8, W/8)


def greedy_coreset(patches, ratio, proj_dim=128, seed=0):
    """k-center greedy subsampling on randomly projected features (runs on CPU:
    thousands of tiny matvecs, where GPU launch overhead would dominate)."""
    n = patches.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return patches
    g = torch.Generator().manual_seed(seed)
    z = patches.float() @ (torch.randn(patches.shape[1], proj_dim, generator=g) / proj_dim ** 0.5)
    zz = (z * z).sum(1)
    idx = int(torch.randint(n, (1,), generator=g))
    selected, min_dist = [], torch.full((n,), float("inf"))
    for _ in range(k):
        selected.append(idx)
        min_dist = torch.minimum(min_dist, zz - 2 * (z @ z[idx]) + zz[idx])
        idx = int(torch.argmax(min_dist))
    return patches[torch.tensor(selected)]


def locate_ring(img):
    """Crop box (left, top, right, bottom) around the part's top ring, in original pixels.

    Works at 1/4 resolution. The part's body is darker than the background on
    both sides, so its silhouette edges are the strongest vertical edges in a
    band through the body; the static white panel on the right gives a second
    edge further out, so the right edge is the first strong one after the left.
    The ring's top edge (dark background above, bright rim below) is sampled in
    8 column chunks and fitted with a robust line, since the ring is tilted.
    """
    W, H = 512, 384
    a = gaussian_filter(np.asarray(img.convert("L").resize((W, H))).astype(float), 2)
    gx, gy = np.abs(sobel(a, axis=1)), sobel(a, axis=0)
    cp = gx[110:260].mean(0)
    left = int(np.argmax(cp[10:W - 250])) + 10
    seg = cp[left + 250:min(left + 400, W - 2)]
    peaks = [i for i in range(1, len(seg) - 1)
             if seg[i] > 0.4 * seg.max() and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    right = left + 250 + (peaks[0] if peaks else int(np.argmax(seg)))
    w = right - left
    xs, tops = [], []
    for k in range(8):
        c0 = left + int((0.05 + 0.85 * k / 8) * w)
        c1 = c0 + max(4, int(0.1 * w))
        xs.append((c0 + c1) / 2)
        tops.append(int(np.argmax(gy[15:160, c0:c1].mean(1))) + 15)
    xs, tops = np.array(xs), np.array(tops)
    slopes = [(tops[j] - tops[i]) / (xs[j] - xs[i]) for i in range(8) for j in range(i + 1, 8)]
    slope = float(np.clip(np.median(slopes), 0.0, 0.12))
    icpt = float(np.median(tops - slope * xs))
    y_l, y_r = icpt + slope * left, icpt + slope * right
    sx, sy = img.width / W, img.height / H
    return (int(max(0, left - 18) * sx), int(max(0, min(y_l, y_r) - 12) * sy),
            int(min(W, right + 22) * sx), int(min(H, max(y_l, y_r) + 42) * sy))


class PartLocator:
    """Crop box around the part, placed from its top-left corner found in each image.

    The part is darker than the lit table and the white panel but brighter than the
    black background to its left and above it, so (at 1/4 resolution):
      left edge = first column, in a band through the body, brighter than half-way
                  from the background to the body level
      top edge  = first row, in columns just inside the left edge, brighter than
                  half-way from the background to the body level there
    --part-box is the crop in the reference image; its offset from the reference's
    corner and its size are reused for every image, grown by `margin` on each side.
    So every crop has the same size and parts at different places end up aligned.
    (Template matching was tried first; this part is too dark and low-contrast for it.)
    """
    SCALE = 4

    def __init__(self, ref_path, box, margin=0.0):
        self.size = (box[2] - box[0], box[3] - box[1])
        self.pad = (round(margin * self.size[0]), round(margin * self.size[1]))
        cx, cy = self.corner(open_image(ref_path))
        self.offset = (box[0] - cx, box[1] - cy)
        self.last_contrast = None

    def corner(self, img):
        s = self.SCALE
        a = gaussian_filter(np.asarray(img.convert("L").reduce(s), dtype=float), 2)
        h, w = a.shape
        bg = np.median(a[:h // 6, :w // 6])  # the top-left corner of the frame is background
        col = uniform_filter1d(np.median(a[int(.35 * h):int(.65 * h)], 0), 5)
        x0 = int(np.argmax(col > bg + 0.5 * (np.percentile(col[:int(.8 * w)], 90) - bg)))
        strip = a[:, x0 + w // 40:x0 + w // 8]
        body = np.median(strip[int(.35 * h):int(.65 * h)])
        y0 = int(np.argmax(uniform_filter1d(np.median(strip, 1), 3) > bg + 0.5 * (body - bg)))
        self.last_contrast = float(body - bg)  # low = the part barely stands out: check that crop
        return x0 * s, y0 * s

    def __call__(self, img):
        cx, cy = self.corner(img)
        l, t = cx + self.offset[0] - self.pad[0], cy + self.offset[1] - self.pad[1]
        return l, t, l + self.size[0] + 2 * self.pad[0], t + self.size[1] + 2 * self.pad[1]

    def rim(self, crop):
        """Top rim's upper edge in a crop: (y at the crop's centre column, tilt in degrees).

        The rim is brighter than the dark background above it, so its edge is the strongest
        dark-to-bright step in the top 30% of the crop; it is sampled in 10 column chunks
        (skipping the ends) and fitted with a robust line, so a bent patch of rim barely moves it.
        """
        s = self.SCALE
        a = gaussian_filter(np.asarray(crop.convert("L").reduce(s), dtype=float), 1.5)
        h, w = a.shape
        gy = sobel(a, axis=0)[:int(.3 * h)]
        xs, ys = [], []
        for k in range(10):
            c0 = int((0.12 + 0.76 * k / 10) * w)
            c1 = c0 + max(3, int(0.06 * w))
            xs.append((c0 + c1) / 2)
            ys.append(int(np.argmax(gy[:, c0:c1].mean(1))))
        xs, ys = np.array(xs), np.array(ys)
        slope = float(np.median([(ys[j] - ys[i]) / (xs[j] - xs[i]) for i in range(10) for j in range(i + 1, 10)]))
        y_mid = float(np.median(ys - slope * xs)) + slope * w / 2
        return y_mid * s, float(np.degrees(np.arctan(slope)))


class SideLocator:
    """Tight crop of a cylindrical part seen from the side (CAM3/CAM4), from its top rim.

    The rim is the most reliable feature in both views: a bright line across exactly the
    part's width, with dark background above it. At 1/4 resolution its row is the strongest
    dark-to-bright step going down through the middle columns, and its extent is the run of
    columns where that step stays strong. The light comes from one side, so the rim fades
    towards the other end; the box is anchored on the end with the stronger rim (the lit
    side) and given the part's size from --part-box (every part is the same size). Margins:
    `margin` of the box on the sides and bottom, `top_margin` above the rim, so a bent or
    raised rim is not cut off. (The brightness-based PartLocator lost the part's edge against
    CAM4's dark background.)"""
    SCALE = 4

    def __init__(self, box, margin=0.02, top_margin=0.06):
        w, h = box[2] - box[0], box[3] - box[1]
        self.size = (round(w * (1 + 2 * margin)), round(h * (1 + margin + top_margin)))
        self.pad, self.pad_top = round(w * margin), round(h * top_margin)
        self.last_contrast = None
        self.side = None  # "left"/"right": anchor every image on this end (set by fix_side)

    def fix_side(self, paths):
        """The light comes from the same side in every image of a camera, so anchor all of them on
        the end that is stronger in most images (a stray highlight cannot flip a single crop)."""
        votes = [s_l >= s_r for *_, s_l, s_r in (self.rim(open_image(p)) for p in paths)]
        self.side = "left" if sum(votes) >= len(votes) / 2 else "right"
        return self.side, sum(votes), len(votes)

    def rim(self, img):
        """(left, top, right, strength at the left end, at the right end), original pixels."""
        s = self.SCALE
        a = gaussian_filter(np.asarray(img.convert("L").reduce(s), dtype=float), 1.5)
        h, w = a.shape
        gy = sobel(a, axis=0)
        c0, c1 = int(.42 * w), int(.58 * w)
        top = int(.08 * h) + int(np.argmax(gy[int(.08 * h):int(.6 * h), c0:c1].mean(1)))
        p = uniform_filter1d(gy[max(0, top - 10):top + 10].max(0), 9)  # rims tilt a little: +-10 rows
        lab, _ = label(p > 0.35 * np.median(p[c0:c1]))
        k = lab[w // 2] or next(iter(lab[c0:c1][lab[c0:c1] > 0]), 0)
        cols = np.flatnonzero(lab == k) if k else np.arange(c0, c1)
        end = max(3, len(cols) // 20)
        self.last_contrast = float(np.median(p[cols]))  # rim strength: low = check that crop
        return cols[0] * s, top * s, (cols[-1] + 1) * s, float(p[cols[:end]].mean()), float(p[cols[-end:]].mean())

    def __call__(self, img):
        left, top, right, s_left, s_right = self.rim(img)
        on_left = self.side == "left" if self.side else s_left >= s_right
        x0 = left - self.pad if on_left else right + self.pad - self.size[0]
        y0 = top - self.pad_top
        # Keep the box inside the frame (a part at the edge is cropped at the frame, not padded).
        x0 = int(np.clip(x0, 0, img.width - self.size[0]))
        y0 = int(np.clip(y0, 0, img.height - self.size[1]))
        return x0, y0, x0 + self.size[0], y0 + self.size[1]


class ColorLocator:
    """Crop to a strongly coloured part (e.g. a yellow cap) on a grey background.

    At 1/8 resolution the part is the largest connected region of saturated colour (HSV
    saturation above `min_sat`), with its holes filled (pale or transparent features inside it
    count as part). The crop is its bounding box grown by `margin` on every side, so parts at any
    position and distance from the camera end up the same size after resizing; `mask(path)` is
    the region outside the part (grown by half the margin), whose patches are left out of the
    memory bank and score 0, so background texture and lighting do not count."""
    SCALE = 8

    def __init__(self, margin=0.04, min_sat=0.35):
        self.margin, self.min_sat = margin, min_sat
        self.last_contrast = None
        self.masks = {}  # path -> part mask at 1/8 resolution, full frame

    def __call__(self, img, path=None):
        s = self.SCALE
        hsv = np.asarray(img.reduce(s).convert("HSV"), dtype=float) / 255
        m = (hsv[..., 1] > self.min_sat) & (hsv[..., 2] > 0.25)
        m = binary_fill_holes(binary_closing(m, iterations=3))
        lab, n = label(m)
        if not n:
            raise SystemExit(f"{getattr(path, 'name', path)}: no coloured part found")
        part = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
        ys, xs = np.nonzero(part)
        self.last_contrast = float(hsv[..., 1][part].mean() - hsv[..., 1][~part].mean()) * 100
        if path is not None:
            self.masks[path] = part
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        pad_x, pad_y = round(self.margin * w), round(self.margin * h)
        return (max(0, (xs.min() - pad_x) * s), max(0, (ys.min() - pad_y) * s),
                min(img.width, (xs.max() + 1 + pad_x) * s), min(img.height, (ys.max() + 1 + pad_y) * s))

    def background(self, path, box):
        """Boolean mask of the crop `box` that is background (outside the part grown by half the
        margin), at the crop's pixel size."""
        s = self.SCALE
        part = self.masks[path]
        grow = max(1, round(self.margin / 2 * max(part.shape)))
        part = binary_dilation(part, iterations=grow)
        crop = part[box[1] // s:-(-box[3] // s), box[0] // s:-(-box[2] // s)]
        return ~np.asarray(Image.fromarray(crop).resize((box[2] - box[0], box[3] - box[1])), dtype=bool)


class DiscLocator(ColorLocator):
    """Crop to a bright round part (e.g. the white cap seen from above) on a dark background.

    At 1/8 resolution the part is the largest bright region (Otsu on grey), opened first so a lit
    fixture edge touching the cap does not join onto it, with its holes filled. The part is then
    taken as the circle of the same area around that region's centroid, so a dark tear or a glint
    at the cap's edge does not change the crop. The crop is the square around that circle grown by
    `margin`; everything outside the circle is background (masked as with ColorLocator), so the
    crop's corners do not count and a rotated copy of a good crop has the same mask."""

    def __call__(self, img, path=None):
        s = self.SCALE
        g = np.asarray(img.reduce(s).convert("L"), dtype=float)
        t = otsu(g)
        m = binary_opening(g > t, iterations=3)
        lab, n = label(m)
        if not n:
            raise SystemExit(f"{getattr(path, 'name', path)}: no bright part found")
        blob = binary_fill_holes(lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:]))))
        ys, xs = np.nonzero(blob)
        cy, cx, r = ys.mean(), xs.mean(), np.sqrt(blob.sum() / np.pi)
        yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
        part = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        self.last_contrast = float(g[part].mean() - g[~part].mean())
        if path is not None:
            self.masks[path] = part
        half = r * (1 + self.margin)
        return (max(0, round((cx - half) * s)), max(0, round((cy - half) * s)),
                min(img.width, round((cx + half) * s)), min(img.height, round((cy + half) * s)))


def otsu(g):
    """Otsu threshold of a grey image (float 0..255)."""
    hist = np.bincount(np.clip(g, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(between))


def align_rims(boxes, train_paths, locator, max_deg=4.0, max_shift=60):
    """Level every crop's top rim to the good images' median tilt and move it to their
    median height: box -> (left, top, right, bottom, angle). Corrections are capped, so a
    badly bent rim cannot swing the crop. Returns the new boxes and the measured rims."""
    rims = {p: locator.rim(open_image(p).crop(b)) for p, b in boxes.items()}
    y_t = float(np.median([rims[p][0] for p in train_paths]))
    a_t = float(np.median([rims[p][1] for p in train_paths]))
    out = {}
    for p, (l, t, r, b) in boxes.items():
        dy = int(np.clip(rims[p][0] - y_t, -max_shift, max_shift))
        out[p] = (l, t + dy, r, b + dy, round(float(np.clip(rims[p][1] - a_t, -max_deg, max_deg)), 3))
    return out, rims, (y_t, a_t)


def apply_crop(img, box):
    """Crop to (left, top, right, bottom[, angle]); a non-zero angle (degrees, as PIL's
    rotate: counter-clockwise) first rotates the image about the box centre.
    ("unwrap", hole ellipse...) instead unwraps the ring around that hole into a strip."""
    if box[0] == "unwrap":
        return unwrap(img, box[1:])
    if len(box) == 5:
        if box[4]:
            img = img.rotate(box[4], resample=Image.BILINEAR,
                             center=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        box = box[:4]
    return img.crop(box)


def load_crop(results):
    """Crop function (img, file name) -> box or None for a finished run, as it was scored."""
    info = json.loads((results / "run_info.json").read_text())
    region = info.get("region", "full")
    if region == "unwrap":
        rings = read_rings(results / "rings.csv")
        return lambda img, name: rings[Path(name).name]
    if region == "part":
        with open(results / "crops.csv", newline="") as f:
            boxes = {r["file"]: tuple(int(r[k]) for k in ("left", "top", "right", "bottom"))
                     + ((float(r["angle"]),) if r.get("angle") else ()) for r in csv.DictReader(f)}
        return lambda img, name: boxes[Path(name).name]
    fn = REGIONS[region]
    return (lambda img, name: fn(img)) if fn else None


def save_crop_sheet(model, paths, contrast, out_path, cols=8, w=240):
    """Contact sheet of the crops, to check the part was found in every image."""
    tiles = []
    for p in paths:
        img = model.load(p)
        tile = img.resize((w, round(w * img.height / img.width)))
        ImageDraw.Draw(tile).text((4, 4), f"{p.stem[-6:]}  contrast {contrast[p]:.0f}", fill=(255, 255, 0))
        tiles.append(tile)
    h = tiles[0].height
    sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * h))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


RING_FIELDS = ("cx", "cy", "a", "b", "angle")


def read_rings(path):
    """{file name: ("unwrap", hole ellipse...[, r_from, r_to])} from rings.csv (--region unwrap);
    the band columns are there when the run chose its own (--unwrap-band)."""
    with open(path, newline="") as f:
        return {r["file"]: ("unwrap", *(float(r[k]) for k in RING_FIELDS),
                            *((float(r["r_from"]), float(r["r_to"])) if r.get("r_from") else ()))
                for r in csv.DictReader(f)}


def save_strip_sheet(model, paths, out_path, cols=3, w=640):
    """Contact sheet of the unwrapped strips, to check the hole was found: both ring edges
    should be straight horizontal lines (a wavy edge means an off-centre fit)."""
    tiles = []
    for p in paths:
        s = model.load(p)
        tile = s.resize((w, round(w * s.height / s.width)))
        ImageDraw.Draw(tile).text((4, 2), p.stem[-6:], fill=(255, 255, 0))
        tiles.append(tile)
    h = tiles[0].height + 4
    sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * h), (60, 0, 60))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


# part: PartLocator, unwrap: the hole found in each image (ring_unwrap); both built in main().
REGIONS = {"full": None, "ring": locate_ring, "part": None, "unwrap": None}
DEFAULT_SIZE = {"full": [768, 1024], "ring": [288, 1440],  # ring ~ native resolution of the crop
                "unwrap": [HEIGHT, WIDTH + SEAM]}


class PatchCore:
    def __init__(self, size, device, coreset_ratio, crop=None, post=None):
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.crop = crop
        self.post = post  # applied to every (cropped) image before scoring, e.g. suppress_white
        self.extractor = FeatureExtractor().to(device)
        self.tf = transforms.Compose([
            transforms.Resize(size), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def load(self, path):
        img = open_image(path).convert("RGB")
        img = apply_crop(img, self.crop(img, path)) if self.crop else img
        return self.post(img) if self.post else img

    def features(self, path, angle=0):
        """Patch features of the image at path; angle (degrees, counter-clockwise) turns the crop
        about its centre first (--rotate-train)."""
        img = self.load(path)
        if angle:
            img = img.rotate(angle, resample=Image.BILINEAR)
        x = self.tf(img)[None].to(self.device)
        return self.extractor(x)[0].half().cpu()  # (C, h, w)

    def build_bank(self, feats, masks=None):
        """Memory bank from the good images' patches; patches under a mask (engravings) are left out."""
        c = feats[0].shape[0]
        masks = masks or [None] * len(feats)
        patches = torch.cat([f.permute(1, 2, 0)[~torch.from_numpy(m)] if m is not None
                             else f.permute(1, 2, 0).reshape(-1, c) for f, m in zip(feats, masks)])
        return greedy_coreset(patches, self.coreset_ratio).float().to(self.device)

    @torch.no_grad()
    def score(self, feat, bank, mask=None, chunk=4096):
        """Returns (image score, patch-level anomaly map (h, w)). Masked patches score 0, so they
        never set the image score or count in the area rules."""
        c, h, w = feat.shape
        q = feat.permute(1, 2, 0).reshape(-1, c).float().to(self.device)
        d = torch.cat([torch.cdist(q[i:i + chunk], bank).min(dim=1).values
                       for i in range(0, q.shape[0], chunk)])
        m = d.reshape(h, w).cpu().numpy()
        if mask is not None:
            m[mask] = 0.0
        return float(m.max()), m


def patch_mask(pixel_mask, h, w, dilate):
    """Pixel mask (original image) -> (h, w) feature-grid mask: a patch is masked if any of its
    pixels is, then grown by `dilate` patches, since the 3x3 pooling and the coarser layer3
    features spread an engraving's influence into the neighbouring patches."""
    m = F.adaptive_max_pool2d(torch.from_numpy(pixel_mask.astype(np.float32))[None, None], (h, w))[0, 0].numpy() > 0
    return binary_dilation(m, structure=np.ones((3, 3)), iterations=dilate) if dilate else m


def engraving_boxes(paths, out, progress, load=None):
    """Engraved text in every image, {file name: boxes}. Cached in <out>/text_boxes.json, since
    detection takes a few seconds per image (delete the file to detect again). With `load`
    (--region unwrap) the text is found in the strip load(path) returns, in strip pixels."""
    cache = out / "text_boxes.json"
    boxes = json.loads(cache.read_text()) if cache.exists() else {}
    detect = None
    progress.start("engravings", len(paths))
    for i, p in enumerate(paths):
        if p.name not in boxes:
            if detect is None:
                from engraving_mask import EngravingDetector
                # Strips are already near native resolution; whole images are halved first.
                detect = EngravingDetector(scale=1.0 if load else 0.5)
            boxes[p.name] = detect(load(p) if load else p)
            cache.write_text(json.dumps(boxes))
        progress.update(i + 1)
    return boxes


def rasterize(boxes, shape):
    """Boolean (H, W) mask of the engraving boxes (same as engraving_mask.rasterize, without
    importing cv2/easyocr when the boxes come from the cache)."""
    img = Image.new("1", shape[::-1])
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.polygon([tuple(pt) for pt in b["box"]], fill=1)
    return np.asarray(img, dtype=bool)


def save_text_sheet(paths, boxes, out_path, cols=8, w=240, load=open_image):
    """Contact sheet of every image (or strip) with its detected engravings outlined, to check the masks."""
    tiles = []
    for p in paths:
        img = load(p).convert("RGB")
        s = w / img.width
        tile = img.resize((w, round(img.height * s)))
        d = ImageDraw.Draw(tile)
        for b in boxes[p.name]:
            d.polygon([(x * s, y * s) for x, y in b["box"]], outline=(0, 255, 255), width=2)
        d.text((4, 4), f"{p.stem[-6:]}  {' '.join(b['text'] for b in boxes[p.name]) or '-'}", fill=(255, 255, 0))
        tiles.append(tile)
    h = tiles[0].height
    sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * h))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


class Progress:
    """Writes <out>/progress.json after every step, so a long run can be checked from
    another terminal (python patchcore_score.py --status <out>)."""
    STAGES = ["features", "leave-one-out", "scoring", "heatmaps"]

    def __init__(self, out, t0, stages=None):
        self.path, self.t0 = out / "progress.json", t0
        self.STAGES = stages or self.STAGES
        self.stage, self.total, self.stage_t0 = None, 0, t0

    def start(self, stage, total):
        self.stage, self.total, self.stage_t0 = stage, total, time.time()
        self.update(0)

    def update(self, done, state="running", **extra):
        now = time.time()
        rate = (now - self.stage_t0) / done if done else None
        info = {"state": state, "stage": self.stage,
                "stage_index": self.STAGES.index(self.stage) + 1 if self.stage in self.STAGES else None,
                "stages": len(self.STAGES), "done": done, "total": self.total,
                "elapsed_s": round(now - self.t0),
                "stage_eta_s": round(rate * (self.total - done)) if rate else None,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(), **extra}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, indent=2))
        os.replace(tmp, self.path)

    def finish(self, **extra):
        self.stage = "done"
        self.update(self.total, state="done", **extra)


def print_status(out):
    path = Path(out) / "progress.json"
    if not path.exists():
        raise SystemExit(f"no progress.json in {out}: the run has not started (or predates progress tracking)")
    p = json.loads(path.read_text())
    alive = True
    try:
        os.kill(p["pid"], 0)
    except (ProcessLookupError, PermissionError, KeyError):
        alive = False
    if p["state"] == "running" and not alive:
        p["state"] = "stopped (process no longer running)"
    eta = f", ~{p['stage_eta_s'] // 60}m{p['stage_eta_s'] % 60:02d}s left in this stage" if p.get("stage_eta_s") else ""
    stage = f"stage {p['stage_index']}/{p['stages']} {p['stage']}" if p.get("stage_index") else p["stage"]
    print(f"{out}: {p['state']}  {stage}  {p['done']}/{p['total']}{eta}  "
          f"(elapsed {p['elapsed_s'] // 60}m, updated {p['updated']})")


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def save_heatmap(img, patch_map, score, threshold, title, out_path, vmax, boxes=()):
    img = np.asarray(img.convert("L"))
    heat = F.interpolate(torch.from_numpy(patch_map)[None, None], size=img.shape,
                         mode="bilinear", align_corners=False)[0, 0].numpy()
    heat = gaussian_filter(heat, sigma=8)
    wide = img.shape[1] > 2.5 * img.shape[0]  # ring strip: stack the panels vertically
    fig, ax = plt.subplots(2, 1, figsize=(14, 5.2)) if wide else plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(img, cmap="gray")
    # Shared colour scale across images so heatmaps are comparable.
    ax[1].imshow(heat, cmap="jet", alpha=0.45, vmin=threshold * 0.6, vmax=vmax)
    for b in boxes:  # masked engravings
        ax[1].add_patch(plt.Polygon(b["box"], fill=False, edgecolor="cyan", linewidth=1.2))
    for a in ax:
        a.axis("off")
    verdict = "BAD" if score > threshold else "good"
    fig.suptitle(f"{title}   score={score:.3f}  threshold={threshold:.3f}  -> {verdict}",
                 color="red" if verdict == "BAD" else "green")
    fig.tight_layout()
    fig.savefig(out_path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="image_data_patchcore/Model1")
    ap.add_argument("--region", choices=REGIONS, default="full",
                    help="full = whole image; ring = crop to the part's top ring at near-native resolution; "
                         "part = crop to the part, located in every image (needs --part-box); "
                         "unwrap = top view of a ring-shaped part (CAM2): unwrap the ring around its hole "
                         "into a straight strip (ring_unwrap.py); check the strips with --preview-crops")
    ap.add_argument("--part-box", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                    help="--region part: the part's box in the reference image, original pixels")
    ap.add_argument("--part-ref", default=None,
                    help="--region part: reference image for --part-box (default: first image in good/)")
    ap.add_argument("--part-margin", type=float, default=0.06,
                    help="--region part: grow the crop by this fraction of the box on every side, so a part "
                         "that looks slightly bigger or tilted is not clipped")
    ap.add_argument("--part-locator", choices=["corner", "rim", "color", "disc"], default="corner",
                    help="--region part: corner = find the part's top-left corner by brightness (PartLocator); "
                         "rim = find its top rim and anchor on the lit end (SideLocator, side views CAM3/CAM4; "
                         "--part-box then only gives the part's size); color = crop to the largest strongly "
                         "coloured region and ignore the background around it (ColorLocator, e.g. the yellow "
                         "cap; no --part-box needed); disc = crop to the largest bright round region and ignore "
                         "everything outside its circle (DiscLocator, e.g. the white cap from above; no --part-box)")
    ap.add_argument("--rotate-train", type=int, default=1, metavar="N",
                    help="--part-locator disc: learn each good crop at N evenly spaced rotations, so a round part "
                         "(and its print) turned to any angle is normal; the leave-one-out threshold leaves out all "
                         "of a good image's rotations")
    ap.add_argument("--part-top-margin", type=float, default=0.06,
                    help="--part-locator rim: room above the rim, as a fraction of the part's height")
    ap.add_argument("--align-rim", action="store_true",
                    help="--region part: also level each crop's top rim and put it at the same height "
                         "(the good images' median), so normal rim tilt/height does not look anomalous")
    ap.add_argument("--preview-crops", action="store_true",
                    help="--region part: only write part_crops.png (every crop, to check them) and stop")
    ap.add_argument("--size", type=int, nargs=2, default=None,
                    help="H W the image (or crop) is resized to (default: full 768 1024, ring 288 1440, "
                         "part 768 x the box's aspect ratio)")
    ap.add_argument("--coreset", type=float, default=0.3, help="fraction of training patches kept")
    ap.add_argument("--out", default=None,
                    help="output folder (default results/<model>, or results/<model>_ring)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="score above which an image is predicted bad (default: automatic, "
                         "max leave-one-out score of the good images)")
    ap.add_argument("--mask-text", action="store_true",
                    help="find engraved alphanumeric codes (EasyOCR) in every image and ignore them: their "
                         "patches are left out of the memory bank and score 0 (whole-image region only)")
    ap.add_argument("--mask-dilate", type=int, default=1,
                    help="--mask-text: grow each engraving mask by this many feature patches")
    ap.add_argument("--preview-text", action="store_true",
                    help="--mask-text: only detect the engravings, write text_masks.png (every image with "
                         "its masks outlined) and stop")
    ap.add_argument("--suppress-white", action="store_true",
                    help="--region unwrap: clamp bright spots (white dots, glints, powder) to the ring's surface "
                         "level before scoring, in good and test strips alike; dark holes, pits and chipped "
                         "edges are untouched (ring_unwrap.suppress_white)")
    ap.add_argument("--text-boxes", default=None, type=Path,
                    help="--mask-text: reuse the engraving detections (text_boxes.json) of an earlier run "
                         "with the same images and region, instead of detecting again")
    ap.add_argument("--unwrap-band", type=float, nargs=2, default=None, metavar=("R_FROM", "R_TO"),
                    help=f"--region unwrap: radii (in hole radii) the strip spans, default {R_FROM} {R_TO}; "
                         f"the ring itself is ~1.0-{OUTER}, so a tighter band crops away more background")
    ap.add_argument("--ring-outer", type=float, default=OUTER,
                    help=f"--region unwrap: the ring's outer radius in hole radii (default {OUTER}, Model1; "
                         "Model2 is 1.60); bounds the face band the engraving mask is limited to")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="ID",
                    help="leave out images whose file name ends with these (e.g. 000125 000126), "
                         "in good/ and all/")
    ap.add_argument("--status", metavar="RESULTS", default=None,
                    help="only print the progress of a run writing to this results folder, and exit")
    args = ap.parse_args()
    if args.status:
        return print_status(args.status)

    root = Path(args.data)
    keep = lambda paths: [p for p in paths if not any(p.stem.endswith(x) for x in args.exclude)]
    train_paths = keep(list_images(root / "good"))
    all_paths = keep(list_images(root / "all"))
    if args.exclude:
        print(f"excluded images ending with {', '.join(args.exclude)}")
    band = tuple(args.unwrap_band) if args.unwrap_band else ()
    if band and args.region != "unwrap":
        ap.error("--unwrap-band needs --region unwrap")
    crop = REGIONS[args.region]
    if args.mask_text and args.region not in ("full", "unwrap"):
        ap.error("--mask-text works on the whole image or the unwrapped strip only (--region full / unwrap)")
    if args.suppress_white and args.region != "unwrap":
        ap.error("--suppress-white needs --region unwrap (the surface level is measured along the strip's rows)")
    color_loc = None
    if args.region == "part":
        if not args.part_box and args.part_locator not in ("color", "disc"):
            ap.error("--region part needs --part-box LEFT TOP RIGHT BOTTOM (the part in the reference image)")
        part_ref = Path(args.part_ref) if args.part_ref else train_paths[0]
        if args.part_locator in ("color", "disc"):
            crop = color_loc = (ColorLocator if args.part_locator == "color" else DiscLocator)(args.part_margin)
            bw, bh = 2, 1  # replaced by the good images' median crop shape once they are located
        elif args.part_locator == "rim":
            crop = SideLocator(args.part_box, args.part_margin, args.part_top_margin)
            bw, bh = crop.size
        else:
            crop = PartLocator(part_ref, args.part_box, args.part_margin)
            bw, bh = crop.size[0] + 2 * crop.pad[0], crop.size[1] + 2 * crop.pad[1]
        DEFAULT_SIZE["part"] = [768, round(768 * bw / bh / 8) * 8]
    size_given = args.size is not None
    args.size = args.size or DEFAULT_SIZE[args.region]
    suffix = "" if args.region == "full" else f"_{args.region}"
    out = Path(args.out or Path("results") / f"{root.name}{suffix}")
    (out / "heatmaps").mkdir(parents=True, exist_ok=True)
    device = get_device()
    t0 = time.time()

    print(f"train: {len(train_paths)} good images   score: {len(all_paths)} images in all/   "
          f"region: {args.region} {args.size[0]}x{args.size[1]}   device: {device}")

    if args.region == "part":
        # Locate the part once per image; PatchCore.load then reuses the box.
        if args.part_locator == "rim":
            side, n_left, n = crop.fix_side(list(dict.fromkeys(train_paths + all_paths)))
            print(f"rim locator: left end stronger in {n_left}/{n} images -> anchoring every crop on the {side} end")
        boxes, contrast = {}, {}
        for p in dict.fromkeys(train_paths + all_paths):
            boxes[p] = crop(open_image(p), p) if color_loc else crop(open_image(p))
            contrast[p] = crop.last_contrast
        if color_loc and not size_given:
            # The colour crops follow each part's own size: resize them all to the good parts' median shape.
            aspect = float(np.median([(b[2] - b[0]) / (b[3] - b[1]) for p, b in boxes.items() if p in train_paths]))
            args.size = [512, round(512 * aspect / 8) * 8]
            print(f"{args.part_locator} locator: crops resized to {args.size[0]}x{args.size[1]} "
                  f"(good parts' median shape {aspect:.2f}:1); background around the part is masked")
        rims = {}
        if args.align_rim:
            boxes, rims, (y_t, a_t) = align_rims(boxes, train_paths, crop)
            ang = np.array([b[4] for b in boxes.values()])
            print(f"rim alignment: target height {y_t:.0f}px in the crop, tilt {a_t:.2f} deg; "
                  f"corrections {np.percentile(np.abs(ang), 50):.2f} deg median, {np.abs(ang).max():.2f} max")
        with open(out / "crops.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "left", "top", "right", "bottom", "angle", "contrast", "folder"])
            for p, b in boxes.items():
                w.writerow([p.name, *b[:4], b[4] if len(b) == 5 else 0, f"{contrast[p]:.1f}", p.parent.name])
        weak = sorted(contrast.items(), key=lambda kv: kv[1])[:5]
        print(f"part located in {len(boxes)} images (reference {part_ref.name}, box {args.part_box}); "
              f"lowest part/background contrast: " + ", ".join(f"{p.name} {c:.0f}" for p, c in weak))
        crop = lambda img, path: boxes[path]
    elif args.region == "unwrap":
        # Find the hole once per image; PatchCore.load then unwraps around it.
        rings = {}
        for p in dict.fromkeys(train_paths + all_paths):
            try:
                rings[p] = ("unwrap", *find_hole(open_image(p)), *band)
            except ValueError as e:
                raise SystemExit(f"{p.name}: {e}; this image cannot be unwrapped")
        with open(out / "rings.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", *RING_FIELDS, *(("r_from", "r_to") if band else ()), "folder"])
            for p, r in rings.items():
                w.writerow([p.name, *(f"{v:.2f}" for v in r[1:]), p.parent.name])
        radii = np.array([(r[3] + r[4]) / 2 for r in rings.values()])
        print(f"hole found in {len(rings)} images: radius {np.median(radii):.0f}px median "
              f"({radii.min():.0f}-{radii.max():.0f}); strips {args.size[0]}x{args.size[1]}  ({time.time() - t0:.0f}s)")
        crop = lambda img, path: rings[path]
    elif crop:
        crop = (lambda fn: lambda img, path: fn(img))(crop)
    model = PatchCore(tuple(args.size), device, args.coreset, crop, suppress_white if args.suppress_white else None)
    if args.region == "part":
        save_crop_sheet(model, sorted(boxes, key=lambda p: contrast[p]), contrast, out / "part_crops.png")
        print(f"crop contact sheet -> {out / 'part_crops.png'}  (lowest contrast first)")
        if args.preview_crops:
            return
    if args.region == "unwrap":
        save_strip_sheet(model, all_paths, out / "strips.png")
        print(f"strip contact sheet -> {out / 'strips.png'}")
        if args.preview_crops:
            return
    progress = Progress(out, t0, (["engravings"] if args.mask_text else []) + Progress.STAGES)
    text_boxes = {}
    if args.mask_text:
        strips = args.region == "unwrap"
        if args.text_boxes and not (out / "text_boxes.json").exists():
            (out / "text_boxes.json").write_text(args.text_boxes.read_text())
        # Text is found in the strip as unwrapped, before any white-spot suppression (which can dim engravings).
        text_boxes = engraving_boxes(list(dict.fromkeys(train_paths + all_paths)), out, progress,
                                     (lambda p: apply_crop(open_image(p).convert("RGB"), rings[p])) if strips else None)
        n_text = sum(bool(b) for b in text_boxes.values())
        print(f"engravings found in {n_text}/{len(text_boxes)} images  ({time.time() - t0:.0f}s)")
        save_text_sheet(all_paths, text_boxes, out / "text_masks.png",
                        **({"cols": 3, "w": 640, "load": model.load} if strips else {}))
        print(f"engraving contact sheet -> {out / 'text_masks.png'}")
        if args.preview_text:
            return progress.finish()
    # Heatmap names include the prediction, so stale ones from a previous run would linger.
    for old in (out / "heatmaps").glob("*.png"):
        old.unlink()
    if args.rotate_train > 1 and args.part_locator != "disc":
        ap.error("--rotate-train needs --region part --part-locator disc (a round part, centred in its crop)")
    angles = [360 * k / args.rotate_train for k in range(args.rotate_train)]
    progress.start("features", len(train_paths))
    train_rot = []  # per good image: its features at every training angle
    for i, p in enumerate(train_paths):
        train_rot.append([model.features(p, a) for a in angles])
        progress.update(i + 1)
    train_feats = [f[0] for f in train_rot]
    grid = train_feats[0].shape[1:]
    if len(angles) > 1:
        print(f"rotations: each good crop learned at {len(angles)} angles, {angles[1]:.0f} deg apart")

    def mask_of(p):
        """Feature-grid mask of p: its engravings (--mask-text) and, with the colour locator, the
        background around the part (None if neither applies)."""
        out = None
        if args.mask_text:
            if args.region == "unwrap":
                # Only the flat face between the edges is masked, so edge chips next to a code still count.
                m = np.zeros((HEIGHT, WIDTH + SEAM), bool)
                face = face_rows(0.06, *(band or (R_FROM, R_TO)), args.ring_outer)
                m[face] = rasterize(text_boxes[p.name], m.shape)[face]
            else:
                m = rasterize(text_boxes[p.name], open_image(p).size[::-1])
            out = patch_mask(m, *grid, args.mask_dilate)
        if color_loc:
            # A patch is background only if (nearly) all of it is, so the part's own outline is still scored.
            bg = color_loc.background(p, boxes[p])
            bg = F.adaptive_avg_pool2d(torch.from_numpy(bg.astype(np.float32))[None, None], tuple(grid))[0, 0].numpy() >= 0.9
            out = bg if out is None else out | bg
        return out

    train_masks = [mask_of(p) for p in train_paths]
    # Outside the disc is masked, and a turned disc stays inside it, so every rotation keeps its image's mask.
    flat = lambda idx: ([f for i in idx for f in train_rot[i]], [train_masks[i] for i in idx for _ in angles])
    bank = model.build_bank(*flat(range(len(train_paths))))
    print(f"memory bank: {bank.shape[0]} patches  ({time.time() - t0:.0f}s)")

    # Leave-one-out scores of the training images -> threshold.
    loo = {}
    progress.start("leave-one-out", len(train_paths))
    for i, p in enumerate(train_paths):
        b = model.build_bank(*flat([j for j in range(len(train_paths)) if j != i]))
        loo[md5(p)] = model.score(train_feats[i], b, train_masks[i])
        print(f"  leave-one-out {p.name}: {loo[md5(p)][0]:.3f}")
        progress.update(i + 1)
    auto_threshold = max(s for s, _ in loo.values())
    print(f"automatic threshold = {auto_threshold:.3f}  (max leave-one-out score of the good images)")
    threshold = args.threshold if args.threshold is not None else auto_threshold
    if args.threshold is not None:
        print(f"using manual threshold = {threshold:.3f}")

    results = []
    progress.start("scoring", len(all_paths))
    for i, p in enumerate(all_paths):
        h = md5(p)
        in_train = h in loo
        s, m = loo[h] if in_train else model.score(model.features(p), bank, mask_of(p))
        results.append({"path": p, "score": s, "map": m, "in_train": in_train})
        progress.update(i + 1, threshold=threshold)
        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{len(all_paths)}")
    results.sort(key=lambda r: -r["score"])
    # Raw patch-level maps, so other decision rules (region_decision.py) can be applied without re-scoring.
    np.savez_compressed(out / "maps.npz", **{r["path"].name: r["map"] for r in results})

    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "file", "score", "prediction", "in_training_set"])
        for rank, r in enumerate(results, 1):
            w.writerow([rank, r["path"].name, f"{r['score']:.4f}",
                        "bad" if r["score"] > threshold else "good", "yes" if r["in_train"] else ""])

    vmax = np.percentile([r["score"] for r in results], 95)
    progress.start("heatmaps", len(results))
    for rank, r in enumerate(results, 1):
        pred = "bad" if r["score"] > threshold else "good"
        tag = " (training image, leave-one-out)" if r["in_train"] else ""
        save_heatmap(model.load(r["path"]), r["map"], r["score"], threshold, f"#{rank} {r['path'].name}{tag}",
                     out / "heatmaps" / f"{rank:03d}_{pred}_{r['score']:.3f}_{r['path'].stem}.png", vmax,
                     text_boxes.get(r["path"].name, ()))
        progress.update(rank, threshold=threshold)

    scores = np.array([r["score"] for r in results])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=40, color="steelblue")
    ax.axvline(threshold, color="r", ls="--", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("images")
    ax.set_title(f"{root.name}: {len(scores)} images in all/")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "score_hist.png", dpi=100)
    plt.close(fig)

    (out / "run_info.json").write_text(json.dumps({
        "threshold": threshold, "auto_threshold": auto_threshold,
        "manual_threshold": args.threshold is not None,
        "region": args.region, "size": args.size, "coreset": args.coreset,
        **({"part_box": args.part_box, "part_ref": part_ref.name, "part_margin": args.part_margin,
            "part_locator": args.part_locator,
            **({"rotate_train": args.rotate_train} if args.rotate_train > 1 else {}),
            "align_rim": args.align_rim}
           if args.region == "part" else {}),
        **({"mask_text": True, "mask_dilate": args.mask_dilate} if args.mask_text else {}),
        **({"unwrap": {"r_from": (band or (R_FROM, R_TO))[0], "r_to": (band or (R_FROM, R_TO))[1],
                       "outer": args.ring_outer, "seam": SEAM}}
           if args.region == "unwrap" else {}),
        **({"excluded": args.exclude} if args.exclude else {}),
        **({"suppress_white": True} if args.suppress_white else {}),
        "train_images": [p.name for p in train_paths]}, indent=2))

    n_bad = int((scores > threshold).sum())
    progress.finish(threshold=threshold, predicted_bad=n_bad, predicted_good=len(scores) - n_bad)
    print(f"\npredicted bad: {n_bad}   predicted good: {len(scores) - n_bad}")
    print(f"done in {time.time() - t0:.0f}s -> {out}/scores.csv, {out}/heatmaps/, {out}/score_hist.png")


if __name__ == "__main__":
    main()
