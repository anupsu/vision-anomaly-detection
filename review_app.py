"""Local review UI for PatchCore results, one or more camera views per part.

Scroll through the parts, see each view's prediction and heatmap, and record the
actual label (good/bad) from a dropdown. Labels are saved as soon as they are
chosen, so you can stop and resume.

Usage:
    python review_app.py                      # Model1, http://127.0.0.1:8765
    python review_app.py --results results/Model1_ring        # review the top-ring run
    # Every camera of a part model: image_data_patchcore/Model1/CAM2, CAM3, ... (each with all/ and good/),
    # scored into results/Model1_CAM2, results/Model1_CAM3, ...
    python review_app.py --model image_data_patchcore/Model1
    # Or name the views (folders with all/ and good/) one by one:
    python review_app.py --view image_data_patchcore/Model1/CAM3 --view image_data_patchcore/Model1/CAM4:results/Model1_CAM4

Multiple views: the images of one part are matched by file name with the camera
token removed (model1_CAM3_T2_..._000005.bmp and model1_CAM4_T2_..._000005.bmp are
the same part). Each view has its own rules (below); a part is bad if ANY view
flags it. Labels are per part, in results/<model>_parts/labels.csv (--model) or
results/parts_<view1>_<view2>/labels.csv; the first time, they are seeded from each
view's own labels.csv (bad in any view = bad).
Single view: labels are per image in results/<model>/labels.csv, shared by every
run on that dataset (full image, ring).

Keyboard: up/down or k/j = previous/next, g = good, b = bad, u = clear label.

The rules bar in the header re-classifies images live without re-running the
model (one tab per view). An image is bad if any enabled rule fires:
  1. max score   - worst patch score > threshold (the model run's rule)
  2. region      - a connected (8-neighbour) region of >= N patches, every patch
                   scoring > a (usually lower) threshold
  3. total area  - >= M patches anywhere in the image scoring > another threshold
Rules 2 and 3 need the patch maps (maps.npz from patchcore_score.py); the maps can
be smoothed (Gaussian, in patches) before they count, so thin streaks fade. The rules
only change what the UI shows; they are remembered in the browser, and
scores.csv is left untouched.
"""

import argparse
import csv
import io
import json
import os
import re
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, label, maximum_filter, median_filter

from ring_unwrap import OUTER, R_FROM, R_TO, unwrap

ARGS = None
VIEWS = []  # [{"name", "id", "data", "results"}], one per camera view
DATA_ROOT = "image_data_patchcore"
LOCK = threading.Lock()
EIGHT = np.ones((3, 3))  # 8-neighbour connectivity, as in region_decision.py
CAM = re.compile(r"_CAM\d+_", re.IGNORECASE)


def part_key(file):
    """The part an image shows: with several views, its file name without the camera token."""
    return CAM.sub("_", file) if len(VIEWS) > 1 else file


def load_items(v):
    res = VIEWS[v]["results"]
    with open(res / "scores.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    heatmaps = {p.name.split("_", 3)[3].rsplit(".", 1)[0]: p.name
                for p in (res / "heatmaps").glob("*.png")}
    return [{"rank": int(r["rank"]), "file": r["file"], "part": part_key(r["file"]), "score": float(r["score"]),
             "prediction": r["prediction"], "in_training": r["in_training_set"] == "yes",
             "heatmap": heatmaps.get(Path(r["file"]).stem)}
            for r in rows]


def model_threshold(v, items):
    """Threshold the scoring run used: from run_info.json, else inferred from the
    predictions (bad means score > threshold, so the highest 'good' score works)."""
    info = VIEWS[v]["results"] / "run_info.json"
    if info.exists():
        return json.loads(info.read_text())["threshold"]
    good = [it["score"] for it in items if it["prediction"] == "good"]
    return max(good) if good else min(it["score"] for it in items)


def run_region(v):
    info = VIEWS[v]["results"] / "run_info.json"
    return json.loads(info.read_text()).get("region", "full") if info.exists() else "full"


@lru_cache(maxsize=8)
def load_geometry(v):
    """Chip/bump and pit candidates of a ring_defects.py run ({file: {"edges", "pits"}}), or None.
    Each edge candidate gets the strip row of its edge, for drawing; the traced edges themselves
    and the faintest candidates (the UI's thresholds never go that low) are not sent."""
    path = VIEWS[v]["results"] / "defects.json"
    if not path.exists():
        return None
    out = {}
    for k, d in json.loads(path.read_text()).items():
        edges = [{**e, "row": d[e["edge"]][(e["col"] + e["cols"] // 2) % len(d[e["edge"]])]}
                 for e in d["edges"] if e["depth"] >= 5]
        out[k] = {"edges": edges, "pits": [p for p in d["pits"] if p["contrast"] >= 25]}
    return out


@lru_cache(maxsize=8)
def load_maps(v):
    """Patch-level anomaly maps {file: (h, w) array}; empty if the run did not save them."""
    path = VIEWS[v]["results"] / "maps.npz"
    if not path.exists():
        return {}
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


@lru_cache(maxsize=16)
def smoothed_maps(v, sigma):
    """Maps blurred by a Gaussian of `sigma` patches before the area rules count patches:
    thin, broken streaks (e.g. a highlight along a rim) fade, solid blobs survive."""
    if not sigma:
        return load_maps(v)
    return {k: gaussian_filter(m, sigma, mode="nearest") for k, m in load_maps(v).items()}


def default_t(v, sigma):
    """Default area-rule threshold: 99th percentile of the good training images' (smoothed) patch scores."""
    maps = smoothed_maps(v, sigma)
    train = [maps[it["file"]].ravel() for it in load_items(v) if it["in_training"] and it["file"] in maps]
    return round(float(np.percentile(np.concatenate(train or [m.ravel() for m in maps.values()]), 99)), 3)


def map_info(v):
    """Grid size and score range for the sliders."""
    maps = load_maps(v)
    if not maps:
        return None
    allv = np.concatenate([m.ravel() for m in maps.values()])
    return {"grid": list(next(iter(maps.values())).shape),
            "range": [float(np.percentile(allv, 50)), float(allv.max())]}


def largest_region(m, t):
    """Mask of the largest connected region of patches scoring > t (None if there is none)."""
    lab, n = label(m > t, structure=EIGHT)
    if not n:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


POL_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def polarity(v):
    """{file: (bright, dark)} on the map grid, in grey levels, for the white-spot filter: how far
    the brightest (98th pct) pixels of each patch rise above the surrounding surface, and how far
    its darkest (2nd pct) pixels fall below it; the surface level is the median of the 7x7
    surrounding patches' medians. Each is the max over the patch and its 8 neighbours, since a
    hotspot can sit a patch off the spot that caused it. Measured on the image as scored (crop or
    strip) and cached in <results>/polarity.npz; the first use takes about a minute."""
    with POL_LOCK:
        maps, cache = load_maps(v), VIEWS[v]["results"] / "polarity.npz"
        if cache.exists():
            with np.load(cache) as d:
                out = {k: (d[k + "|b"], d[k + "|d"]) for k in maps if k + "|b" in d.files}
            if len(out) == len(maps):
                return out
        out, boxes = {}, part_crops(v)
        for k, m in maps.items():
            h, w = m.shape
            img = apply_box(open_image(VIEWS[v]["data"] / "all" / k), boxes.get(k))
            g = np.asarray(img.convert("L").resize((w * 8, h * 8), Image.BILINEAR), dtype=np.float32)
            cells = g.reshape(h, 8, w, 8).transpose(0, 2, 1, 3).reshape(h, w, 64)
            level = median_filter(np.median(cells, -1), size=7, mode="nearest")
            out[k] = (maximum_filter(np.percentile(cells, 98, -1) - level, 3).astype(np.float32),
                      maximum_filter(level - np.percentile(cells, 2, -1), 3).astype(np.float32))
        np.savez_compressed(cache, **{f"{k}|b": b for k, (b, _) in out.items()},
                            **{f"{k}|d": d for k, (_, d) in out.items()})
        return out


def split_white(m, t, pol, bright, dark, k=5):
    """Connected regions of patches scoring > t, split into (kept, ignored) masks. A region is a
    white spot, and ignored, when its k hottest patches show pixels more than `bright` grey levels
    above the surrounding surface and none more than `dark` below it. Holes and pits are dark, and
    a chipped edge lets the dark background show, so those are always kept."""
    lab, n = label(m > t, structure=EIGHT)
    if not n:
        return np.zeros(m.shape, bool), np.zeros(m.shape, bool)
    idx = np.flatnonzero(lab)
    idx = idx[np.argsort(-m.flat[idx], kind="stable")]  # hottest first...
    idx = idx[np.argsort(lab.flat[idx], kind="stable")]  # ...within each region
    reg = lab.flat[idx]
    rank = np.arange(len(idx)) - np.searchsorted(reg, reg)
    top = rank < k
    b_max, d_max = np.zeros(n + 1), np.zeros(n + 1)
    np.maximum.at(b_max, reg[top], pol[0].flat[idx[top]])
    np.maximum.at(d_max, reg[top], pol[1].flat[idx[top]])
    white = (b_max > bright) & (d_max < dark)
    white[0] = False
    ignored = white[lab]
    return (lab > 0) & ~ignored, ignored


def largest(mask):
    """Size of the largest connected (8-neighbour) region of a boolean mask."""
    lab, n = label(mask, structure=EIGHT)
    return int(np.bincount(lab.ravel())[1:].max()) if n else 0


@lru_cache(maxsize=64)
def area_stats(v, t_region, t_total, sigma, t_max=None, bright=None, dark=None):
    """{file: [largest connected area > t_region, total area > t_total, max-score rule fires,
    patches ignored as white spots]}, areas in patches. Without the white-spot filter (bright
    None), the last two are None and 0: the page compares the max score itself."""
    out = {}
    pol = polarity(v) if bright is not None else None
    raw = load_maps(v)
    for k, m in smoothed_maps(v, sigma).items():
        if pol is None:
            reg = largest_region(m, t_region)
            out[k] = [int(reg.sum()) if reg is not None else 0, int((m > t_total).sum()), None, 0]
            continue
        keep_r, ign_r = split_white(m, t_region, pol[k], bright, dark)
        keep_t, ign_t = split_white(m, t_total, pol[k], bright, dark)
        keep_m, ign_m = split_white(raw[k], t_max if t_max is not None else np.inf, pol[k], bright, dark)
        out[k] = [largest(keep_r), int(keep_t.sum()), int(keep_m.any()), int((ign_r | ign_t | ign_m).sum())]
    return out


def mask_png(m, t_region, t_total, background, pol=None, bright=None, dark=None):
    """Patch-grid PNG: yellow = patches counted by the total-area rule, red = the
    largest connected region of the region rule, cyan = hotspots ignored as white spots
    (with the white-spot filter). Optionally on a grey map background."""
    rgba = np.zeros(m.shape + (4,), np.uint8)
    if background:
        g = np.clip((m - m.min()) / (np.ptp(m) or 1) * 200, 0, 255).astype(np.uint8)
        rgba[..., :3], rgba[..., 3] = g[..., None], 255
    a = lambda alpha: alpha if not background else 255
    white = lambda t: split_white(m, t, pol, bright, dark) if pol is not None else (m > t, None)
    if t_total is not None:
        keep, ign = white(t_total)
        rgba[keep] = (255, 200, 0, a(140))
        if ign is not None:
            rgba[ign] = (0, 210, 255, a(150))
    if t_region is not None:
        keep, ign = white(t_region)
        if ign is not None:
            rgba[ign] = (0, 210, 255, a(150))
        reg = largest_region(np.where(keep, m, -np.inf), t_region)
        if reg is not None:
            rgba[reg] = (255, 40, 40, a(190))
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def read_labels(path):
    with open(path, newline="") as f:
        return {r["file"]: r["actual_label"] for r in csv.DictReader(f) if r["actual_label"]}


def load_labels():
    """{part: label}. With several views and no labels file yet, seeded from each view's
    own labels (keyed by image): a part is bad if it was marked bad in any view."""
    if ARGS.labels.exists():
        return read_labels(ARGS.labels)
    labels = {}
    if len(VIEWS) > 1:
        for view in VIEWS:
            path = view["results"] / "labels.csv"
            if path.exists():
                for file, lab in read_labels(path).items():
                    k = part_key(file)
                    labels[k] = "bad" if "bad" in (lab, labels.get(k)) else lab
    return labels


def save_label(part, label):
    with LOCK:
        labels = load_labels()
        if label:
            labels[part] = label
        else:
            labels.pop(part, None)
        path = ARGS.labels
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "actual_label"])
            for k in sorted(labels):
                w.writerow([k, labels[k]])
        os.replace(tmp, path)  # atomic, so a crash never leaves a half-written file


@lru_cache(maxsize=8)
def part_crops(v):
    """{file: crop box} for runs scored with --region part, {file: ("unwrap", hole ellipse...)}
    for --region unwrap, else {}."""
    res = VIEWS[v]["results"]
    if (res / "rings.csv").exists():
        with open(res / "rings.csv", newline="") as f:
            return {r["file"]: ("unwrap", *(float(r[k]) for k in ("cx", "cy", "a", "b", "angle")),
                                *((float(r["r_from"]), float(r["r_to"])) if r.get("r_from") else ()))
                    for r in csv.DictReader(f)}
    path = res / "crops.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["file"]: tuple(int(r[k]) for k in ("left", "top", "right", "bottom"))
                + ((float(r["angle"]),) if r.get("angle") else ()) for r in csv.DictReader(f)}


def open_image(path):
    """Open an image upright: phone photos carry an EXIF rotation flag (same as patchcore_score)."""
    return ImageOps.exif_transpose(Image.open(path))


def apply_box(img, box):
    """The image as it was scored: unwrapped strip, part crop, or unchanged (box None).
    Same as patchcore_score.apply_crop (not imported, to keep torch out of the UI)."""
    if box and box[0] == "unwrap":
        return unwrap(img, box[1:])
    if box:
        if len(box) == 5 and box[4]:
            img = img.rotate(box[4], resample=Image.BILINEAR, center=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        img = img.crop(box[:4])
    return img


@lru_cache(maxsize=128)
def jpeg(path, max_w, box=None):
    img = apply_box(open_image(path).convert("RGB"), box)  # crop/strip, so the area overlay lines up
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self.send(PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/items":
            views = []
            for v, view in enumerate(VIEWS):
                items = load_items(v)
                info = view["results"] / "run_info.json"
                info = json.loads(info.read_text()) if info.exists() else {}
                # Unwrapped strips: per image the hole radius (original pixels) and the band of radii
                # the strip spans, so the page can label the strip's axes in pixels and degrees.
                # box = ("unwrap", cx, cy, a, b, angle[, r_from, r_to])
                strip = {k: [round((b[3] + b[4]) / 2, 1), *(b[6:8] if len(b) >= 8 else (R_FROM, R_TO))]
                         for k, b in part_crops(v).items() if b[0] == "unwrap"}
                outer = info.get("outer") or info.get("unwrap", {}).get("outer") or OUTER
                views.append({"name": view["name"], "id": view["id"], "items": items,
                              "strip": strip or None, "outer": outer,
                              "geometry": load_geometry(v), "geo_defaults": info.get("defaults"),
                              "threshold": model_threshold(v, items),
                              "maps": map_info(v), "region": run_region(v),
                              "full_frame": run_region(v) in ("full", "part", "unwrap")})
            names = " + ".join(view["name"] for view in VIEWS)
            title = f"{ARGS.model.name}: {names}" if ARGS.model else names
            return self.send(json.dumps({"views": views, "title": title, "labels": load_labels(),
                                         "labels_file": str(ARGS.labels)}).encode(), "application/json")
        try:
            v = int(q.get("view", ["0"])[0])
            if not 0 <= v < len(VIEWS):
                raise ValueError
        except ValueError:
            return self.send(b"bad view", "text/plain", 400)
        if url.path in ("/api/areas", "/mask", "/api/default_t"):
            def t(key):
                x = q.get(key, [None])[0]
                return round(float(x), 4) if x not in (None, "") else None
            try:
                t_region, t_total = t("region"), t("total")
                sigma = min(max(t("sigma") or 0.0, 0.0), 10.0)
                t_max, bright, dark = t("max"), t("bright"), t("dark")  # white-spot filter, when on
            except ValueError:
                return self.send(b"bad threshold", "text/plain", 400)
            if bright is not None and dark is None:
                return self.send(b"bright needs dark", "text/plain", 400)
            if url.path == "/api/default_t":
                return self.send(json.dumps(default_t(v, sigma)).encode(), "application/json")
            if url.path == "/api/areas":
                stats = area_stats(v, t_region if t_region is not None else np.inf,
                                   t_total if t_total is not None else np.inf, sigma, t_max, bright, dark)
                return self.send(json.dumps(stats).encode(), "application/json")
            name = Path(q.get("file", [""])[0]).name
            m = smoothed_maps(v, sigma).get(name)
            if m is None:
                return self.send(b"not found", "text/plain", 404)
            pol = polarity(v)[name] if bright is not None else None
            return self.send(mask_png(m, t_region, t_total, q.get("bg") == ["1"], pol, bright, dark), "image/png")
        if url.path.startswith("/img/"):
            name = Path(unquote(url.path[5:])).name  # basename only: no path traversal
            kind = q.get("kind", ["orig"])[0]
            folder = VIEWS[v]["results"] / "heatmaps" if kind == "heat" else VIEWS[v]["data"] / "all"
            path = folder / name
            if not path.is_file():
                return self.send(b"not found", "text/plain", 404)
            width = {"thumb": 160, "heat": 1400}.get(kind, 1600)
            # Part-cropped runs show the crop that was scored, so the area overlay lines up;
            # unwrapped runs show the strip, except in the list, where a strip is too thin to recognise.
            box = part_crops(v).get(name) if kind != "heat" else None
            if box and box[0] == "unwrap" and kind == "thumb":
                box = None
            return self.send(jpeg(str(path), width, box), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self.send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("label") not in ("", "good", "bad") or not body.get("part"):
            return self.send(b"bad label", "text/plain", 400)
        save_label(Path(body["part"]).name, body["label"])
        self.send(b'{"ok":true}', "application/json")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inspection Review</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --text: #1d2330; --muted: #677085; --line: #e2e5eb;
  --sel: #e8efff; --accent: #2f6fed; --good: #1f8a4c; --good-bg: #e3f4ea; --bad: #c8372d; --bad-bg: #fbe6e4;
  --warn: #9a6a00; --warn-bg: #fff3d6;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171d; --panel: #1c2028; --text: #e6e9ef; --muted: #98a1b3; --line: #2c323d;
    --sel: #23304a; --accent: #6d9bff; --good: #4cc27f; --good-bg: #173426; --bad: #ff7a6e; --bad-bg: #3d1f1c;
    --warn: #f0c060; --warn-bg: #3a3018; }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
header { display: flex; align-items: center; gap: 4px 12px; padding: 4px 12px; background: var(--panel);
  border-bottom: 1px solid var(--line); flex-wrap: wrap; font-size: 13px; }
header h1 { font-size: 14px; margin: 0; }
header select, .thrbar button, .thrbar select, #rulesToggle { padding: 2px 6px; font-size: 13px; }
header select { max-width: 210px; }
.stat { color: var(--muted); }
.stat b { color: var(--text); font-variant-numeric: tabular-nums; }
.grow { flex: 1; }
.cm { border-collapse: collapse; font-size: 11px; font-variant-numeric: tabular-nums; }
.cm th { font-weight: 500; color: var(--muted); padding: 0 6px; text-align: right; white-space: nowrap; }
.cm td { padding: 1px 8px; text-align: center; font-weight: 600; border: 1px solid var(--line); cursor: pointer; min-width: 36px; }
.cm td.ok { color: var(--good); background: var(--good-bg); }
.cm td.err { color: var(--bad); background: var(--bad-bg); }
.cm td.active { outline: 2px solid var(--accent); outline-offset: -2px; }
select, button { font: inherit; color: var(--text); background: var(--panel); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; }
button { cursor: pointer; }
button:hover { border-color: var(--accent); }
.thrbar { display: flex; flex-direction: column; gap: 2px; padding: 3px 12px; background: var(--panel);
  border-bottom: 1px solid var(--line); font-size: 13px; }
/* Collapsed: only the camera tabs and the one-line summary stay. */
.thrbar.collapsed .rule:not(.summary), .thrbar.collapsed .rule.summary > :not(#ruleSummary) { display: none; }
.thrbar.collapsed .rule.summary { border-top: 0; padding-top: 0; }
.tabs { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.tabs button { border-radius: 6px 6px 0 0; border-bottom-width: 2px; }
.tabs button.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
.tabs button b { color: var(--bad); font-variant-numeric: tabular-nums; }
.rule { display: flex; align-items: center; gap: 2px 8px; flex-wrap: wrap; }
.rule.off > :not(.lbl) { opacity: .45; }
.rule .lbl { font-weight: 600; min-width: 170px; display: flex; align-items: center; gap: 6px; }
.rule input[type=range] { width: 180px; max-width: 30vw; accent-color: var(--accent); }
.rule input[type=number] { width: 72px; font: inherit; font-variant-numeric: tabular-nums; color: var(--text);
  background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1px 5px; }
#thrRange { width: 260px; }
#thrNum.changed { border-color: var(--warn); }
.rule .pct { color: var(--muted); font-size: 12px; min-width: 52px; font-variant-numeric: tabular-nums; }
.rule .flags { color: var(--muted); }
.rule .flags b { color: var(--bad); font-variant-numeric: tabular-nums; }
.rule.summary { border-top: 1px solid var(--line); padding-top: 2px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.badge.why { color: var(--muted); background: var(--bg); text-transform: none; font-weight: 500; }
.stack { position: relative; }
.stack:not(.zoom):not(.strip) { max-width: calc(80vh * 4 / 3); }  /* a full-frame 4:3 image (nearly) fits on screen */
.stack img { width: 100%; display: block; }
.stack .mask { position: absolute; inset: 0; height: 100%; pointer-events: none; image-rendering: pixelated; }
.stack .geo { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
/* Axes around an unwrapped strip: room on the left for pixel labels, below for degrees. */
.axwrap { position: relative; padding: 2px 0 20px 58px; }
.axwrap .axes { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; z-index: 1; }
.axes text { fill: var(--muted); font-size: 10px; font-variant-numeric: tabular-nums; }
.axes .tick { stroke: var(--muted); stroke-width: 1; }
.axes .ref { stroke-width: 1; stroke-dasharray: 4 3; opacity: .7; }
.axes .seam { fill: var(--muted); opacity: .45; }
.stack .geo rect { pointer-events: all; }
.zoomwrap .stack.zoom { width: 200%; }
.maskAlone { image-rendering: pixelated; cursor: default; }
#strip { flex: 1; min-width: 220px; height: 28px; display: block; }
#strip .tick { stroke-width: 2; }
#strip .t-good { stroke: var(--good); }
#strip .t-bad { stroke: var(--bad); }
#strip .t-none { stroke: var(--muted); opacity: .45; }
#strip .t-train { stroke: var(--accent); }
#strip .thr { stroke: var(--text); stroke-width: 2; stroke-dasharray: 3 2; }
#strip .axis { fill: var(--muted); font-size: 10px; }
.legend { display: flex; gap: 10px; font-size: 11px; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 3px; margin-right: 4px; vertical-align: middle; }
main { flex: 1; display: flex; min-height: 0; }
#list { width: 290px; overflow-y: auto; background: var(--panel); border-right: 1px solid var(--line); font-size: 12px; }
.row { display: grid; grid-template-columns: auto 1fr; gap: 8px; padding: 5px 8px;
  border-bottom: 1px solid var(--line); cursor: pointer; }
.row:hover { background: var(--bg); }
.row.sel { background: var(--sel); }
.thumbs { display: flex; flex-direction: column; gap: 3px; }
.row img { width: 64px; height: 48px; object-fit: cover; border-radius: 4px; background: var(--line); display: block; }
.row img.flag { outline: 2px solid var(--bad); outline-offset: -2px; }
.row .name { font-size: 11px; color: var(--muted); word-break: break-all; }
.row .meta { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; margin-top: 2px; }
.row .vs { font-size: 11px; font-variant-numeric: tabular-nums; }
.row .vs .bad { color: var(--bad); font-weight: 600; }
.badge { font-size: 11px; font-weight: 600; padding: 1px 7px; border-radius: 10px; text-transform: uppercase; }
.badge.good { color: var(--good); background: var(--good-bg); }
.badge.bad { color: var(--bad); background: var(--bad-bg); }
.badge.none { color: var(--muted); background: var(--bg); border: 1px dashed var(--line); }
.badge.mismatch { color: var(--warn); background: var(--warn-bg); }
.score { font-variant-numeric: tabular-nums; font-weight: 600; }
#viewer { flex: 1; overflow-y: auto; padding: 8px 10px; display: flex; flex-direction: column; gap: 8px; }
.toolbar { display: flex; gap: 4px 12px; align-items: center; flex-wrap: wrap; background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px; position: sticky; top: 0; z-index: 1;
  font-size: 13px; }
.toolbar button { padding: 2px 8px; }
.toolbar .title { font-weight: 600; }
.field { display: flex; flex-direction: column; gap: 2px; }
.field label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.vt { border-collapse: collapse; font-size: 11px; font-variant-numeric: tabular-nums; line-height: 1.25; }
.vt th { font-weight: 500; color: var(--muted); text-align: left; padding: 0 8px 0 0; font-size: 10px; }
.vt td { padding: 0 8px 0 0; }
.vt .badge { font-size: 10px; padding: 0 5px; }
#actual { font-size: 14px; padding: 3px 8px; min-width: 150px; }
#actual.good { border-color: var(--good); color: var(--good); }
#actual.bad { border-color: var(--bad); color: var(--bad); }
.views { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 8px; }
.imgs { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.imgs h2 { font-size: 13px; margin: 0; display: flex; gap: 8px; align-items: center; }
.imgs figure { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 4px; }
.imgs figcaption { font-size: 11px; color: var(--muted); margin-bottom: 3px; }
.imgs img { width: 100%; display: block; border-radius: 4px; cursor: zoom-in; }
.imgs img.zoom { cursor: zoom-out; width: 200%; max-width: none; }
.zoomwrap { overflow: auto; }
.missing { color: var(--muted); padding: 24px; text-align: center; border: 1px dashed var(--line); border-radius: 8px; }
.hint { color: var(--muted); font-size: 12px; }
kbd { font: 11px ui-monospace, monospace; border: 1px solid var(--line); border-bottom-width: 2px;
  border-radius: 4px; padding: 0 4px; background: var(--bg); }
#saved { font-size: 12px; color: var(--good); opacity: 0; transition: opacity .3s; }
#saved.show { opacity: 1; }
@media (max-width: 760px) {
  main { flex-direction: column; }
  #list { width: 100%; height: 38vh; border-right: 0; border-bottom: 1px solid var(--line); }
  #viewer { padding: 12px 16px; }
  .views { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
  <h1 id="h1">Inspection Review</h1>
  <span class="stat">Reviewed <b id="nrev">0</b> / <b id="ntot">0</b></span>
  <span class="stat" title="Among reviewed parts; positive class = bad.
Bad precision: of the parts the model flags bad, the share you marked bad.
Bad recall: of the parts you marked bad, the share the model flags bad.">Accuracy <b id="acc">–</b> ·
    Bad precision <b id="prec">–</b> · Bad recall <b id="rec">–</b></span>
  <table class="cm" title="Confusion matrix of reviewed parts (click a cell to show those parts)">
    <tr><th></th><th>model bad</th><th>model good</th></tr>
    <tr><th>you: bad</th><td class="ok" data-f="cm-tp" id="cmTP">0</td><td class="err" data-f="cm-fn" id="cmFN">0</td></tr>
    <tr><th>you: good</th><td class="err" data-f="cm-fp" id="cmFP">0</td><td class="ok" data-f="cm-tn" id="cmTN">0</td></tr>
  </table>
  <span class="grow"></span>
  <button id="rulesToggle" title="Show or hide the rule settings (key r)">Rules ▾</button>
  <label class="stat">Show
    <select id="filter">
      <option value="all">All parts</option>
      <option value="unreviewed">Not reviewed</option>
      <option value="pred-bad">Predicted bad</option>
      <option value="pred-good">Predicted good</option>
      <optgroup label="Flagged by view" id="viewFilters"></optgroup>
      <option value="why-max">Flagged by max score</option>
      <option value="why-region">Flagged by region rule</option>
      <option value="why-area">Flagged by total-area rule</option>
      <option value="actual-bad">Marked bad</option>
      <option value="actual-good">Marked good</option>
      <option value="mismatch">Model disagrees with me</option>
      <option value="cm-tp">Caught (bad, model bad)</option>
      <option value="cm-fn">Missed (bad, model good)</option>
      <option value="cm-fp">False alarms (good, model bad)</option>
      <option value="cm-tn">Correct pass (good, model good)</option>
    </select>
  </label>
  <label class="stat">Sort
    <select id="sort">
      <option value="margin">Worst view, furthest over its threshold</option>
      <optgroup label="By view" id="viewSorts"></optgroup>
      <option value="name">File name (capture time)</option>
    </select>
  </label>
</header>
<div class="thrbar" id="thrbar">
  <div class="tabs" id="tabs"></div>
  <div class="rule" id="ruleMax">
    <label class="lbl"><input type="checkbox" id="maxOn"> 1 · Max score &gt;</label>
    <input type="range" id="thrRange" step="0.001" aria-label="Max score threshold slider">
    <input type="number" id="thrNum" step="0.01" aria-label="Max score threshold">
    <button id="thrReset" title="Back to the threshold the model run used">Reset to model (<span id="thrModel">–</span>)</button>
    <span class="flags">flags <b id="nMax">0</b></span>
    <svg id="strip" role="img" aria-label="Max scores of all images in this view, threshold marked"></svg>
    <span class="legend"><span><i style="background:var(--bad)"></i>you: bad</span>
      <span><i style="background:var(--good)"></i>you: good</span>
      <span><i style="background:var(--muted)"></i>not reviewed</span>
      <span><i style="background:var(--accent)"></i>training</span></span>
  </div>
  <div class="rule" id="ruleRegion">
    <label class="lbl"><input type="checkbox" id="regionOn"> <span class="swatch" style="background:#ff2828"></span>
      2 · Connected region ≥</label>
    <input type="number" id="regionA" min="1" step="1" aria-label="Minimum connected area, patches"> patches
    <span class="pct" id="regionPct"></span>
    each scoring &gt;
    <input type="range" id="regionTR" step="0.001" aria-label="Region score threshold slider">
    <input type="number" id="regionT" step="0.01" aria-label="Region score threshold">
    <span class="flags">flags <b id="nRegion">0</b></span>
  </div>
  <div class="rule" id="ruleTotal">
    <label class="lbl"><input type="checkbox" id="totalOn"> <span class="swatch" style="background:#ffc800"></span>
      3 · Total area ≥</label>
    <input type="number" id="totalA" min="1" step="1" aria-label="Minimum total area, patches"> patches
    <span class="pct" id="totalPct"></span>
    each scoring &gt;
    <input type="range" id="totalTR" step="0.001" aria-label="Total-area score threshold slider">
    <input type="number" id="totalT" step="0.01" aria-label="Total-area score threshold">
    <span class="flags">flags <b id="nTotal">0</b></span>
  </div>
  <div class="rule" id="ruleWhite">
    <label class="lbl" title="Hotspots that are only brighter than the surface around them (white dots, glints, powder) are not counted by rules 1-3. Holes and pits are dark, and a chipped edge lets the dark background show, so they are never ignored."><input type="checkbox" id="whiteOn"> <span class="swatch" style="background:#00d2ff"></span>
      Ignore white spots</label>
    hotspot brighter than its surroundings by &gt;
    <input type="number" id="whiteB" min="1" step="5" aria-label="Brightness above the surface, grey levels"> grey levels
    and nothing darker than −<input type="number" id="whiteD" min="1" step="5" aria-label="Darkness below the surface, grey levels">
    <span class="flags">ignored in <b id="nWhite">0</b> images</span>
    <span class="hint" id="whiteInfo">applies to rules 1–3 · holes, pits and chipped edges are dark, so they always count</span>
  </div>
  <div id="ruleGeo" hidden>
    <div class="rule" id="geoChips">
      <label class="lbl"><input type="checkbox" id="g_chips"> <span class="swatch" style="background:#ff2828"></span> Chips</label>
      edge dented in by ≥ <input type="number" id="g_chip_depth" min="1" step="1"> px
      over ≥ <input type="number" id="g_chip_width" min="0" step="0.5">° of the ring
      <span class="flags">flags <b id="nChips">0</b></span>
    </div>
    <div class="rule" id="geoBumps">
      <label class="lbl"><input type="checkbox" id="g_bumps"> <span class="swatch" style="background:#ffa000"></span> Bumps (extra material)</label>
      edge sticking out by ≥ <input type="number" id="g_bump_depth" min="1" step="1"> px
      over ≥ <input type="number" id="g_bump_width" min="0" step="0.5">°
      <span class="flags">flags <b id="nBumps">0</b></span>
    </div>
    <div class="rule" id="geoPits">
      <label class="lbl"><input type="checkbox" id="g_pits"> <span class="swatch" style="background:#ff00ff"></span> Holes / pits</label>
      darker than the surface by ≥ <input type="number" id="g_pit_contrast" min="1" step="5"> grey levels,
      area ≥ <input type="number" id="g_pit_area" min="1" step="10"> px,
      length/width ≤ <input type="number" id="g_pit_elong" min="1" step="0.5">
      <span class="flags">flags <b id="nPits">0</b></span>
      <span class="hint">measured on the unwrapped ring, in original-image pixels; shade, streaks and white spots are ignored</span>
    </div>
  </div>
  <div class="rule summary">
    <span class="stat" id="ruleSummary"></span>
    <label class="stat" title="Blur each patch map by a Gaussian of this width (in patches) before rules 2 and 3 count patches. Thin streaks fade, solid blobs survive. 0 = off. Rule 1 always uses the raw max score.">Smoothing
      <input type="range" id="sigmaR" min="0" max="4" step="0.25" aria-label="Smoothing slider">
      <input type="number" id="sigma" min="0" max="10" step="0.25" aria-label="Smoothing, patches"> patches</label>
    <button id="areaReset" title="Score thresholds = 99th pct of good training patches (after smoothing); areas = smallest that flags no training image">Auto values for rules 2 &amp; 3</button>
    <label class="hint"><input type="checkbox" id="overlayOn" checked> show area overlay on the image</label>
    <span class="hint" id="gridInfo"></span>
  </div>
</div>
<main>
  <div id="list"></div>
  <div id="viewer">
    <div class="toolbar">
      <div class="field"><label>Part</label><span class="title" id="vtitle">–</span></div>
      <table class="vt" id="vtable"></table>
      <div class="field"><label>Part verdict</label><span id="vpred">–</span></div>
      <div class="field"><label>Actual</label>
        <select id="actual">
          <option value="">— not reviewed —</option>
          <option value="good">good</option>
          <option value="bad">bad</option>
        </select>
      </div>
      <label class="hint"><input type="checkbox" id="advance" checked> next part after labeling</label>
      <span id="saved">saved</span>
      <span class="grow"></span>
      <button id="prev">← Prev</button><button id="next">Next →</button>
      <span class="hint"><kbd>↑</kbd><kbd>↓</kbd> move · <kbd>g</kbd> good · <kbd>b</kbd> bad · <kbd>u</kbd> clear · <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> camera · <kbd>r</kbd> rules · click image to zoom</span>
    </div>
    <div class="views" id="views"></div>
  </div>
</main>
<script>
// V[v]: one camera view = { name, modelThr, maps, fullFrame, items, R } where R holds its rules.
// parts: one entry per physical part = { key, actual, its: [item of view v, or null] }.
let V = [], parts = [], view = [], cur = 0, tab = 0;
const $ = id => document.getElementById(id);
const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
const multi = () => V.length > 1;
// Which rules of view v fire for its image; the image is bad if any does.
// Rule 1; with the white-spot filter the server decides, as white hotspots above the threshold don't count.
const maxFires = (v, it) => V[v].R.whiteOn && it.maxOk != null ? !!it.maxOk : it.score > V[v].R.thr;
// Geometry views (ring_defects.py): which measured chips / bumps / pits count under the rules.
// Same logic as ring_defects.passes and .severity.
const GEO_KEYS = ["chips", "chip_depth", "chip_width", "bumps", "bump_depth", "bump_width",
                  "pits", "pit_contrast", "pit_area", "pit_elong"];
function geoCounted(v, it) {
  const g = V[v].R.geo, d = V[v].geometry?.[it.file];
  if (!d) return [];
  return [...d.edges.filter(e => e.kind === "chip" ? g.chips && e.depth >= g.chip_depth && e.width >= g.chip_width
                                                  : g.bumps && e.depth >= g.bump_depth && e.width >= g.bump_width),
          ...d.pits.filter(p => g.pits && p.contrast >= g.pit_contrast && p.area >= g.pit_area && p.elong <= g.pit_elong)];
}
function geoSeverity(v, it) {
  const g = V[v].R.geo, d = V[v].geometry?.[it.file];
  if (!d) return 0;
  const s = [...d.edges.filter(e => e.kind === "chip" ? g.chips : g.bumps).map(e => e.kind === "chip"
               ? Math.min(e.depth / g.chip_depth, e.width / (g.chip_width || 1e-9))
               : Math.min(e.depth / g.bump_depth, e.width / (g.bump_width || 1e-9))),
             ...d.pits.filter(p => g.pits && p.elong <= g.pit_elong).map(p => Math.min(p.contrast / g.pit_contrast, p.area / g.pit_area))];
  return s.length ? Math.max(...s) : 0;
}
// After a geometry rule changes: each image's score is its severity (1 = exactly at the thresholds).
function geoRescore(v) { for (const it of V[v].items) it.score = geoSeverity(v, it); }
const fires = (v, it) => {
  const f = [], R = V[v].R, maps = V[v].maps;
  if (!it) return f;
  if (V[v].geometry) {
    const kinds = new Set(geoCounted(v, it).map(c => c.kind || "pit"));
    return ["chip", "bump", "pit"].filter(k => kinds.has(k));
  }
  if (R.maxOn && maxFires(v, it)) f.push("max");
  if (maps && R.regionOn && it.region >= R.regionA) f.push("region");
  if (maps && R.totalOn && it.total >= R.totalA) f.push("area");
  return f;
};
const viewBad = (p, v) => fires(v, p.its[v]).length > 0;
// The part is bad if ANY view flags it.
const pred = p => V.some((_, v) => viewBad(p, v)) ? "bad" : "good";
const cell = p => (pred(p) === "bad" ? (p.actual === "bad" ? "tp" : "fp") : (p.actual === "bad" ? "fn" : "tn"));
const pct = (v, a) => V[v].maps ? `${(100 * a / (V[v].maps.grid[0] * V[v].maps.grid[1])).toFixed(2)}%` : "";
// How far the part's worst view is over (or under) that view's max-score threshold, relative to it.
const margin = p => Math.max(...V.map((_, v) => p.its[v] ? p.its[v].score / V[v].R.thr - 1 : -Infinity));
const enc = encodeURIComponent;
const imgUrl = (v, file, kind) => `/img/${enc(file)}?view=${v}&kind=${kind}`;

function badge(x) { return x ? `<span class="badge ${x}">${x}</span>` : `<span class="badge none">not reviewed</span>`; }
const whyHtml = p => V.map((view, v) => fires(v, p.its[v]).map(w =>
  `<span class="badge why">${multi() ? view.name + ": " : ""}${w}</span>`).join(" ")).join(" ");

function applyView(keepKey) {
  const f = $("filter").value, s = $("sort").value;
  view = parts.filter(p => f === "all" || (f === "unreviewed" && !p.actual) ||
    (f === "pred-bad" && pred(p) === "bad") || (f === "pred-good" && pred(p) === "good") ||
    (f.startsWith("view-") && viewBad(p, +f.slice(5))) ||
    (f.startsWith("only-") && viewBad(p, +f.slice(5)) && V.every((_, v) => v === +f.slice(5) || !viewBad(p, v))) ||
    (f.startsWith("why-") && V.some((_, v) => fires(v, p.its[v]).includes(f.slice(4)))) ||
    (f === "actual-bad" && p.actual === "bad") || (f === "actual-good" && p.actual === "good") ||
    (f === "mismatch" && p.actual && p.actual !== pred(p)) ||
    (f.startsWith("cm-") && p.actual && cell(p) === f.slice(3)));
  // Parts without an image in view v (or without its value) go last either way.
  const by = (v, k, dir = -1) => (a, b) => dir * ((a.its[v]?.[k] ?? Infinity * dir) - (b.its[v]?.[k] ?? Infinity * dir))
    || a.key.localeCompare(b.key);
  let cmp = (a, b) => margin(b) - margin(a) || a.key.localeCompare(b.key);
  if (s === "name") cmp = (a, b) => a.key.localeCompare(b.key);
  const m = s.match(/^(score|score-asc|region|total)-(\d+)$/);
  if (m) cmp = m[1] === "score-asc" ? by(+m[2], "score", 1) : by(+m[2], m[1]);
  view.sort(cmp);
  const i = keepKey ? view.findIndex(p => p.key === keepKey) : -1;
  cur = i >= 0 ? i : Math.min(cur, Math.max(0, view.length - 1));
  renderList(); show(cur);
}

function rowHtml(p, i) {
  const mm = p.actual && p.actual !== pred(p) ? `<span class="badge mismatch">disagree</span>` : "";
  // Thumbnail of the selected camera only (the scores below cover every camera).
  const w = V[tab], it0 = p.its[tab];
  const thumbs = it0
    ? `<img loading="lazy" class="${viewBad(p, tab) ? "flag" : ""}" src="${imgUrl(tab, it0.file, "thumb")}" alt="${w.name}" title="${w.name}">`
    : `<img alt="${w.name}: missing" title="${w.name}: no image">`;
  const scores = V.map((view, v) => {
    const it = p.its[v];
    if (!it) return `<div>${view.name}: –</div>`;
    const extra = view.maps && it.region !== undefined
      ? ` <span class="hint" title="largest connected region · total area, in patches">· ${it.region}/${it.total}</span>` : "";
    return `<div><span class="${viewBad(p, v) ? "bad" : ""}">${multi() ? view.name + " " : ""}#${it.rank} · ${it.score.toFixed(3)}</span>${extra}</div>`;
  }).join("");
  const train = V.filter((_, v) => p.its[v]?.in_training).map(w => w.name);
  return `<div class="row${i === cur ? " sel" : ""}" data-i="${i}">
    <div class="thumbs">${thumbs}</div>
    <div><div class="vs">${scores}</div>
      <div class="name">${p.key}${train.length ? ` (training${multi() ? " in " + train.join(", ") : ""})` : ""}</div>
      <div class="meta">model ${badge(pred(p))} ${whyHtml(p)} you ${badge(p.actual)} ${mm}</div></div></div>`;
}

function renderList() {
  $("list").innerHTML = view.length ? view.map(rowHtml).join("") : `<p class="hint" style="padding:16px">No parts match this filter.</p>`;
  stats();
}

function stats() {
  const rev = parts.filter(p => p.actual);
  $("nrev").textContent = rev.length; $("ntot").textContent = parts.length;
  const n = { tp: 0, fn: 0, fp: 0, tn: 0 };
  for (const p of rev) n[cell(p)]++;
  const pc = (a, b) => b ? `${(100 * a / b).toFixed(1)}% (${a}/${b})` : "–";
  $("acc").textContent = pc(n.tp + n.tn, rev.length);
  $("prec").textContent = pc(n.tp, n.tp + n.fp);
  $("rec").textContent = pc(n.tp, n.tp + n.fn);
  for (const k in n) $("cm" + k.toUpperCase()).textContent = n[k];
  document.querySelectorAll(".cm td").forEach(td => td.classList.toggle("active", td.dataset.f === $("filter").value));
  const nb = parts.filter(p => pred(p) === "bad").length;
  const perView = V.map((w, v) => `${w.name} <b>${parts.filter(p => viewBad(p, v)).length}</b>`).join(" · ");
  $("ruleSummary").innerHTML = (multi()
    ? `Part is bad if <b>any</b> checked rule fires in <b>any</b> view. Flagged by ${perView} → `
    : `Bad if <b>any</b> checked rule fires → `) + `predicted bad <b>${nb}</b> · good <b>${parts.length - nb}</b>`;
  // Per-rule counts are independent of whether the rule is enabled, so you can see what it would do.
  const its = V[tab].items, R = V[tab].R;
  if (V[tab].geometry) {
    const on = { ...R.geo, chips: true, bumps: true, pits: true }, saved = R.geo;
    R.geo = on;  // count what each rule would flag even when it is switched off
    const n = k => its.filter(it => geoCounted(tab, it).some(c => (c.kind || "pit") === k)).length;
    $("nChips").textContent = n("chip"); $("nBumps").textContent = n("bump"); $("nPits").textContent = n("pit");
    R.geo = saved;
    V.forEach((w, v) => { const b = document.querySelector(`#tabs button[data-v="${v}"] b`);
                          if (b) b.textContent = parts.filter(p => viewBad(p, v)).length; });
    return;
  }
  $("nMax").textContent = its.filter(it => maxFires(tab, it)).length;
  $("nWhite").textContent = R.whiteOn ? its.filter(it => it.ignored > 0).length : "–";
  $("nRegion").textContent = its.filter(it => it.region >= R.regionA).length;
  $("nTotal").textContent = its.filter(it => it.total >= R.totalA).length;
  V.forEach((w, v) => { const b = document.querySelector(`#tabs button[data-v="${v}"] b`);
                        if (b) b.textContent = parts.filter(p => viewBad(p, v)).length; });
  drawStrip();
}

// One tick per image of the selected view along the score axis, coloured by your label,
// so you can see where bad parts sit relative to the threshold. Click it to move the threshold.
const PAD = 8;
function stripScale() {
  const w0 = V[tab], s = w0.items.map(it => it.score), w = $("strip").clientWidth || 400;
  const lo = Math.min(...s, w0.modelThr), hi = Math.max(...s, w0.modelThr);
  return { w, lo, hi, x: v => PAD + (v - lo) / (hi - lo || 1) * (w - 2 * PAD) };
}
function drawStrip() {
  if (!V.length) return;
  const { w, lo, hi, x } = stripScale(), svg = $("strip");
  svg.setAttribute("viewBox", `0 0 ${w} 34`);
  const order = { none: 0, good: 1, train: 2, bad: 3 };  // draw bad on top
  const ticks = V[tab].items.map(it => ({ it, c: it.p.actual || (it.in_training ? "train" : "none") }))
    .sort((a, b) => order[a.c] - order[b.c])
    .map(({ it, c }) => `<line class="tick t-${c}" x1="${x(it.score).toFixed(1)}" x2="${x(it.score).toFixed(1)}" y1="4" y2="${c === "none" ? 16 : 22}"><title>${it.file}: ${it.score.toFixed(3)}</title></line>`);
  svg.innerHTML = ticks.join("") +
    `<line class="thr" x1="${x(V[tab].R.thr)}" x2="${x(V[tab].R.thr)}" y1="0" y2="24"></line>` +
    `<text class="axis" x="${PAD}" y="33">${lo.toFixed(2)}</text>` +
    `<text class="axis" x="${w - PAD}" y="33" text-anchor="end">${hi.toFixed(2)}</text>`;
}

function setThreshold(val, from) {
  if (!Number.isFinite(val)) return;
  const w = V[tab];
  w.R.thr = Math.round(val * 1000) / 1000;
  if (from !== "range") $("thrRange").value = w.R.thr;
  if (from !== "num") $("thrNum").value = w.R.thr.toFixed(3);
  $("thrNum").classList.toggle("changed", w.R.thr !== w.modelThr);
  store.set(`review.thr.${w.id}`, JSON.stringify({ modelThr: w.modelThr, thr: w.R.thr }));
  if (w.R.whiteOn) fetchAreasSoon(tab);  // which hotspots above the threshold are white spots comes from the server
  applyView(view[cur]?.key);
}

function buildViewers() {
  $("views").innerHTML = V.map((w, v) => `<div class="imgs" id="col${v}">
    ${multi() ? `<h2>${w.name} <span id="vb${v}"></span><span class="hint">· other cameras: pick their tab above, or press ${V.map((_, i) => i + 1).join("/")}</span></h2>` : ""}
    <div class="missing" id="miss${v}" hidden>No ${w.name} image for this part</div>
    <figure id="of${v}"><figcaption>${w.region === "unwrap" ? "Unwrapped ring (hole side at the top)" : "Original image"}<span id="ovlCap${v}"> · red = largest connected region (rule 2), yellow = patches counted by rule 3, cyan = ignored white spots</span></figcaption>
      <div class="zoomwrap"><div class="${w.strip ? "axwrap" : ""}" id="ax${v}">${w.strip ? `<svg class="axes" id="axsvg${v}"></svg>` : ""}
        <div class="stack${w.region === "unwrap" ? " strip" : ""}" id="stack${v}"><img id="orig${v}" alt=""><img id="mask${v}" class="mask" alt="">
        <svg id="geo${v}" class="geo" viewBox="0 0 ${STRIP_W} ${STRIP_H}" preserveAspectRatio="none"></svg></div></div></div>
      ${w.strip ? `<div class="hint" id="cursor${v}">x: angle around the ring · y: original-image pixels from the hole edge (outwards +) — the units of the depth thresholds</div>` : ""}
      ${w.geometry ? `<div class="hint">red = chip, orange = bump, magenta = hole/pit; dashed grey = near miss (hover for its measurements)</div>` : ""}</figure>
    <figure id="maskFig${v}" hidden><figcaption>Area rules on the patch grid (this run scores a crop, so it is not overlaid):
      red = largest connected region (rule 2), yellow = patches counted by rule 3</figcaption>
      <img id="maskAlone${v}" class="maskAlone" alt=""></figure>
    <figure id="hf${v}"><figcaption>${w.geometry
      ? "Measurement: traced edges in green, chips red, bumps orange, pits magenta, at the default thresholds (grey = below them)"
      : "Model heatmap: red = where the part looks least like the good training images (the verdict printed in this picture is from the original run's threshold)"}</figcaption>
      <div class="zoomwrap"><img id="heat${v}" alt=""></div></figure></div>`).join("");
  V.forEach((w, v) => {
    if (w.strip) {
      $("orig" + v).addEventListener("load", () => drawAxes(v));
      $("stack" + v).addEventListener("mousemove", e => stripCursor(v, e));
    }
    $("orig" + v).onclick = () => { $("stack" + v).classList.toggle("zoom"); drawAxes(v); };
    $("heat" + v).onclick = () => $("heat" + v).classList.toggle("zoom");
  });
}

function show(i) {
  if (!view.length) return;
  cur = Math.max(0, Math.min(i, view.length - 1));
  const p = view[cur];
  document.querySelectorAll(".row.sel").forEach(r => r.classList.remove("sel"));
  const row = document.querySelector(`.row[data-i="${cur}"]`);
  if (row) { row.classList.add("sel"); row.scrollIntoView({ block: "nearest" }); }
  $("vtitle").textContent = p.key;
  $("vtable").innerHTML = `<tr>${multi() ? "<th>View</th>" : ""}<th>Rank</th><th>Max score</th><th>Largest region</th><th>Total area</th><th>Says</th></tr>` +
    V.map((w, v) => { const it = p.its[v];
      return `<tr>${multi() ? `<td>${w.name}</td>` : ""}${it ? `<td>#${it.rank}${it.in_training ? " (training)" : ""}</td>
        <td class="score">${it.score.toFixed(3)}</td>
        <td>${it.region !== undefined ? `${it.region} (${pct(v, it.region)})` : "–"}</td>
        <td>${it.total !== undefined ? `${it.total} (${pct(v, it.total)})` : "–"}</td>
        <td>${badge(viewBad(p, v) ? "bad" : "good")}</td>` : `<td colspan="5" class="hint">no image</td>`}</tr>`; }).join("");
  $("vpred").innerHTML = badge(pred(p)) + " " + whyHtml(p);
  $("actual").value = p.actual; $("actual").className = p.actual;
  // Only the camera of the selected rules tab is shown.
  V.forEach((w, v) => {
    const it = p.its[v];
    $("col" + v).hidden = v !== tab;
    if (v !== tab) return;
    for (const id of ["of", "hf"]) $(id + v).hidden = !it;
    $("miss" + v).hidden = !!it;
    if (multi()) $("vb" + v).innerHTML = it ? badge(viewBad(p, v) ? "bad" : "good") : "";
    if (!it) { $("maskFig" + v).hidden = true; return; }
    $("orig" + v).src = imgUrl(v, it.file, "orig");
    showMask(v, it);
    $("heat" + v).src = it.heatmap ? imgUrl(v, it.heatmap, "heat") : "";
  });
  store.set("review.part", p.key);
  if (V[tab].strip) requestAnimationFrame(() => drawAxes(tab));  // a cached image fires no load event
  // Warm the cache for the next part so stepping through feels instant.
  const nx = view[cur + 1]?.its[tab];
  if (nx) { new Image().src = imgUrl(tab, nx.file, "orig");
            if (nx.heatmap) new Image().src = imgUrl(tab, nx.heatmap, "heat"); }
}

// Red/yellow patch overlay for the enabled area rules: on top of the image when the
// map covers the full frame, otherwise as a separate picture of the patch grid.
function showMask(v, it) {
  const w = V[v], R = w.R;
  if (w.geometry) return showGeo(v, it);
  const on = w.maps && $("overlayOn").checked && (R.regionOn || R.totalOn);
  const qs = `view=${v}&file=${enc(it.file)}&sigma=${R.sigma}` + (R.regionOn ? `&region=${R.regionT}` : "") +
             (R.totalOn ? `&total=${R.totalT}` : "") + whiteQs(R);
  $("mask" + v).hidden = !(on && w.fullFrame); $("ovlCap" + v).hidden = !(on && w.fullFrame);
  $("maskFig" + v).hidden = !(on && !w.fullFrame);
  if (on && w.fullFrame) $("mask" + v).src = `/mask?${qs}`;
  if (on && !w.fullFrame) $("maskAlone" + v).src = `/mask?${qs}&bg=1`;
}

// Geometry view: outline the chips, bumps and pits that count on the strip (strip pixels), and
// faintly the near misses (at least half way to the depth / contrast threshold).
const STRIP_W = 2176, STRIP_H = 256, LOOP = 2048;
function showGeo(v, it) {
  $("mask" + v).hidden = true; $("maskFig" + v).hidden = true; $("ovlCap" + v).hidden = true;
  const svg = $("geo" + v), g = V[v].R.geo, d = V[v].geometry[it.file];
  if (!d) { svg.innerHTML = ""; return; }
  const counted = new Set(geoCounted(v, it));
  const near = c => c.kind ? c.depth >= 0.5 * (c.kind === "chip" ? g.chip_depth : g.bump_depth)
                           : c.contrast >= 0.5 * g.pit_contrast && c.area >= 0.5 * g.pit_area;
  const tip = c => c.kind ? `${c.kind}, ${c.edge} edge: ${c.depth} px deep, ${c.width}° wide, at ${c.angle}°`
                          : `pit: ${c.contrast} grey levels darker, ${c.area} px, length/width ${c.elong}`;
  const out = [];
  for (const e of d.edges) {
    if (!counted.has(e) && !near(e)) continue;
    const col = e.kind === "chip" ? "#ff2828" : "#ffa000";
    // An edge run near the start of the loop also shows in the repeated seam at the strip's end.
    for (const x of e.col < STRIP_W - LOOP ? [e.col, e.col + LOOP] : [e.col])
      out.push(`<rect x="${x}" y="${e.row - 14}" width="${Math.max(e.cols, 4)}" height="28" fill="none"
        stroke="${counted.has(e) ? col : "#aaa"}" stroke-width="${counted.has(e) ? 4 : 2}"
        ${counted.has(e) ? "" : 'stroke-dasharray="6 4"'}><title>${tip(e)}</title></rect>`);
  }
  for (const p of d.pits) {
    if (!counted.has(p) && !near(p)) continue;
    out.push(`<rect x="${p.x0 - 5}" y="${p.y0 - 5}" width="${p.x1 - p.x0 + 10}" height="${p.y1 - p.y0 + 10}" fill="none"
      stroke="${counted.has(p) ? "#ff00ff" : "#aaa"}" stroke-width="${counted.has(p) ? 4 : 2}"
      ${counted.has(p) ? "" : 'stroke-dasharray="6 4"'}><title>${tip(p)}</title></rect>`);
  }
  svg.innerHTML = out.join("");
}

// Axes of an unwrapped strip, in screen pixels around the image: x = angle around the ring
// (columns 0-2048 are 0-360 deg; the rest repeats the start), y = original-image pixels from the
// hole edge, from this image's hole radius and the band the strip spans.
const ringScale = (v, file) => {
  const s = V[v].strip?.[file];
  if (!s) return null;
  const [rIn, rFrom, rTo] = s;
  // strip row (0..STRIP_H-1) <-> pixels from the hole edge
  return { rIn, rowOf: d => ((1 + d / rIn) - rFrom) / (rTo - rFrom) * (STRIP_H - 1),
           pxOf: row => (rFrom + row / (STRIP_H - 1) * (rTo - rFrom) - 1) * rIn };
};
function drawAxes(v) {
  const it = view[cur]?.its[v], svg = $("axsvg" + v), img = $("orig" + v);
  if (!svg || !it || !img.clientWidth) return;
  const sc = ringScale(v, it.file);
  if (!sc) { svg.innerHTML = ""; return; }
  // Position of the image within the axes box (the stack's offsetParent is the .axwrap).
  const st = $("stack" + v), x0 = st.offsetLeft, y0 = st.offsetTop, w = img.clientWidth, h = img.clientHeight;
  const X = col => x0 + col / STRIP_W * w, Y = row => y0 + row / (STRIP_H - 1) * h;
  const out = [];
  for (let deg = 0; deg <= 405; deg += 30) {
    const col = deg * LOOP / 360;
    if (col > STRIP_W) break;
    const seam = deg > 360;
    out.push(`<line class="tick" x1="${X(col)}" x2="${X(col)}" y1="${y0 + h}" y2="${y0 + h + 4}"/>`,
             `<text x="${X(col)}" y="${y0 + h + 15}" text-anchor="middle" ${seam ? 'class="seam"' : ""}>${seam ? deg - 360 : deg}°</text>`);
  }
  out.push(`<rect class="seam" x="${X(LOOP)}" y="${y0 + h}" width="${X(STRIP_W) - X(LOOP)}" height="3"><title>repeat of the start (0°…), so a defect at 0°/360° shows whole</title></rect>`);
  // y ticks every 25 or 50 px, whichever gives ~5-10 labels
  const lo = sc.pxOf(0), hi = sc.pxOf(STRIP_H - 1), step = (hi - lo) > 300 ? 50 : 25;
  for (let d = Math.ceil(lo / step) * step; d <= hi; d += step) {
    const y = Y(sc.rowOf(d));
    out.push(`<line class="tick" x1="${x0 - 4}" x2="${x0}" y1="${y}" y2="${y}"/>`,
             `<text x="${x0 - 6}" y="${y + 3}" text-anchor="end">${d}</text>`);
  }
  out.push(`<text transform="translate(8 ${y0 + h / 2}) rotate(-90)" text-anchor="middle">px from hole edge</text>`);
  // Dashed reference lines: the hole edge and where the ring's outer rim is expected.
  const rim = (V[v].outer - 1) * sc.rIn;
  for (const [d, col, label] of [[0, "#3fb950", "hole edge"], [rim, "#58a6ff", "rim"]]) {
    const y = Y(sc.rowOf(d));
    if (y < y0 || y > y0 + h) continue;
    out.push(`<line class="ref" x1="${x0}" x2="${x0 + w}" y1="${y}" y2="${y}" stroke="${col}"/>`,
             `<text x="${x0 + w - 4}" y="${y - 3}" text-anchor="end" style="fill:${col}">${label}</text>`);
  }
  svg.innerHTML = out.join("");
}
// Cursor readout: angle and pixels from the hole edge under the pointer.
function stripCursor(v, e) {
  const it = view[cur]?.its[v], sc = it && ringScale(v, it.file), img = $("orig" + v);
  if (!sc) return;
  const r = img.getBoundingClientRect();
  const col = (e.clientX - r.left) / r.width * STRIP_W, row = (e.clientY - r.top) / r.height * (STRIP_H - 1);
  if (col < 0 || col > STRIP_W || row < 0 || row > STRIP_H - 1) return;
  $("cursor" + v).textContent = `cursor: ${((col % LOOP) * 360 / LOOP).toFixed(1)}° around the ring, ` +
    `${sc.pxOf(row).toFixed(0)} px from the hole edge (1 px here = ${(sc.pxOf(1) - sc.pxOf(0)).toFixed(2)} px of the photo radially)`;
}

// Largest connected area / total area per image depend on the two score thresholds,
// so they come from the server; the area comparisons then run here, instantly.
const areaReq = [], areaTimer = [];
const whiteQs = R => R.whiteOn ? `&max=${R.thr}&bright=${R.whiteB}&dark=${R.whiteD}` : "";
async function fetchAreas(v) {
  const w = V[v], R = w.R;
  if (!w.maps) return;
  const id = areaReq[v] = (areaReq[v] || 0) + 1;
  // The first white-spot request per camera waits while the server measures brightness (about a minute).
  if (R.whiteOn && v === tab) $("whiteInfo").textContent = "measuring white spots…";
  const d = await (await fetch(`/api/areas?view=${v}&region=${R.regionT}&total=${R.totalT}&sigma=${R.sigma}` + whiteQs(R))).json();
  if (id !== areaReq[v]) return;  // a newer request superseded this one
  if (v === tab) $("whiteInfo").textContent = "applies to rules 1–3 · holes, pits and chipped edges are dark, so they always count";
  for (const it of w.items) { const s = d[it.file]; it.region = s?.[0]; it.total = s?.[1]; it.maxOk = s?.[2]; it.ignored = s?.[3]; }
  applyView(view[cur]?.key);
}
const fetchAreasSoon = v => { clearTimeout(areaTimer[v]); areaTimer[v] = setTimeout(() => fetchAreas(v), 120); };

function saveRules(v) {
  const { thr, ...rules } = V[v].R;  // the max-score threshold is stored with the run's threshold, below
  store.set(`review.rules.${V[v].id}`, JSON.stringify(rules));
}
// Show the rules of the selected view (tab) in the rules bar.
function syncRuleInputs() {
  const w = V[tab], R = w.R;
  // A geometry view has its own rules (chips, bumps, pits) instead of rules 1-3.
  const geo = !!w.geometry;
  $("ruleGeo").hidden = !geo;
  for (const id of ["ruleMax", "ruleRegion", "ruleTotal", "ruleWhite", "areaReset"]) $(id).hidden = geo;
  $("overlayOn").parentElement.hidden = geo; $("sigma").parentElement.hidden = geo;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", +b.dataset.v === tab));
  if (geo) {
    for (const k of GEO_KEYS) {
      if (typeof R.geo[k] === "boolean") $("g_" + k).checked = R.geo[k]; else $("g_" + k).value = R.geo[k];
    }
    for (const [row, k] of [["geoChips", "chips"], ["geoBumps", "bumps"], ["geoPits", "pits"]])
      $(row).classList.toggle("off", !R.geo[k]);
    $("gridInfo").textContent = `${w.name}: chips and holes measured on the unwrapped ring (no training)`;
    return;
  }
  $("maxOn").checked = R.maxOn; $("ruleMax").classList.toggle("off", !R.maxOn);
  const { lo, hi } = stripScale();
  $("thrRange").min = lo.toFixed(3); $("thrRange").max = hi.toFixed(3);
  $("thrRange").value = R.thr; $("thrNum").value = R.thr.toFixed(3);
  $("thrNum").classList.toggle("changed", R.thr !== w.modelThr);
  $("thrModel").textContent = w.modelThr.toFixed(3);
  for (const k of ["region", "total"]) {
    $(k + "On").checked = R[k + "On"]; $("rule" + k[0].toUpperCase() + k.slice(1)).classList.toggle("off", !R[k + "On"]);
    $(k + "A").value = R[k + "A"]; $(k + "Pct").textContent = pct(tab, R[k + "A"]);
    $(k + "T").value = R[k + "T"].toFixed(3); $(k + "TR").value = R[k + "T"];
    if (w.maps) { $(k + "TR").min = w.maps.range[0].toFixed(3); $(k + "TR").max = w.maps.range[1].toFixed(3); }
  }
  $("sigma").value = R.sigma; $("sigmaR").value = R.sigma;
  $("whiteOn").checked = R.whiteOn; $("ruleWhite").classList.toggle("off", !R.whiteOn);
  $("whiteB").value = R.whiteB; $("whiteD").value = R.whiteD;
  for (const id of ["ruleRegion", "ruleTotal", "ruleWhite", "areaReset"]) $(id).hidden = !w.maps;
  $("overlayOn").parentElement.hidden = !w.maps; $("sigma").parentElement.hidden = !w.maps;
  $("gridInfo").textContent = w.maps
    ? `${multi() ? w.name + ": " : ""}patch grid ${w.maps.grid[0]}×${w.maps.grid[1]} = ${w.maps.grid[0] * w.maps.grid[1]} patches`
    : "Area rules need maps.npz in this results folder (re-run patchcore_score.py).";
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", +b.dataset.v === tab));
}
// Score thresholds -> 99th pct of good patches (at the current smoothing); areas -> one
// more than the largest any training (good) image has, i.e. the smallest values that flag
// no training image.
async function autoAreaRules(v) {
  const R = V[v].R;
  R.regionT = R.totalT = await (await fetch(`/api/default_t?view=${v}&sigma=${R.sigma}`)).json();
  await fetchAreas(v);
  const train = V[v].items.filter(it => it.in_training && it.region !== undefined);
  R.regionA = Math.max(1, ...train.map(it => it.region + 1));
  R.totalA = Math.max(1, ...train.map(it => it.total + 1));
  if (v === tab) syncRuleInputs();
  saveRules(v); applyView(view[cur]?.key);
}

// The selected tab picks the camera whose rules are edited AND whose images are shown.
function selectTab(v) {
  if (!(v >= 0 && v < V.length) || v === tab) return;
  tab = v; store.set("review.tab", tab);
  syncRuleInputs(); applyView(view[cur]?.key);
}

async function setLabel(label) {
  const p = view[cur]; if (!p) return;
  const r = await fetch("/api/label", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ part: p.key, label }) });
  if (!r.ok) { alert("Could not save the label: " + await r.text()); return; }
  p.actual = label;
  $("saved").classList.add("show"); setTimeout(() => $("saved").classList.remove("show"), 700);
  const row = document.querySelector(`.row[data-i="${cur}"]`);
  if (row) row.outerHTML = rowHtml(p, cur);
  stats();
  if (label && $("advance").checked) show(cur + 1); else show(cur);
}

$("actual").addEventListener("change", e => { setLabel(e.target.value); e.target.blur(); });
$("list").addEventListener("click", e => { const r = e.target.closest(".row"); if (r) show(+r.dataset.i); });
$("prev").onclick = () => show(cur - 1);
$("next").onclick = () => show(cur + 1);
$("filter").onchange = () => { store.set("review.filter", $("filter").value); applyView(view[cur]?.key); };
$("sort").onchange = () => { store.set("review.sort", $("sort").value); applyView(view[cur]?.key); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey) return;
  const k = e.key;
  if (k === "ArrowDown" || k === "j") { show(cur + 1); e.preventDefault(); }
  else if (k === "ArrowUp" || k === "k") { show(cur - 1); e.preventDefault(); }
  else if (k === "g") setLabel("good");
  else if (k === "b") setLabel("bad");
  else if (k === "u") setLabel("");
  else if (/^[1-9]$/.test(k)) selectTab(+k - 1);
  else if (k === "r") toggleRules();
});
// Fold the rule rows away to give the images more room; remembered in the browser.
function toggleRules(collapsed = !$("thrbar").classList.contains("collapsed")) {
  $("thrbar").classList.toggle("collapsed", collapsed);
  $("rulesToggle").textContent = collapsed ? "Rules ▸" : "Rules ▾";
  store.set("review.rulesCollapsed", collapsed ? "1" : "");
  if (!collapsed) drawStrip();
}
$("rulesToggle").onclick = () => toggleRules();
toggleRules(store.get("review.rulesCollapsed", "") === "1");

(async () => {
  const d = await (await fetch("/api/items")).json();
  const byKey = new Map();
  V = d.views.map((w, v) => {
    // Restore a manual threshold only if it was set against this same model run.
    let saved = null, savedR = null;
    try { saved = JSON.parse(store.get(`review.thr.${w.id}`, "null")); } catch {}
    try { savedR = JSON.parse(store.get(`review.rules.${w.id}`, "null")); } catch {}
    const R = { maxOn: true, regionOn: true, regionT: 0, regionA: 1, totalOn: true, totalT: 0, totalA: 1, sigma: 0,
                whiteOn: false, whiteB: 60, whiteD: 20,
                ...(savedR || {}), thr: saved && saved.modelThr === w.threshold ? saved.thr : w.threshold };
    for (const it of w.items) {
      if (!byKey.has(it.part)) byKey.set(it.part, { key: it.part, actual: d.labels[it.part] || "", its: d.views.map(() => null) });
      it.p = byKey.get(it.part);
      it.p.its[v] = it;
    }
    if (w.geometry) R.geo = { ...(w.geo_defaults || {}), ...(savedR?.geo || {}) };
    return { name: w.name, id: w.id, region: w.region, geometry: w.geometry, strip: w.strip, outer: w.outer,
             modelThr: w.threshold, maps: w.maps, fullFrame: w.full_frame, items: w.items, R, fresh: !savedR };
  });
  parts = [...byKey.values()];
  $("h1").textContent = `${d.title} review`;
  $("h1").title = `labels -> ${d.labels_file}`;
  $("tabs").innerHTML = multi() ? `<span class="stat">Rules for</span>` + V.map((w, v) =>
    `<button data-v="${v}">${w.name} · flags <b>0</b></button>`).join("") : "";
  $("tabs").hidden = !multi();
  $("tabs").onclick = e => { const b = e.target.closest("button"); if (b) selectTab(+b.dataset.v); };
  $("viewFilters").innerHTML = V.map((w, v) => `<option value="view-${v}">Flagged by ${w.name}</option>` +
    (multi() ? `<option value="only-${v}">Flagged only by ${w.name}</option>` : "")).join("");
  $("viewSorts").innerHTML = V.map((w, v) => `<option value="score-${v}">${w.name} score, high → low</option>
    <option value="score-asc-${v}">${w.name} score, low → high</option>` + (w.maps
    ? `<option value="region-${v}">${w.name} largest region, big → small</option>
       <option value="total-${v}">${w.name} total area, big → small</option>` : "")).join("");
  if (!V.some(w => w.maps)) for (const o of document.querySelectorAll('option[value="why-region"], option[value="why-area"]')) o.remove();
  const pick = (sel, val, dflt) => { $(sel).value = val; if ($(sel).value !== val) $(sel).value = dflt; };
  pick("filter", store.get("review.filter", "all"), "all");
  pick("sort", store.get("review.sort", "margin"), "margin");
  tab = Math.min(+store.get("review.tab", 0) || 0, V.length - 1);
  V.forEach((w, v) => { if (w.geometry) geoRescore(v); });
  buildViewers();
  syncRuleInputs();
  await Promise.all(V.map((w, v) => w.maps && w.fresh ? autoAreaRules(v) : fetchAreas(v)));
  applyView(store.get("review.part", null));
})();

$("maxOn").onchange = e => { V[tab].R.maxOn = e.target.checked; syncRuleInputs(); saveRules(tab); applyView(view[cur]?.key); };
for (const k of ["region", "total"]) {
  $(k + "On").onchange = e => { V[tab].R[k + "On"] = e.target.checked; syncRuleInputs(); saveRules(tab); applyView(view[cur]?.key); };
  $(k + "A").addEventListener("input", e => {
    const val = Math.round(+e.target.value);
    if (!(val >= 1)) return;
    V[tab].R[k + "A"] = val; $(k + "Pct").textContent = pct(tab, val); saveRules(tab); applyView(view[cur]?.key);
  });
  const setT = (val, from) => {
    if (!Number.isFinite(val)) return;
    const R = V[tab].R;
    R[k + "T"] = Math.round(val * 1000) / 1000;
    if (from !== "range") $(k + "TR").value = R[k + "T"];
    if (from !== "num") $(k + "T").value = R[k + "T"].toFixed(3);
    saveRules(tab); fetchAreasSoon(tab);
  };
  $(k + "TR").addEventListener("input", e => setT(+e.target.value, "range"));
  $(k + "T").addEventListener("change", e => setT(+e.target.value, "num"));
}
$("areaReset").onclick = () => autoAreaRules(tab);
for (const k of GEO_KEYS) $("g_" + k).addEventListener("change", e => {
  const g = V[tab].R.geo;
  if (e.target.type === "checkbox") g[k] = e.target.checked;
  else { const val = +e.target.value; if (!(val >= 0)) return; g[k] = val; }
  geoRescore(tab); syncRuleInputs(); saveRules(tab); applyView(view[cur]?.key);
});
$("whiteOn").onchange = e => { V[tab].R.whiteOn = e.target.checked; syncRuleInputs(); saveRules(tab); fetchAreas(tab); };
for (const k of ["whiteB", "whiteD"]) $(k).addEventListener("change", e => {
  const val = Math.round(+e.target.value);
  if (!(val >= 1)) return;
  V[tab].R[k] = val; saveRules(tab); fetchAreasSoon(tab);
});
// Smoothing lowers the scores, so the area thresholds usually need moving afterwards
// ("Auto values" re-derives them for the new smoothing).
const setSigma = (val, from) => {
  if (!(val >= 0)) return;
  const R = V[tab].R;
  R.sigma = Math.min(10, Math.round(val * 100) / 100);
  if (from !== "range") $("sigmaR").value = R.sigma;
  if (from !== "num") $("sigma").value = R.sigma;
  saveRules(tab); fetchAreasSoon(tab);
};
$("sigmaR").addEventListener("input", e => setSigma(+e.target.value, "range"));
$("sigma").addEventListener("change", e => setSigma(+e.target.value, "num"));
// Click a confusion-matrix cell to list those parts; click it again to show all.
document.querySelectorAll(".cm td").forEach(td => td.onclick = () => {
  $("filter").value = $("filter").value === td.dataset.f ? "all" : td.dataset.f;
  $("filter").onchange();
});
$("overlayOn").onchange = () => { const p = view[cur]; if (p) V.forEach((_, v) => p.its[v] && showMask(v, p.its[v])); };

$("thrRange").addEventListener("input", e => setThreshold(+e.target.value, "range"));
$("thrNum").addEventListener("change", e => setThreshold(+e.target.value, "num"));
$("thrReset").onclick = () => setThreshold(V[tab].modelThr);
$("strip").addEventListener("click", e => {
  const { lo, hi, w } = stripScale(), r = $("strip").getBoundingClientRect();
  const px = (e.clientX - r.left) * w / r.width;
  setThreshold(lo + (px - PAD) / (w - 2 * PAD) * (hi - lo));
});
window.addEventListener("resize", () => { drawStrip(); V.forEach((w, v) => w.strip && drawAxes(v)); });
</script>
</body>
</html>
"""


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="image_data_patchcore/Model1", type=Path)
    ap.add_argument("--results", default=None, type=Path, help="default results/<model>")
    ap.add_argument("--model", default=None, type=Path,
                    help="a part folder with one subfolder per camera (each with all/ and good/); every camera "
                         "is a view, its results in results/<model>_<camera> (replaces --data/--results/--view)")
    ap.add_argument("--view", action="append", metavar="DATA[:RESULTS]",
                    help="a camera view of the same parts (repeat for each view; replaces --data/--results). "
                         "RESULTS defaults to results/<model>, or results/<model>_<camera> for a camera folder")
    ap.add_argument("--labels", default=None, type=Path,
                    help="labels CSV (default: one view: results/<model>/labels.csv, shared by all runs on "
                         "that dataset; --model: results/<model>_parts/labels.csv; several --view: "
                         "results/parts_<view1>_<view2>/labels.csv; per part when there are several views)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ARGS = ap.parse_args()
    specs = ARGS.view or [f"{ARGS.data}:{ARGS.results}" if ARGS.results else str(ARGS.data)]
    if ARGS.model:
        specs = [str(d) for d in sorted(ARGS.model.iterdir()) if (d / "all").is_dir()]
        if not specs:
            raise SystemExit(f"no camera folders (with all/ inside) in {ARGS.model}")
    for spec in specs:
        data, _, results = spec.partition(":")
        data = Path(data)
        # A camera folder inside a part folder (Model1/CAM3) is named "Model1_CAM3" in results/
        # and in the browser's saved rules; it is shown as just "CAM3".
        uid = data.name if data.parent.name == DATA_ROOT else f"{data.parent.name}_{data.name}"
        # An explicitly named results folder (e.g. a masked or unwrapped variant) keeps its own saved rules.
        VIEWS.append({"name": data.name, "id": Path(results).name if results else uid, "data": data,
                      "results": Path(results) if results else Path("results") / uid})
    if len({v["name"] for v in VIEWS}) < len(VIEWS):
        raise SystemExit("each view needs a different data folder name (it names the view)")
    ARGS.labels = ARGS.labels or (
        Path("results") / f"{ARGS.model.name}_parts" / "labels.csv" if ARGS.model else
        Path("results") / VIEWS[0]["name"] / "labels.csv" if len(VIEWS) == 1 else
        Path("results") / ("parts_" + "_".join(v["name"] for v in VIEWS)) / "labels.csv")
    for v in VIEWS:
        if not (v["results"] / "scores.csv").exists():
            raise SystemExit(f"{v['results'] / 'scores.csv'} not found; run patchcore_score.py first")

    # Measure the white-spot filter's brightness data in the background, so it is ready when switched on.
    for v in range(len(VIEWS)):
        if (VIEWS[v]["results"] / "maps.npz").exists():
            threading.Thread(target=polarity, args=(v,), daemon=True).start()
    url = f"http://127.0.0.1:{ARGS.port}"
    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    views = ", ".join(f"{v['name']} ({v['results']})" for v in VIEWS)
    print(f"Review UI: {url}   views: {views}   (labels -> {ARGS.labels})   Ctrl+C to stop")
    if not ARGS.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
