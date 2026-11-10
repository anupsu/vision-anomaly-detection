"""Unwrap a ring-shaped part (top view, CAM2) into a straight strip.

The hole is the highest-contrast feature: a near-black disc inside the grey ring. It
is found at 1/4 resolution as the largest dark, near-circular blob not touching the
frame (the dark threshold is scanned, since exposure varies), and an ellipse is
fitted to its outline (a slightly tilted camera makes circles ellipses). The ring is
then sampled on ellipses concentric with the hole, from r_from to r_to times its
size: rows = radius (hole side at the top), columns = angle. A rotated part becomes a
sideways shift of the strip and both edges become straight lines. The strip ends with
a repeat of its first `seam` columns, so a defect at 0/360 degrees appears whole once.

Kept free of torch so review_app.py can show the strips.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter

OUTER = 1.85  # outer / inner radius of the Model1 ring (median over the CAM2 images)
R_FROM, R_TO = 0.85, 2.15  # band unwrapped, in hole radii: ~50 px margin inside the hole and outside the rim
HEIGHT, WIDTH, SEAM = 256, 2048, 128  # ~1.6 px per row and ~1.5 px per column at mid-ring, near native resolution


def find_hole(img, scale=4):
    """The hole's fitted ellipse (cx, cy, semi-axis a, semi-axis b, angle in degrees), original pixels."""
    import cv2  # only needed to locate rings, not to unwrap with known ellipses
    a = np.asarray(img.convert("L").reduce(scale), dtype=np.float32)
    b = gaussian_filter(a, 3)
    h, w = b.shape
    best = None
    # Opening removes dark connections a few pixels wide, so a hole joined to the background
    # through a slit or a narrow chip (a split ring) is still found on its own.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for t in np.linspace(np.percentile(b, 1) + 2, np.percentile(b, 60), 25):
        dark = cv2.morphologyEx((b < t).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if x == 0 or y == 0 or x + bw >= w or y + bh >= h or area < 500:
                continue  # background touches the frame; specks are too small
            if area / (np.pi * (max(bw, bh) / 2) ** 2) > 0.8 and (best is None or area > best[0]):
                best = (area, (lab == i).astype(np.uint8))
    if best is None:
        raise ValueError("no hole found")
    contours, _ = cv2.findContours(best[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(max(contours, key=len))
    return cx * scale, cy * scale, d1 * scale / 2, d2 * scale / 2, ang


def unwrap(img, ell):
    """(HEIGHT, WIDTH + SEAM) RGB strip of the ring around the hole ellipse `ell` =
    (cx, cy, a, b, angle[, r_from, r_to]); the band defaults to R_FROM..R_TO hole radii."""
    cx, cy, ea, eb, ang = ell[:5]
    r_from, r_to = ell[5:7] if len(ell) >= 7 else (R_FROM, R_TO)
    phi = np.radians(ang)
    k = np.linspace(r_from, r_to, HEIGHT)[:, None]
    # Anticlockwise on screen (image y points down), so engraved text reads unmirrored in the strip.
    th = -(np.arange(WIDTH + SEAM) % WIDTH * 2 * np.pi / WIDTH)[None, :]
    ex, ey = ea * np.cos(th), eb * np.sin(th)
    x = cx + k * (ex * np.cos(phi) - ey * np.sin(phi))
    y = cy + k * (ex * np.sin(phi) + ey * np.cos(phi))
    a = gaussian_filter(np.asarray(img.convert("L"), dtype=np.float32), 0.7)  # light anti-aliasing
    s = map_coordinates(a, [y, x], order=1, mode="constant", cval=0)
    return Image.fromarray(np.clip(s, 0, 255).astype(np.uint8)).convert("RGB")


def suppress_white(strip, excess=25, window=801):
    """Clamp bright spots (white dots, glints, powder) in a strip to the surface level plus
    `excess` grey levels; darker pixels (holes, pits, background behind a chipped edge) are left
    as they are. The surface level is the median along each row over `window` columns: a row of
    the strip is one radius of the ring, so the level stays the ring's own level right up to its
    edges (a square window would mix in the background there and dim the edges), and a window
    wider than the largest powder patches is not fooled by them."""
    g = np.asarray(strip.convert("L"), dtype=np.float32)
    # Median on a 4x horizontally reduced copy (and 3 rows), then stretched back: same result, far faster.
    w4 = g.shape[1] // 4
    small = g[:, :w4 * 4].reshape(g.shape[0], w4, 4).mean(-1)
    level = median_filter(small, size=(3, window // 4), mode="wrap")
    level = np.repeat(level, 4, axis=1)
    level = np.pad(level, ((0, 0), (0, g.shape[1] - level.shape[1])), mode="edge")
    out = np.minimum(g, level + excess)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGB")


def face_rows(margin=0.06, r_from=R_FROM, r_to=R_TO, outer=OUTER):
    """Strip rows of the flat face between the two edges (where engravings are), as a slice:
    an engraving mask is limited to them, so chips on the inner and outer edges are still scored.
    `outer` is the ring's outer radius in hole radii (1.85 for Model1, 1.60 for Model2)."""
    row = lambda k: round((k - r_from) / (r_to - r_from) * (HEIGHT - 1))
    return slice(row(1 + margin), row(outer - margin) + 1)
