"""TI-RADS-motivated features computed from a 2D ultrasound image and the nodule mask.

PyRadiomics measures the nodule on its own; several TI-RADS signs are about orientation or about
the nodule against its surroundings, which no PyRadiomics feature captures. One or two features
per TI-RADS category (ACR TI-RADS; mapping in the spirit of Salmanpour et al., TU1.0):

    shape            TallerThanWideRatio   vertical extent / horizontal extent of the mask (> 1: taller than wide)
    echogenicity     EchogenicityRatio     mean brightness inside / in a ring of surrounding tissue (< 1: hypoechoic)
    echogenic foci   PunctateFociDensity   small bright spots inside the nodule per 1,000 nodule pixels
    margin           Solidity              area / convex-hull area (< 1: lobulated or irregular outline)
                     MarginSharpness       brightness gradient across the boundary, relative to the ring's spread
    composition      AnechoicFraction      share of nodule pixels much darker than the surrounding tissue

All features are computed on the image as stored (no per-image rescaling), and each is a ratio or a
density, so it does not depend on the image size or on the gain of the scanner as a whole. The
image is assumed to be oriented as acquired (depth downwards), so "vertical" is the
anteroposterior direction; the view (transverse or longitudinal) is unknown in TN3K.
These are heuristics with fixed, pre-registered settings; they are meant to be tested, not trusted.
"""

import math

import numpy as np
from scipy import ndimage as ndi

PREFIX = "original_clinical_"
NAMES = ["TallerThanWideRatio", "EchogenicityRatio", "PunctateFociDensity", "Solidity",
         "MarginSharpness", "AnechoicFraction"]
COLUMNS = [PREFIX + n for n in NAMES]

# Fixed settings (chosen before looking at labels).
RING_FRACTION = 0.25        # ring width as a fraction of the nodule's equivalent radius
MIN_RING = 4                # pixels
OUTSIDE_SECTOR = 5          # gray values below this are treated as outside the ultrasound sector
FOCI_OPENING_RADIUS = 3     # top-hat radius: bright structures narrower than ~7 px count as punctate
FOCI_Z = 6.0                # top-hat threshold: median + FOCI_Z * robust spread (MAD) inside the nodule
FOCI_MIN_AREA, FOCI_MAX_AREA = 2, 40   # pixels; single pixels are speckle, larger regions are not punctate
ANECHOIC_FRACTION_OF_RING = 0.25


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def _convex_hull_area(mask: np.ndarray) -> float:
    """Area of the convex hull of the pixel centres (shoelace formula on the hull of the boundary)."""
    ys, xs = np.nonzero(mask ^ ndi.binary_erosion(mask))
    pts = np.unique(np.column_stack([xs, ys]).astype(np.float64), axis=0)
    if len(pts) < 3:
        return float(mask.sum())
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    # Pixel centres underestimate the area by about half a pixel along the outline.
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) + 0.5 * len(hull)


def clinical_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """TI-RADS-motivated features of one nodule. `gray`: 2D image as stored; `mask`: 2D boolean."""
    gray = np.asarray(gray, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    area = float(mask.sum())
    if area == 0:
        raise ValueError("empty mask")

    rows, cols = np.nonzero(mask)
    height, width = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1

    radius = math.sqrt(area / math.pi)
    ring_width = max(MIN_RING, round(RING_FRACTION * radius))
    ring = ndi.binary_dilation(mask, _disk(ring_width)) & ~mask & (gray >= OUTSIDE_SECTOR)
    inside = gray[mask]
    ring_values = gray[ring] if ring.any() else np.array([np.nan])
    ring_mean, ring_std = float(np.mean(ring_values)), float(np.std(ring_values))

    # Echogenic foci: small structures much brighter than their neighbourhood (white top-hat) and
    # brighter than the surrounding tissue. The threshold uses a robust spread (MAD), so the foci
    # themselves do not raise it.
    core = ndi.binary_erosion(mask, iterations=2) if area > 50 else mask
    tophat = gray - ndi.grey_opening(gray, footprint=_disk(FOCI_OPENING_RADIUS))
    th = tophat[core]
    med = float(np.median(th)) if th.size else 0.0
    mad = 1.4826 * float(np.median(np.abs(th - med))) if th.size else 0.0
    bright = core & (tophat > med + FOCI_Z * max(mad, 1.0)) & (gray > ring_mean)
    labelled, n = ndi.label(bright)
    sizes = ndi.sum(bright, labelled, index=range(1, n + 1)) if n else np.array([])
    n_foci = int(np.sum((sizes >= FOCI_MIN_AREA) & (sizes <= FOCI_MAX_AREA)))

    # Margin: mean gradient magnitude on the boundary, relative to the spread of the surroundings.
    smooth = ndi.gaussian_filter(gray, 1.0)
    grad = np.hypot(ndi.sobel(smooth, axis=0), ndi.sobel(smooth, axis=1)) / 8.0
    boundary = mask & ~ndi.binary_erosion(mask)
    spread = math.sqrt(0.5 * (np.var(inside) + ring_std ** 2)) if ring.any() else float(np.std(inside))

    return {
        PREFIX + "TallerThanWideRatio": float(height / width),
        PREFIX + "EchogenicityRatio": float(np.mean(inside) / ring_mean) if ring_mean > 0 else np.nan,
        PREFIX + "PunctateFociDensity": 1000.0 * n_foci / area,
        PREFIX + "Solidity": float(min(1.0, area / _convex_hull_area(mask))),
        PREFIX + "MarginSharpness": float(np.mean(grad[boundary]) / spread) if spread > 0 else np.nan,
        PREFIX + "AnechoicFraction": float(np.mean(inside < ANECHOIC_FRACTION_OF_RING * ring_mean))
        if ring_mean > 0 else np.nan,
    }
