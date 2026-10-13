"""Ring of tissue around the nodule, for peritumoral radiomics.

PyRadiomics features computed inside this ring describe the tissue next to the nodule (its
texture and brightness), which the intratumoral features and the TI-RADS-motivated features do
not cover. Fixed settings, chosen before looking at labels:

    width        RING_FRACTION of the nodule's equivalent radius, at least MIN_WIDTH pixels
    excluded     the nodule itself, and pixels darker than OUTSIDE_SECTOR on the image as stored
                 (the black area outside the ultrasound sector)

Feature names get the prefix 'peritumoral_' in place of PyRadiomics' 'original_'.
"""

import math

import numpy as np
from scipy import ndimage as ndi

from .clinical import OUTSIDE_SECTOR

PREFIX = "peritumoral_"
RING_FRACTION = 0.5
MIN_WIDTH = 5
FAMILIES = ["firstorder", "glcm", "glszm", "glrlm", "gldm", "ngtdm"]   # no shape: the ring's shape is set by the mask


def ring_width(mask: np.ndarray, fraction: float = RING_FRACTION, min_width: int = MIN_WIDTH) -> int:
    radius = math.sqrt(float(np.asarray(mask, dtype=bool).sum()) / math.pi)
    return max(min_width, round(fraction * radius))


def ring_mask(gray: np.ndarray, mask: np.ndarray, fraction: float = RING_FRACTION,
              min_width: int = MIN_WIDTH, outside: float = OUTSIDE_SECTOR) -> np.ndarray:
    """Boolean ring around `mask`, inside the image and inside the ultrasound sector."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("empty mask")
    width = ring_width(mask, fraction, min_width)
    # Euclidean distance to the nodule: the same ring as a dilation with a disk of radius `width`,
    # without the memory and time a large structuring element needs on full-size images.
    near = ndi.distance_transform_edt(~mask) <= width
    return near & ~mask & (np.asarray(gray) >= outside)


def crop_to(ring: np.ndarray, margin: int = 2) -> tuple[slice, slice]:
    """Bounding box of the ring plus a small margin, to pass PyRadiomics a small image."""
    rows, cols = np.nonzero(ring)
    return (slice(max(rows.min() - margin, 0), rows.max() + margin + 1),
            slice(max(cols.min() - margin, 0), cols.max() + margin + 1))


def rename(pyradiomics_key: str) -> str:
    """'original_glcm_Idm' -> 'peritumoral_glcm_Idm'."""
    return PREFIX + pyradiomics_key.split("_", 1)[1]
