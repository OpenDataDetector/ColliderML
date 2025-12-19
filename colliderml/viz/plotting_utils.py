"""Small plotting/geometry helpers used by notebooks.

Keep notebook cells short by centralizing non-essential plotting glue here.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def r_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.sqrt(x * x + y * y)


def eta_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pseudorapidity from position vector (x,y,z)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    r = r_from_xy(x, y)
    R = np.sqrt(r * r + z * z)
    return 0.5 * np.log((R + z + eps) / (R - z + eps))


def eta_from_pxpypz(px: np.ndarray, py: np.ndarray, pz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pseudorapidity from momentum vector (px,py,pz)."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    pz = np.asarray(pz, dtype=float)
    p = np.sqrt(px * px + py * py + pz * pz)
    return 0.5 * np.log((p + pz + eps) / (p - pz + eps))


def plot_binned_sums_with_xerr(ax, x: np.ndarray, w: np.ndarray, *, bins: np.ndarray, label: str, **kwargs):
    """Compute weighted sums in bins of x and plot with xerr = bin width / 2."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    bins = np.asarray(bins, dtype=float)
    y, _ = np.histogram(x, bins=bins, weights=w)
    centers = 0.5 * (bins[:-1] + bins[1:])
    xerr = 0.5 * (bins[1] - bins[0])
    ax.errorbar(centers, y, xerr=xerr, label=label, **kwargs)
    return centers, y


def scatter_eta_logr(ax, eta: np.ndarray, r: np.ndarray, c: Optional[np.ndarray] = None, *, s=10, alpha=0.5, **kwargs):
    logr = np.log(np.asarray(r, dtype=float) + 1e-12)
    return ax.scatter(eta, logr, c=c, s=s, alpha=alpha, **kwargs)


def scatter_xy(ax, x: np.ndarray, y: np.ndarray, c: Optional[np.ndarray] = None, *, s=10, alpha=0.5, **kwargs):
    return ax.scatter(x, y, c=c, s=s, alpha=alpha, **kwargs)


