import numpy as np
from matplotlib import colors, cm

from typing import Optional
import warnings


def colorize_image(x: np.ndarray, x_min: Optional[float] = None, x_max: Optional[float] = None, threshold: Optional[float] = None, cmap=None, opacity_absolute: bool=False) -> np.ndarray:
    """create image suitable for plt.imshow()

    This function takes a np.ndarray of the raw 2D data, and returns a 4-channel np.ndarray which can be plotted by plt.imshow()

    Args:
        x (np.ndarray): 2-dimensional image. x.shape == (h, w)
        x_min (float, optional): upper plotting threshold of x
        x_max (float, optional): lower plotting threshold of x
        threshold (float): a float value ∈ [0, 1], which defines the range of transparency of non 1. default is 0.5.
        cmap: Reds, Purples, Greens ..., imported by `from matplotlib.cm import Reds, Purples, Greens, Oranges`. default is hot.
        opacity_absolute (bool): if True, values near 0 will be transparent. if False, smaller value will be transparent.

    Returns:
        np.ndarray: 4-channel (RGBα) data, which can be plotted by plt.imshow()

    """
    assert x.ndim == 2
    if x_min is None:
        x_min = x.min()
        warnings.warn("x_min was automatically set to {:.3e}".format(x_min))

    if x_max is None:
        x_max = x.max()
        warnings.warn("x_max was automatically set to {:.3e}".format(x_max))

    if threshold is None:
        threshold = 0.5

    if cmap is None:
        if opacity_absolute:
            cmap = cm.RdBu
        else:
            cmap = cm.hot

    rgba = cmap(colors.Normalize(x_min, x_max, clip=True)(x))
    alpha = np.copy(x)
    if opacity_absolute:
        x_mean = (x_min + x_max) / 2
        alpha = (alpha - x_mean) / (x_max - x_mean)
        alpha = np.abs(alpha)
    else:
        alpha = (alpha - x_min) / (x_max - x_min)

    alpha = alpha / threshold
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    rgba[..., -1] = alpha
    return rgba