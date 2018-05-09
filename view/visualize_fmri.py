from typing import Sequence
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

from os import path

from .util import colorize_image


def main():
    data_relative_path = "example/data"
    mask_path = path.join(path.dirname(__file__), data_relative_path, "average_optthr.nii")
    mask = np.asarray(nib.load(mask_path).get_data(), dtype=np.float32)
    mask[mask.nonzero()] = 1
    data_path = path.join(path.dirname(__file__), data_relative_path, "niftiDATA_Subject001_Condition000.nii")
    data = np.asarray(nib.load(data_path).get_data(), dtype=np.float32)
    base_path = path.join(path.dirname(__file__), data_relative_path, "average_COBRE148subjects.nii")
    base = np.asarray(nib.load(base_path).get_data(), dtype=np.float32)
    plot_4d_data(data, mask, base, [0, 25,  50, 75,  100], [6, 26, 46, 66])


def plot_4d_data(data: np.ndarray, mask: np.ndarray, base: np.ndarray, ts: Sequence[int], zs: Sequence[int])->None:
    assert data.ndim == 4
    fig, axes = plt.subplots(len(zs), len(ts))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray(axes,)
    for z_i, z in enumerate(zs):
        for t_i, t in enumerate(ts):
            ax = axes[z_i, t_i]
            ax.set_axis_off()
            ax.imshow(base[:, :, z], cmap="gray")
            ax.imshow(colorize_image(data[:, :, z, t] * mask[:, :, z], -0.5, 0.5, opacity_absolute=True))
    plt.show()

