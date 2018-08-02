import numpy as np
import nibabel as nib
from numpy import load
from numpy.lib.format import open_memmap
from matplotlib import pyplot as plt

from .visualize_fmri import plot_4d_data
from .mask_loader import load_mask_nib
from .util import colorize_image

import itertools
from os import path

from matplotlib.cm import hot

import pdb

def chain(*iterables):
    return tuple(itertools.chain.from_iterable(iterables))


def get_absolute_path(relative_path):
    return path.join(path.dirname(__file__), relative_path).replace(path.sep, "/")


def plot_SimpleFCAE():
    crop = [[9, 81], [11, 99], [0, 80]]
    mask_path = get_absolute_path("example/data/average_optthr.nii")
    # mask_path = get_absolute_path("example/data/binary_mask4grey_BerlinMargulies26subjects.nii")
    mask = load_mask_nib(mask_path, crop)

    base_path = get_absolute_path("example/data/average_COBRE148subjects.nii")
    base = nib.load(base_path).get_data()[[slice(*crop[i]) for i in range(3)]]

    zs = [6, 16, 26, 36, 46, 56, 66, 76]
    ts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    grad_stack = []
    for t, subject in itertools.product(ts, subjects):
        grad_path = get_absolute_path("grad/sensitivity_map_feature_2_7_4_subject{:03d}_frame{:03d}.npy".format(subject, t))
        grad = np.load(grad_path)
        grad = grad * mask
        grad_stack.append(grad)
    grad = np.stack(grad_stack)

    sig_max, sig_min = grad.max(), grad.min()
    assert sig_max > 0 and sig_min < 0
    # sig_thr = max(sig_max, -sig_min)
    sig_thr = 0.0007
    print(sig_thr)
    fig, axes = plt.subplots(len(zs), len(ts) * len(subjects))
    fig.subplots_adjust(left=0.01, right=1 - 0.01, bottom=0.01, top=1 - 0.01, wspace=0.01, hspace=0.01)
    fig, ax = plt.subplots(1, 1)
    for s_i, s in enumerate(subjects):
        for z_i, z in enumerate(zs):
            for t_i, t in enumerate(ts):
                g = grad[s_i * len(ts) + t_i, :, :, z]
                b = base[:, :, z]
                # ax = axes[z_i, s_i * len(ts) + t_i]
                # ax = axes[t_i]
                ax.set_axis_off()
                ax.imshow(b, cmap="gray")
                ax.imshow(colorize_image(g, x_min=-sig_thr/4, x_max=sig_thr/4, threshold=0.5, opacity_absolute=True))
                # plt.show()
                plt.savefig(get_absolute_path("pictures/grad/sensitivity_map_feature_2_7_4_subject{:03d}_frame{:03d}_z{:03d}.eps".format(s, t, z)))
                ax.cla()
    # plt.show()
