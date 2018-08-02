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

import pdb


def chain(*iterables):
    return tuple(itertools.chain.from_iterable(iterables))


def get_absolute_path(relative_path):
    return path.join(path.dirname(__file__), relative_path).replace(path.sep, "/")


def plot_SimpleFCAE():
    crop = [[9, 81], [11, 99], [0, 80]]
    crop_4d = crop + [[None, None]]
    mask_path = get_absolute_path("example/data/average_optthr.nii")
    mask = load_mask_nib(mask_path, crop)

    base_path = get_absolute_path("example/data/average_COBRE148subjects.nii")
    base = nib.load(base_path).get_data()[[slice(*crop[i]) for i in range(3)]]

    ts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    subject_names = [6, 48, 63, 64, 82, 100, 101, 108, 120, 140]
    zs = [6, 16, 26, 36, 46, 56, 66, 76]

    # ts = [0, 10]
    # subjects = [1, 2, 3]
    # subject_names = [48, 63, 64]
    # zs = [46]

    sig_thr = 0.5  # mean std == 0.5

    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    for s_i, s_name in enumerate(subject_names):
        before_path = get_absolute_path(
            "reconstruct/niftiDATA_Subject{:03d}_Condition000.nii".format(s_name))
        before_all = nib.load(before_path).get_data()
        before_cropped = before_all[[slice(*crop_4d[i]) for i in range(4)]]
        for t_i, t in enumerate(ts):
            before = before_cropped[:, :, :, t] * mask
            after_path = get_absolute_path("reconstruct/reconstruction_subject{:03d}_frame{:03d}.npy".format(subjects[s_i], t))
            after = np.load(after_path) * mask
            for z_i, z in enumerate(zs):
                # fig, ax = plt.subplots(1, 1)
                # ax.set_axis_off()
                for i in range(3):
                    ax.imshow(base[:, :, z], cmap="gray")
                    bef = before[:, :, z]
                    aft = after[:, :, z]
                    dif = bef - aft
                    if i == 0:
                        im = colorize_image(bef, x_min=-sig_thr, x_max=sig_thr, threshold=0.1, opacity_absolute=True)
                        name = "before_subject{:03d}_frame{:03d}_z{:03d}.eps".format(s_i, t, z)
                    elif i == 1:
                        im = colorize_image(aft, x_min=-sig_thr, x_max=sig_thr, threshold=0.1, opacity_absolute=True)
                        name = "after_subject{:03d}_frame{:03d}_z{:03d}.eps".format(s_i, t, z)
                    elif i == 2:
                        im = colorize_image(dif, x_min=-sig_thr, x_max=sig_thr, threshold=0.1, opacity_absolute=True)
                        name = "diff_subject{:03d}_frame{:03d}_z{:03d}.eps".format(s_i, t, z)
                    ax.imshow(im)
                    plt.savefig(get_absolute_path(path.join("pictures", "reconstruct", name)))
                    # plt.show()