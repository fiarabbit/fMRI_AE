import nibabel as nib
import numpy as np

mask_path = "C:/Users/hashimoto/PycharmProjects/fMRI_AE/view/example/data/average_optthr.nii"
mask: np.ndarray = nib.load(mask_path).get_data()
mask_idx = mask.nonzero()


def mean_and_var(nib_path):
    d = nib.load(nib_path).get_data()

    for t in range()