import numpy as np
import nibabel as nib
import warnings


def load_mask_nib(mask_path: str, crop: list):
    mask = np.asarray(nib.load(mask_path).get_data(), dtype=np.float32)
    if mask.shape[0] != crop[0][1] - crop[0][0] or mask.shape[1] != crop[1][1] - crop[1][0] or mask.shape[2] != crop[2][1] - crop[2][0]:
        slice_ = [slice(*crop[i]) for i in range(3)]
        mask = mask[slice_]
    try:
        assert (1 == mask[mask.nonzero()]).all()
    except AssertionError:
        warnings.warn("Non-bool mask")
        print("converting to boolean...")
        mask[mask.nonzero()] = 1
    return mask


def load_mask_npy(mask_path: str, crop: list):
    mask = np.load(mask_path)
    if mask.shape[0] != crop[0][1] - crop[0][0] or mask.shape[1] != crop[1][1] - crop[1][0] or mask.shape[2] != crop[2][1] - crop[2][0]:
        slice_ = [slice(*crop[i]) for i in range(3)]
        mask = mask[slice_]
    try:
        assert mask.dtype == np.float32
        assert (1 == mask[mask.nonzero()]).all()
    except AssertionError:
        warnings.warn("Non-bool mask")
        print("converting to boolean...")
        mask[mask.nonzero()] = 1
    return mask
