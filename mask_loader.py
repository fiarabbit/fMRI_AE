import numpy as np
import nibabel as nib
import warnings

def load_mask(mask_path: str):
    mask = np.asarray(nib.load(mask_path).get_data(), dtype=np.float32)
    try:
        assert (1 == mask[mask.nonzero()]).all()
    except AssertionError:
        warnings.warn("Non-bool mask")
        print("converting to boolean...")
        mask[mask.nonzero()] = 1
    return mask
