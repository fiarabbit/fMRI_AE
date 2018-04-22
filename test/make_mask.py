import numpy as np
import nibabel as nib


mask_path = "/data/mask/average_optthr.nii"
_d = nib.load(mask_path).get_data()

# _d には，0と1.00000006がたくさん入ってる（死ぬほどクソ）

d = np.logical_not(_d == 0).astype(np.float32)


d_cropped = d[9:81, 11:99, 0:80]
mask_save = "/hoge/npy/mask/average_optthr.npy"
np.save(mask_save, d_cropped)