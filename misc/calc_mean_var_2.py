import numpy as np
import nibabel as nib

from os.path import join

root_path = "C:\\Users\\hashimoto\\PycharmProjects\\fMRI_AE"


mask_path = join(root_path, "view\\example\\data\\average_optthr.nii")
mask: np.ndarray = nib.load(mask_path).get_data()
mask_idx = list(mask.nonzero())

n_nonzero = len(mask_idx[0])
n_frames = 150
for i in range(3):
    mask_idx[i] = np.tile(mask_idx[i], n_frames)
mask_idx.append(np.repeat(np.arange(n_frames), n_nonzero))

subject_ids = [6, 48, 63, 64, 82, 100, 101, 108, 120, 140]
for s_i, s_id in enumerate(subject_ids):
    data_path = join(root_path, "view\\reconstruct\\niftiDATA_Subject{:03d}_Condition000.nii".format(s_id))
    data = nib.load(data_path).get_data()
    data_masked = data[mask_idx]
    print(data_masked.mean(), data_masked.std())