import numpy as np
import nibabel as nib
import warnings
import itertools
from scipy.stats import pearsonr


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


def get_f_path(feature_idx, t, s):
    return "/efs/fMRI_AE/SimpleFCAE_E32D32/grad/sensitivity_map_feature_{}_{}_{}_subject{:03d}_frame{:03d}.npy".format(feature_idx[0], feature_idx[1], feature_idx[2], s, t)


def main():
    feature_idx = [5, 2, 3]
    subjects = tuple(range(10))
    ts = tuple(range(150))
    indexer = tuple(itertools.product(ts, subjects))

    crop = [[9, 81], [11, 99], [0, 80]]
    mask = load_mask_nib("/data/mask/average_optthr.nii", crop)
    data_list = []
    for i, ts_1 in enumerate(indexer):
        t_1, s_1 = ts_1
        for j, ts_2 in enumerate(indexer):
            if i > j:
                continue
            t_2, s_2 = ts_2
            f_1_path = get_f_path(feature_idx, t_1, s_1)
            f_1 = np.load(f_1_path)[mask.nonzero()]
            f_2_path = get_f_path(feature_idx, t_2, s_2)
            f_2 = np.load(f_2_path)[mask.nonzero()]
            r, _ = pearsonr(f_1, f_2)
            data_list.append(r)
    np.save("pearsonlist.npy", np.array(data_list))

if __name__ == "__main__":
    main()