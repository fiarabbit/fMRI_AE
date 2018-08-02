import sys

import nibabel as nib
import numpy as np
from numpy import ndarray

mask_path = "/data/mask/average_optthr.nii"
mask = nib.load(mask_path).get_data()
assert isinstance(mask, ndarray)
mask_idx = mask.nonzero()

m = []
for i, subject_idx in enumerate([6, 48, 63, 64, 82, 100, 101, 108, 120, 140]):
    # subject_idx = 6
    d = nib.load("/data/test/niftiDATA_Subject{:03d}_Condition000.nii".format(subject_idx)).get_data()
    assert isinstance(d, ndarray)
    for f in range(150):
        # f = 0
        sys.stdout.write("\rsubject {:2d}, frame {:3d}".format(i, f))
        sys.stdout.flush()
        s = d[:, :, :, f]
        t = s[mask_idx]
        m.append(t.mean())
m = np.mean(m)
v = []
for i, subject_idx in enumerate([6, 48, 63, 64, 82, 100, 101, 108, 120, 140]):
    d = nib.load("/data/test/niftiDATA_Subject{:03d}_Condition000.nii".format(subject_idx)).get_data()
    d -= m
    assert isinstance(d, ndarray)
    for f in range(150):
        # f = 0
        sys.stdout.write("\rsubject {:2d}, frame {:3d}".format(i, f))
        sys.stdout.flush()
        s = d[:, :, :, f]
        t = s[mask_idx]
        # v.append(t.var())
        v.append(np.mean(np.abs(t)))

# var: 0.17089622
# std: 0.41339597
# abs: 0.2837288