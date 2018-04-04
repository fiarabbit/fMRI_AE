import pickle
import numpy as np
import nibabel as nib
from os import listdir, path
from itertools import product
import re


species = ["valid"]
for spec in species:
    directory = path.join("/data", spec)
    out = path.join("/serialize/npy", spec)

    files_nib = sorted(listdir(directory))
    frames = 150
    subject_ids = [re.match("^niftiDATA_Subject(?P<subject>[0-9]{3})_Condition000.nii", file).group("subject") for file in files_nib]
    crop = [[9, 81], [11, 99], [0, 80]]
    slice = [slice(*crop[i]) for i in range(3)]

    class NibDescripterLoader:
        def __init__(self):
            self.descriptors  = []
            for file in files_nib:
                self.descriptors.append(nib.load(path.join(directory, file)))

        def __call__(self, s:int, f:int):
            d = self.descriptors[s]
            r = d.get_data()[slice + [f]]
            return np.asarray(r, dtype=np.float32)

    d = NibDescripterLoader()


    def npyize(s:int, f:int):
        print(path.join(out, "subject{}_frame{:03d}.npy".format(subject_ids[s], f)))
        np.save(path.join(out, "subject{}_frame{:03d}.npy".format(subject_ids[s], f)), d(s, f))

    for s, f in product(range(len(files_nib)), range(150)):
        npyize(s, f)
