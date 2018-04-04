from time import time
from os import listdir, path

import numpy as np
import nibabel as nib
import pickle

directory_nib = "/data/train"
directory_pickle = "/serialize/pickle"
directory_npy = "/serialize/npy"
files_nib = sorted(listdir(directory_nib))
files_pickle = sorted(listdir(directory_pickle))
files_npy = sorted(listdir(directory_npy))
frames = 150
subjects = 127
crop = [[9, 81], [11, 99], [0, 80]]
slice = [slice(*crop[i]) for i in range(3)]

time()


def serial_load(i: int):
    return np.asarray(nib.load(path.join(directory_nib, files_nib[i // frames])).get_data()[slice + [i % frames]])


class PickleDescriptorLoader:
    def __init__(self):
        self.descriptors = []
        for s in range(subjects):
            self.descriptors.append([])
            for f in range(frames):
                self.descriptors[s].append(open(path.join(directory_pickle, "subject{:03d}_frame{:03d}.pickle".format(s, f)), "rb"))
        print("end init")

    def __call__(self, i: int):
        s = i // frames
        f = i % frames
        ret = pickle.load(self.descriptors[s][f])
        self.descriptors[s][f].seek(0)
        return ret


class NpyMemmapLoader:
    def __init__(self):
        self.memmap = []
        for s in range(subjects):
            self.memmap.append([])
            for f in range(frames):
                self.memmap[s].append(np.load(path.join(directory_npy, "subject{:03d}_frame{:03d}.npy".format(s, f)), "r"))
        print("end init")

    def __call__(self, i: int):
        s = i // frames
        f = i % frames
        return self.memmap[s][f]


class NpyMemmapLoader2(NpyMemmapLoader):
    def __call__(self, i: int):
        s = i // frames
        f = i % frames
        return np.asarray(self.memmap[s][f])


class NibDescriptorLoader:
    def __init__(self):
        self.descriptors = []
        for file in (files_nib):
            self.descriptors.append(nib.load(path.join(directory_nib, file)))

    def __call__(self, i: int):
        return self.descriptors[i // frames].get_data()[slice + [i % frames]]


class NibDescriptorLoader2(NibDescriptorLoader):
    def __call__(self, i: int):
        return np.asarray(self.descriptors[i // frames].get_data()[slice + [i % frames]])


def iterate(loader):
    start_time = time()
    order = np.random.permutation(np.arange(frames * subjects))
    for i in order:
        d = loader(i)
    print(time() - start_time)


# loader = NibDescriptorLoader() # 0.087
# loader = NibDescriptorLoader2() # 0.119
loader = PickleDescriptorLoader()  # inf
# loader = NpyMemmapLoader() # 0.0172
# loader = NpyMemmapLoader2() # 0.0430
# loader = serial_load # 12.388

iterate(loader)

# 答え：288倍高速になったよ
