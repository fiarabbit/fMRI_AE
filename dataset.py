from chainer.dataset import DatasetMixin
from os import listdir, path
import nibabel as nib
import numpy as np
import chainer


class NibDataset(DatasetMixin):
    frames = 150

    def __init__(self, directory: str, crop: list):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.slice = [slice(*crop[i]) for i in range(3)]
        self.descriptors = []
        for file in self.files:
            self.descriptors.append(nib.load(path.join(directory, file)))

    def __len__(self):
        return len(self.files) * self.frames

    def get_example(self, i):
        # print(i)
        return chainer.cuda.get_array_module().asarray(self.descriptors[i // self.frames].get_data()[self.slice + [i % self.frames]])


class NpyCroppedDataset(DatasetMixin):
    frames = 150
    def __init__(self, directory: str):
        self.files = sorted(listdir(directory))
        self.memmaps = []
        self.len = len(self.files) * self.frames
        for s in range(len(self.files)):
            self.memmaps.append([])
            for f in range(self.frames):
                self.memmaps[s].append(np.load(path.join(directory, "subject{}_frame{:03d}.npy".format(self.files[s], f)), "r"))

    def __len__(self):
        return self.len

    def get_example(self, i):
        return self.memmaps[i // self.frames][i % self.frames]