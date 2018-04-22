from chainer.dataset import DatasetMixin
from os import listdir, path
import nibabel as nib
import numpy as np
import chainer
import sys


class NibDataset(DatasetMixin):
    frames = 150
    def __init__(self, directory: str, crop: list):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.slice = [slice(*crop[i]) for i in range(3)]
        self.descriptors = []
        for i, file in enumerate(self.files):
            sys.stdout.write('\r\033[K' + "{}/{} loaded".format(i, len(self.files)))
            sys.stdout.flush()
            self.descriptors.append(nib.load(path.join(directory, file)))

    def __len__(self):
        return len(self.files) * self.frames

    def get_example(self, i):
        # print(i)
        return chainer.cuda.get_array_module().asarray(self.descriptors[i // self.frames].get_data()[self.slice + [i % self.frames]])


class NpyCroppedDataset(DatasetMixin):
    frames = 150
    def __init__(self, directory: str, crop):
        self.files = sorted(listdir(directory))
        self.memmaps = []
        self.len = len(self.files)
        for i, filename in enumerate(self.files):
            if i % self.frames == 0:
                sys.stdout.write('\r\033[K' + "{}/{} loaded".format(i // 150, self.len // 150))
                sys.stdout.flush()
            self.memmaps.append(np.load(path.join(directory, filename), mmap_mode="r"))

    def __len__(self):
        return self.len

    def get_example(self, i):
        return self.memmaps[i]