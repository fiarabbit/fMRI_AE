from chainer.dataset import DatasetMixin
from os import listdir, path
import nibabel as nib
import numpy as np
import chainer
import sys

class NibNoiseDataset190(DatasetMixin):
    frames = 190

    def __init__(self, directory: str, crop: list, noise: float=0.1):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.slice = [slice(*crop[i]) for i in range(3)]
        self.noise = noise
        self.descriptors = []
        for i, file in enumerate(self.files):
            sys.stdout.write('\r\033[K' + "{:3d}/{:3d} loaded".format(i+1, len(self.files)))
            sys.stdout.flush()
            self.descriptors.append(nib.load(path.join(directory, file)))
        print("\nfinish")

    def __len__(self):
        return len(self.files) * self.frames

    def get_example(self, i):
        # print(i)
        xp = chainer.cuda.get_array_module()
        sample = xp.asarray(self.descriptors[i // self.frames].get_data()[self.slice + [i % self.frames]])
        sample_noise = sample + self.noise * xp.random.randn(*sample.shape)
        return sample_noise, sample

    def get_frame(self, i):
        return i % self.frames

    def get_subject(self, i):
        return i // self.frames

    def get_example_from_subject_frame(self, subject, frame):
        return chainer.cuda.get_array_module().asarray(self.descriptors[subject].get_data()[self.slice + [frame]])


class NibDataset190(DatasetMixin):
    frames = 190

    def __init__(self, directory: str, crop: list):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.slice = [slice(*crop[i]) for i in range(3)]
        self.descriptors = []
        for i, file in enumerate(self.files):
            sys.stdout.write('\r\033[K' + "{:3d}/{:3d} loaded".format(i+1, len(self.files)))
            sys.stdout.flush()
            self.descriptors.append(nib.load(path.join(directory, file)))
        print("\nfinish")

    def __len__(self):
        return len(self.files) * self.frames

    def get_example(self, i):
        # print(i)
        return chainer.cuda.get_array_module().asarray(self.descriptors[i // self.frames].get_data()[self.slice + [i % self.frames]])

    def get_frame(self, i):
        return i % self.frames

    def get_subject(self, i):
        return i // self.frames

    def get_example_from_subject_frame(self, subject, frame):
        return chainer.cuda.get_array_module().asarray(self.descriptors[subject].get_data()[self.slice + [frame]])

class NibDataset(DatasetMixin):
    frames = 150

    def __init__(self, directory: str, crop: list):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.slice = [slice(*crop[i]) for i in range(3)]
        self.descriptors = []
        for i, file in enumerate(self.files):
            sys.stdout.write('\r\033[K' + "{:3d}/{:3d} loaded".format(i+1, len(self.files)))
            sys.stdout.flush()
            self.descriptors.append(nib.load(path.join(directory, file)))
        print("\nfinish")

    def __len__(self):
        return len(self.files) * self.frames

    def get_example(self, i):
        # print(i)
        return chainer.cuda.get_array_module().asarray(self.descriptors[i // self.frames].get_data()[self.slice + [i % self.frames]])

    def get_frame(self, i):
        return i % self.frames

    def get_subject(self, i):
        return i // self.frames

    def get_example_from_subject_frame(self, subject, frame):
        return chainer.cuda.get_array_module().asarray(self.descriptors[subject].get_data()[self.slice + [frame]])

class NibNoiseDataset(DatasetMixin):
    frames = 150

    def __init__(self, directory: str, crop: list, noise: float=0.1):
        self.directory = directory
        self.files = sorted(listdir(self.directory))
        self.slice = [slice(*crop[i]) for i in range(3)]
        self.noise = noise
        self.descriptors = []
        for i, file in enumerate(self.files):
            sys.stdout.write('\r\033[K' + "{:3d}/{:3d} loaded".format(i+1, len(self.files)))
            sys.stdout.flush()
            self.descriptors.append(nib.load(path.join(directory, file)))
        print("\nfinish")

    def __len__(self):
        return len(self.files) * self.frames

    def get_example(self, i):
        # print(i)
        xp = chainer.cuda.get_array_module()
        sample = xp.asarray(self.descriptors[i // self.frames].get_data()[self.slice + [i % self.frames]])
        sample_noise = sample + self.noise * xp.random.randn(*sample.shape)
        return sample_noise, sample

    def get_frame(self, i):
        return i % self.frames

    def get_subject(self, i):
        return i // self.frames

    def get_example_from_subject_frame(self, subject, frame):
        return chainer.cuda.get_array_module().asarray(self.descriptors[subject].get_data()[self.slice + [frame]])

class SexDataset(DatasetMixin):
    def __init__(self, directory: str):
        pass

class NpyCroppedDataset(DatasetMixin):
    frames = 150

    def __init__(self, directory: str, crop):
        self.files = sorted(listdir(directory))
        self.memmaps = []
        self.len = len(self.files)
        for i, filename in enumerate(self.files):
            if i % self.frames == 0:
                sys.stdout.write('\r\033[K' + "{:3d}/{:3d} loaded".format(i+1 // 150, self.len // 150))
                sys.stdout.flush()
            self.memmaps.append(np.load(path.join(directory, filename), mmap_mode="r"))
        print("\nfinish")

    def __len__(self):
        return self.len

    def get_example(self, i):
        return self.memmaps[i]