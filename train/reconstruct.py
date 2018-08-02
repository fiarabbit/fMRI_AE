import chainer
from chainer import Variable
from chainer import functions as F
from chainer.backends.cuda import get_device_from_id, to_gpu, to_cpu
from chainer.serializers.npz import load_npz
from chainer.iterators import SerialIterator
from chainer.dataset.convert import concat_examples

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import nibabel as nib

from train.model import SimpleFCAE_E32D32 as Model
from train.dataset import NibDataset
from train.mask_loader import load_mask_nib

import sys


def main(id):
    with chainer.using_config("train", False):
        with chainer.using_config("enable_backprop", False):
            model_path = "/efs/fMRI_AE/SimpleFCAE_E32D32/model/model_iter_108858"

            gpu = 0
            get_device_from_id(gpu).use()

            """NibDataset
            def __init__(self, directory: str, crop: list):
            """
            crop = [[9, 81], [11, 99], [0, 80]]
            test_dataset = NibDataset("/data/test", crop=crop)
            """
            def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
            """
            mask = load_mask_nib("/data/mask/average_optthr.nii", crop)

            model = Model(mask, 2, "mask", "mask")
            load_npz(model_path, model)
            model.to_gpu()

            for i in range(len(test_dataset)):
                if i % 8 != id:
                    continue
                inp = to_gpu(test_dataset.get_example(i))
                inp = xp.expand_dims(inp, 0)
                subject = test_dataset.get_subject(i)
                frame = test_dataset.get_frame(i)
                sys.stdout.write("\rsubject{:03d} frame{:03d}".format(subject, frame))
                sys.stdout.flush()
                out = model.reconstruct(inp).array
                out = xp.squeeze(out)
                xp.save("/efs/fMRI_AE/SimpleFCAE_E32D32/reconstruct/reconstruction_subject{:03d}_frame{:03d}.npy".format(subject, frame), out)


