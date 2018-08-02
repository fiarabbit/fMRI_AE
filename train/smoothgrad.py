import chainer
from chainer import Variable
from chainer import functions as F
from chainer.backends.cuda import get_device_from_id, to_gpu, to_cpu
from chainer.serializers.npz import load_npz

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import nibabel as nib

from train.model import SimpleFCAE_E32D32 as Model
from train.dataset import NibDataset
from train.mask_loader import load_mask_nib

import itertools
import pdb


def chain(*iterables):
    return tuple(itertools.chain.from_iterable(iterables))


def main(id):
    model_path = "/efs/fMRI_AE/SimpleFCAE_E32D32/model/model_iter_108858"

    gpu = 0
    get_device_from_id(gpu).use()

    """NibDataset
    def __init__(self, directory: str, crop: list):
    """
    crop = [[9, 81], [11, 99], [0, 80]]
    test_dataset = NibDataset("/data/test", crop=crop)

    mask = load_mask_nib("/data/mask/average_optthr.nii", crop)
    """SimpleFCAE_E32D32
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
    """
    model = Model(mask, 2, "mask", "mask")
    load_npz(model_path, model)
    model.to_gpu()

    # feature_idx = 0
    # feature_idx = (0, 4, 5, 5) # == [0, 9/2, 11/2, 10/2]
    # feature_idx = (0, 1, 1, 1)
    feature_idx = (0, 2, 7, 4)
    resample_size = 100
    batch_size = 10
    noise_level = 0.2

    for i in range(len(test_dataset)):
        if i % 8 != id:
            continue
        print("{:4}/{:4}".format(i, len(test_dataset)))
        subject = test_dataset.get_subject(i)
        frame = test_dataset.get_frame(i)
        test_img = xp.asarray(test_dataset[i])

        resample_remain = resample_size
        resample_processed = 0
        ret = xp.zeros(test_img.shape)
        while resample_remain > 0:
            batch_size_this_loop = min(batch_size, resample_remain)
            resample_remain -= batch_size_this_loop

            batch = xp.broadcast_to(test_img, chain((batch_size_this_loop,), test_img.shape))
            sigma = noise_level / (xp.max(test_img) - xp.min(test_img))
            batch += sigma * xp.random.randn(*batch.shape)

            x = Variable(batch)

            feature = model.extract(x)
            assert feature.shape == (batch_size, 1, 9, 11, 10)
            feature = F.sum(feature, axis=0)
            assert feature.shape == (1, 9, 11, 10)
            feature = F.get_item(feature,  feature_idx)
            feature.backward()
            grad = xp.mean(x.grad, axis=0)
            ret = (ret * resample_processed + grad * batch_size_this_loop) / (resample_processed + batch_size_this_loop)
            model.cleargrads()

        xp.save("/efs/fMRI_AE/SimpleFCAE_E32D32/grad/sensitivity_map_feature_{}_{}_{}_subject{:03d}_frame{:03d}".format(feature_idx[1], feature_idx[2], feature_idx[3], subject, frame), ret)
