import chainer
from chainer.backends.cuda import to_cpu
from chainer.serializers import load_npz
from chainer.iterators import SerialIterator
from chainer.dataset import concat_examples

from chainer.utils.conv import im2col_gpu

import numpy as np

# from train.model2 import StackedFCAE as Model
from train.model import SimpleFCAE_E16D16_wo_BottleNeck_Denoise as Model
# from train.model import SimpleFCAE_E16D16_wo_BottleNeck as Model
from train.mask_loader import load_mask_nib
from train.dataset import NibDataset, NibNoiseDataset, NibDataset190, NibNoiseDataset190

import pdb

def main():
    crop = [[9, 81], [7, 103], [0, 80]]
    # crop = [[9, 81], [11, 99], [0, 80]]
    # crop = [[5, 85], [7, 103], [0, 80]]
    # mask = load_mask_nib("/data/mask/average_optthr.nii", crop=crop)
    mask = load_mask_nib("/data_berlin/mask/binary_mask4grey_BerlinMargulies26subjects.nii", crop=crop)
    model = Model(mask, r=2, in_mask="mask", out_mask="mask")
    load_npz("/efs/fMRI_AE/SimpleFCAE_E16D16_wo_BottleNeck_Denoise_Berlin_noise_1/model/model_iter_11000", model)
    model.to_gpu(0)
    # dataset = NibNoiseDataset190("/data_berlin/timeseries", crop=crop, noise=0)
    # dataset = NibDataset("/data_cobre/test", crop=crop)
    dataset = NibNoiseDataset("/data_cobre/test", crop=crop, noise=1)
    batch_size = 15
    assert(len(dataset) % batch_size == 0)
    iterator = SerialIterator(dataset, batch_size=batch_size, repeat=False, shuffle=False)
    loss_stack = []

    count = 0
    for _ in iterator:
        print("{}/{}".format(count, len(dataset)//batch_size))
        # pdb.set_trace()
        batch = concat_examples(_, device=0)
        loss = model(*batch)
        loss_stack.append(np.mean(to_cpu(loss.array)))
        count += 1
        print(np.mean(loss_stack))


