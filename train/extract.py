import chainer
from chainer.backends.cuda import to_cpu
from chainer.serializers import load_npz
from chainer.iterators import SerialIterator
from chainer.dataset import concat_examples

import numpy as np

from train.model2 import StackedFCAE as Model
from train.mask_loader import load_mask_nib
from train.dataset import NibDataset

from os import makedirs


def main():
    experiment_name = "Stacked_16_16_16_16"
    model_name = "model_iter_27215" # 20 epoch -> pca is not attached yet
    makedirs("/out/{}/feature".format(experiment_name), exist_ok=True)

    batch_size = 15

    crop = [[10, 80], [11, 99], [3, 77]]
    mask = load_mask_nib("/data/mask/average_optthr.nii", crop=crop)
    model = Model(mask, channels=[16, 16, 16, 16], n_conv_per_block=2, pca_dim=990, init_n_blocks=4, with_reconstruction_loss=True, with_pca=True, with_pca_loss=False, debug=False)
    load_npz("/efs/fMRI_AE/{}/model/{}".format(experiment_name, model_name), model)
    model.to_gpu(0)
    dataset = NibDataset("/data/train", crop=crop)
    iterator = SerialIterator(dataset, batch_size=batch_size, repeat=False, shuffle=False)
    with chainer.using_config("train", False):
        for subject in range(len(dataset) // 150):
            print("{}/{}".format(subject, len(dataset)//150))
            feature_stack = []
            for batch_idx in range(150 // batch_size):
                batch = concat_examples(next(iterator), device=0)
                feature = model.extract(batch)
                feature_stack.append(feature.array)
            feature_subject = to_cpu(model.xp.vstack(feature_stack))
            print(feature_subject.shape)
            np.save("/out/{}/feature/subject{}.npy".format(experiment_name, subject), feature_subject)
