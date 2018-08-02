import chainer
from chainer.backends.cuda import to_cpu
from chainer.serializers import load_npz
from chainer.iterators import SerialIterator
from chainer.dataset import concat_examples

import numpy as np

# from train.model2 import StackedFCAE as Model
from train.model2 import StackedFCAE as Model
from train.mask_loader import load_mask_nib
from train.dataset import NibDataset


def main():
    experiment_name = "Stacked_16_16_16_16"
    crop = [[10, 80], [11, 99], [3, 77]]
    mask = load_mask_nib("/data/mask/average_optthr.nii", crop=crop)
    model = Model(mask, channels=[16,16,16,16], n_conv_per_block=2, pca_dim=1, init_n_blocks=4, with_reconstruction_loss=True, with_pca=False, with_pca_loss=False, debug=False)
    load_npz("/efs/fMRI_AE/{}/model/model_iter_27215".format(experiment_name), model)
    pickle_data = np.load("/efs/fMRI_AE/{}_feature/ipca.pickle".format(experiment_name))
    model.pca.W = chainer.Parameter(pickle_data.components_)
    model.pca.bias = chainer.Parameter(pickle_data.mean_)
    model.pca.disable_update()
    model.attach_pca()
    model.detach_pca_loss()
    model.attach_reconstruction_loss()
    model.disable_update()
    model.to_gpu(0)

    dataset = NibDataset("/data/test", crop=crop)
    iterator = SerialIterator(dataset, batch_size=14, repeat=False, shuffle=False)
    loss_stack = []

    with chainer.using_config("train", False):
        count = 0
        for _ in iterator:
            print(count)
            batch = concat_examples(_, device=0)
            loss = model(batch)
            print(loss)
            loss_stack.append(np.mean(to_cpu(loss.array)))
            count += 1

    print(np.mean(loss_stack))
