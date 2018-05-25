import chainer
from chainer.backends.cuda import to_cpu
from chainer.serializers import load_npz
from chainer.iterators import SerialIterator
from chainer.dataset import concat_examples

import numpy as np

from train.model import SimpleFCAE_E8D8
from train.mask_loader import load_mask_nib
from train.dataset import NibDataset

mask = load_mask_nib("/data/mask/average_optthr.nii", crop=[[9, 81], [11, 99], [0, 80]])
model = SimpleFCAE_E8D8(mask, 2, "mask", "mask")
model.to_gpu(0)
dataset = NibDataset("/data/test", crop=[[9, 81], [11, 99], [0, 80]])
iterator = SerialIterator(dataset, batch_size=1, repeat=False, shuffle=False)
loss_stack = []

for _ in iterator:
    batch = concat_examples(_, device=0)
    loss = model(batch)
    loss_stack.append(to_cpu(loss))

print(np.mean(loss_stack))
