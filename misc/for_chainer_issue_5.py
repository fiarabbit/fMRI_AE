import chainer
import chainer.functions as F

from chainer.optimizers import Adam
from chainer.iterators import SerialIterator
from chainer.training import Trainer, ParallelUpdater
from chainer.backends import cuda

import numpy as np


class Model(chainer.Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        h = F.elu(x)
        return F.mean(h)


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1024

    def get_example(self, i):
        return np.arange(2, dtype=np.float32)


model = Model()
optimizer = Adam()
optimizer.setup(model)

train_dataset = Dataset()
train_iterator = SerialIterator(train_dataset, 2)

updater = ParallelUpdater(train_iterator, optimizer, devices={"main": 0, "second": 1})
trainer = Trainer(updater, (1, "iteration"))
trainer.run()