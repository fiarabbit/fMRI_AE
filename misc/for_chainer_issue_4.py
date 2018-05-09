import chainer
import chainer.links as L
import chainer.functions as F

from chainer.optimizers import Adam
from chainer.iterators import SerialIterator
from chainer.training import Trainer, ParallelUpdater

import numpy as np


class Model(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(5, 5)

    def __call__(self, x):
        s = list(x.shape)
        s.pop()
        loss = F.mean_absolute_error(self.l1(x), x.reshape(s))
        return loss


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 1024

    def get_example(self, i):
        return np.arange(5, dtype=np.float32).reshape((5, 1))

model = Model()
optimizer = Adam()
optimizer.setup(model)

train_dataset = Dataset()
train_iterator = SerialIterator(train_dataset, 2)

updater = ParallelUpdater(train_iterator, optimizer, devices={"main": 0, "second": 1})
trainer = Trainer(updater)
trainer.run()
