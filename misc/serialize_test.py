import chainer
from chainer import Chain, Variable
from chainer import functions as F
from chainer.dataset import DatasetMixin
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training import Trainer
from chainer.training.updater import StandardUpdater
from chainer.training.extensions import snapshot_object
from chainer.serializers import load_npz

import numpy as np


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__()
        self.mask = Variable(np.zeros((2, 2), dtype=np.float32))
        self._persistent.add("mask")

    def __call__(self, x):
        return F.sum(x)

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)


class Dataset(DatasetMixin):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 1024

    def get_example(self, i):
        return np.array([[1, 2], [3, 4]], dtype=np.float32)


dataset = Dataset()
iterator = SerialIterator(dataset, 2, False, False)
model_training = Model()
model_training.to_gpu()
optimizer = Adam()
optimizer.setup(model_training)
updater = StandardUpdater(iterator, optimizer, device=0)
trainer = Trainer(updater, stop_trigger=[1, "iteration"])
trainer.extend(snapshot_object(model_training, "model_iter_{.updater.iteration}"), trigger=[1, "iteration"])
trainer.run()
model_test = Model()
load_npz("result/model_iter_1", model_test)
model_test.to_gpu()