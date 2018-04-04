import chainer
from chainer import Variable, Parameter
from chainer import functions as F
from chainer import links as L

import numpy as np

from os import remove
from os.path import exists


class NG_None(chainer.Chain):
    def __init__(self):
        super().__init__()


class NG_Variable(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.v = Variable(np.array([1, 2, 3]))


class NG_Function(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.f = F.ReLU()


class OK_Parameter(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.p = Parameter(np.array([1, 2, 3]))


class OK_Link(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l = L.BatchNormalization(1)


def save_and_load(Class):
    try:
        instance_1 = Class()
        chainer.serializers.save_npz("tmp.npz", instance_1)
        chainer.serializers.load_npz("tmp.npz", instance_1)
        print("Succeeded")
    except Exception as e:
        print("Failed")
        raise e
    finally:
        if exists("tmp.npz"):
            remove("tmp.npz")


