import chainer
from chainer import Variable, Parameter
from chainer import functions as F
from chainer import links as L

import numpy as np

from os import remove
from os.path import exists


class NG_Mixed(chainer.Chain):
    def __init__(self, v: Variable, f: chainer.Function, p: Parameter, l: chainer.Link):
        super().__init__()
        with self.init_scope():
            self.v = v
            self.f = f
            self.p = p
            self.l = l


def test_mixed():
    try:
        instance_1 = NG_Mixed(Variable(np.array([0])), F.ReLU(), Parameter(np.array([0])), L.BatchNormalization(1))
        chainer.serializers.save_npz("tmp.npz", instance_1)
        instance_2 = NG_Mixed(Variable(np.array([1])), F.ReLU(), Parameter(np.array([1])), L.BatchNormalization(1))
        chainer.serializers.load_npz("tmp.npz", instance_2)
        assert (instance_1.p.data == instance_2.p.data).all()
        assert (instance_1.v.data == instance_2.v.data).all()
        print("Succeeded")
    except Exception as e:
        print("Failed")
        raise e
    finally:
        if exists("tmp.npz"):
            remove("tmp.npz")
