"""
Link.params()
Link.disable_update()
の効能をテストしています．
"""

import chainer
from chainer import Chain, Variable, Parameter
from chainer import links as L
from chainer import functions as F


class Model(Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 10)
            self.conv2 = L.Convolution2D(10, 1)
            self.const = Parameter(self.xp.array(2))

    def __call__(self, x):
        h = self.conv1(x)
        h = h * self.const
        h = self.conv2(h)
        return F.mean_squared_error(x, h)

model = Model()
for i in model.params():
    print(i)
