import chainer
from chainer import functions as F


class PixelShuffler3D(chainer.Chain):
    def __init__(self, r):
        super().__init__()
        self._r = r
        self._persistent.add('_r')

    def __call__(self, x: chainer.Variable):
        batchsize, in_channel, height, width, depth = x.shape
        out_channel = in_channel // self._r ** 3
        h = F.reshape(x, (batchsize, self._r, self._r, self._r, out_channel, height, width, depth))
        h = h.transpose((0, 4, 5, 1, 6, 2, 7, 3))
        return F.reshape(h, (batchsize, out_channel, self._r * height, self._r * width, self._r * depth))
