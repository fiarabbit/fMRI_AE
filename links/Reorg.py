import chainer
from chainer import functions as F


class Reorg(chainer.Chain):
    def __init__(self, r):
        super().__init__()
        self._r = r
        self._persistent.add('_r')

    def __call__(self, x: chainer.Variable):
        batchsize, channel, in_height, in_width, in_depth = x.shape
        out_height = in_height // self._r
        out_width = in_width // self._r
        out_depth = in_depth // self._r
        h = F.reshape(x, (batchsize, channel, self._r, out_height, self._r, out_width, self._r, out_depth)) # -> b, c, x, r, y, r, z, r
        h = h.transpose((0, 3, 5, 7, 1, 2, 4, 6)) # -> b, r, r, r, c, x, y, z
        return F.reshape(h, (batchsize, channel * self._r ** 3, out_height, out_width, out_depth))
