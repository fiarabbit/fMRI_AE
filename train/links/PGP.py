import chainer
from chainer import functions as F


class PGP3D(chainer.Chain):
    def __init__(self, r):
        super().__init__()
        self._r = r
        self._persistent.add('_r')

    def __call__(self, x: chainer.Variable):
        batchsize, channel, in_height, in_width, in_depth = x.shape
        out_height, out_width, out_depth = [x // self._r for x in (in_height, in_width, in_depth)]
        h = F.reshape(x, (batchsize, channel, out_height, self._r, out_width, self._r, out_depth, self._r))  # -> b, c, v, x, r, y, r, z, r
        h = h.transpose((3, 5, 7, 0, 1, 2, 4, 6))  # -> r, r, r, b, c, x, y, z
        return h.reshape((batchsize * self._r ** 3, channel, out_height, out_width, out_depth))  # -> b, c, x, y, z
