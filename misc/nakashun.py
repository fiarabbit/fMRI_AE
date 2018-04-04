import chainer

from chainer import functions as F
from chainer import links as L

class PixelShuffler(chainer.Chain):
    def __init__(self, r):
        super(PixelShuffler, self).__init__()
        self._r = r

    def __call__(self, x):
        batchsize, in_channel, height, width = x.shape
        out_channel = int(in_channel / (self._r * self._r))
        h = F.reshape(x, (batchsize, self._r, self._r, out_channel, height, width))
        h = h.transpose((0, 3, 4, 1, 5, 2))
        return F.reshape(h, (batchsize, out_channel, self._r * height, self._r * width))


class UpScale(chainer.Chain):
    def __init__(self, r):
        self._r = r
        super(UpScale, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=64, out_channels=256, ksize=3, stride=1, pad=1)
            self.pixel_shuffler = PixelShuffler(self._r)

    def __call__(self, x):
        return F.leaky_relu(self.pixel_shuffler(self.conv1(x)))