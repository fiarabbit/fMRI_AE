import numpy as np
import chainer
from chainer import links as L
from chainer import functions as F


def conv(in_size: int, kernel: int, stride: int, padding: int):
    batch = 2
    in_channel = 3
    out_channel = 5
    x = np.arange(batch * in_channel * in_size ** 3, dtype=np.float32).reshape((batch, in_channel, in_size, in_size, in_size))
    l = L.ConvolutionND(3, in_channel, out_channel, kernel, stride, padding)
    y = l(x)
    return y.shape


def dcnv(in_size: int, kernel: int, stride: int, padding: int):
    batch = 2
    in_channel = 5
    out_channel = 3
    x = np.arange(batch * in_channel * in_size**3, dtype=np.float32).reshape((batch, in_channel, in_size, in_size, in_size))
    l = L.DeconvolutionND(3, in_channel, out_channel, kernel, stride, padding)
    y = l(x)
    return y.shape


def pooling(in_size: int, kernel: int, stride: int, padding: int):
    batch = 2
    in_channel = 3
    out_channel = 5
    x = np.arange(batch * in_channel * in_size ** 3, dtype=np.float32).reshape((batch, in_channel, in_size, in_size, in_size))
    y = F.average_pooling_nd(x, kernel, stride, padding)
    return y.shape


def unpooling(in_size: int, kernel: int, stride: int, padding: int):
    batch = 2
    in_channel = 5
    out_channel = 3
    x = np.arange(batch * in_channel * in_size ** 3, dtype=np.float32).reshape((batch, in_channel, in_size, in_size, in_size))
    y = F.unpooling_nd(x, kernel, stride, padding)
    return y.shape