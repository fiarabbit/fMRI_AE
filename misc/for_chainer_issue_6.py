import chainer
from chainer import functions as F
import numpy, cupy

xp = cupy
dtype = numpy.float32

n = 2
c_in = 3
c_out = 2
size_in = (4, 3)
ksize = (3, 3)
x_shape = (n, c_in) + size_in
W_shape = (c_out, c_in) + ksize
b_shape = (c_out,)
dilate = 2
pad = ((ksize[0] // 2) * dilate, (ksize[1] // 2) * dilate)

x = chainer.Variable(xp.random.randn(*x_shape).astype(dtype))
W = chainer.Variable(xp.random.randn(*W_shape).astype(dtype))

y = F.convolution_nd(x, W, dilate=dilate, pad=pad)
y.grad = xp.ones(y.shape, dtype=y.dtype)

with chainer.using_config('use_cudnn', 'always'), \
     chainer.using_config('cudnn_deterministic', True):
    y.backward()