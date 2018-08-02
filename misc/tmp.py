from chainer import Variable, Chain
from chainer import links as L
from chainer import functions as F

import numpy as np
from six import print_
import cupy as cp

batch_size = 100
in_channel = 1
out_channel = 1

class MyLink(Chain):
    def __init__(self):
        super(MyLink, self).__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(3, 1, 1, (3, 3, 3), nobias=True, initialW=np.ones((in_channel, out_channel, 3, 3, 3), dtype=np.float32))

    def __call__(self, x):
        return self.conv(x)

if __name__ == "__main__":
    my_link = MyLink()
    my_link.to_gpu(0)
    batch = Variable(np.ones((batch_size, in_channel, 3, 3, 3), dtype=np.float32))
    batch.to_gpu(0)
    out = my_link(batch)
    print_(out.shape)
    out.grad = cp.ones(out.shape, dtype=cp.float32)
    out.backward()

