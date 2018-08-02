import chainer
from chainer import Variable, Parameter, Chain
from chainer import links as L
from chainer import functions as F

from chainermn import links as Lmn


class ResBlockMN(Chain):
    """
    x -> bn -> relu -> conv -> bn -> relu -> conv -> y
      |------ residual_conv -> residual_bn ------|
    """
    def __init__(self, comm, in_channel, out_channel, stride=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.comm = comm
        with self.init_scope():
            """
                def __init__(self, size, comm, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
             """
            self.bn1 = Lmn.MultiNodeBatchNormalization(in_channel, comm)
            """
                def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False):
            """
            # facebook model = stride second
            self.conv1 = L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, True)
            self.bn2 = Lmn.MultiNodeBatchNormalization(out_channel, comm)
            self.conv2 = L.ConvolutionND(3, out_channel, out_channel, (3, 3, 3), stride, 1, True)
            if self.stride != 1 or self.in_channel != self.out_channel:
                self.residual_conv = L.ConvolutionND(3, in_channel, out_channel, (1, 1, 1), stride, 0, True)
                self.residual_bn = Lmn.MultiNodeBatchNormalization(out_channel, comm)

    def __call__(self, x):
        h = F.relu(self.bn1(x))
        if self.stride != 1 or self.in_channel != self.out_channel:
            residual = self.residual_conv(h)
        else:
            residual = x
        h = self.conv1(h)
        # print(h.shape)
        h = self.bn2(self.conv2(h))
        # print(h.shape)
        h += residual
        return h


