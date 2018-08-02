from chainer import Chain, Parameter
from chainer.functions import convolution_nd, deconvolution_nd, transpose, linear, reshape, broadcast_to
from chainer.initializers import _get_initializer
import chainer.links.connection.convolution_2d

from chainer.functions.connection.linear import LinearFunction


class PCA(Chain):
    """
    extract: (batch, in_channel, H, W, D) -> (batch, hidden_channel)
    reverse: (batch, hidden_channel) -> (batch, in_channel * H * W * D)
    """
    def __init__(self, hidden_channel, initialW=None, initial_bias=None):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.initial_bias = initial_bias
        self.initialW = initialW

    def __call__(self, x):
        original_shape = x.shape
        if not hasattr(self, "W"):
            with self.init_scope():
                self.W = Parameter(_get_initializer(self.initialW), (self.hidden_channel, int(self.xp.prod(self.xp.array(original_shape[1:])))))
                self.W.to_gpu()
        if not hasattr(self, "bias"):
            self.bias = Parameter(_get_initializer(self.initial_bias), (int(self.xp.prod(self.xp.array(original_shape[1:]))),))
            self.bias.to_gpu()

        if x.ndim > 2:
            x = x.reshape(len(x), -1)

        h = x - broadcast_to(self.bias, x.shape)
        h = linear(h, self.W)
        h = linear(h, self.W.T)
        h = h + broadcast_to(self.bias, h.shape)
        h = reshape(h, original_shape)
        return h

    def to_gpu(self, device=None):
        if hasattr(self, "W"):
            self.W.to_gpu(device)
        if hasattr(self, "bias"):
            self.bias.to_gpu(device)


class PCA_across_channel(Chain):
    """
    extract: in_channel -> out_channel with F.convolution_nd()
    reverse: out_channel -> in_channel with F.deconvolution_nd()
    """
    def __init__(self, in_channels, hidden_channel, initialW=None):
        super().__init__()
        ksize = (1, 1, 1)
        self.W_shape = (hidden_channel, in_channels) + ksize
        self.stride = 1
        self.pad = 0
        self.nobias = True
        self.b = None
        self.initialW = initialW
        with self.init_scope():
            self.W = Parameter(_get_initializer(self.initialW), self.W_shape)

    def __call__(self, x):
        """Applies N-dimensional convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of convolution.

        """
        return deconvolution_nd(
            convolution_nd(
                x,
                self.W, self.b, self.stride, self.pad
            ),
            self.W, self.b, self.stride, self.pad
        )
