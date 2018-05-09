import chainer
import chainer.links as L
import chainer.functions as F


class ResBlock3D_Identity_BN(chainer.Chain):
    """
     assert B_in, C_in, W_in, H_in, D_in == B_out, C_out, W_out, H_out, D_out
     assert C % 2 ** 3 == 0
    """
    def __init__(self, channel):
        super().__init__()
        self.in_channel = channel
        self.out_channel = channel
        self.hidden_channel = channel // 2 ** 3
        with self.init_scope():
            self.bn1 = L.BatchNormalization(self.in_channel)
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            self.conv1 = L.ConvolutionND(3, self.in_channel, self.hidden_channel, (1, 1, 1), 1, 0)
            self.bn2 = L.BatchNormalization(self.hidden_channel)
            self.conv2 = L.ConvolutionND(3, self.hidden_channel, self.hidden_channel, (3, 3, 3), 1)
            self.bn3 = L.BatchNormalization(self.hidden_channel)
            self.conv3 = L.ConvolutionND(3, self.hidden_channel, self.out_channel, (1, 1, 1), 0)

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        return x + h
