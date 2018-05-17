import chainer
import chainer.links as L
import chainer.functions as F


class ResBlock3D_BottleNeck(chainer.Chain):
    """ResBlock w/ bottleneck and w/o residual convolution
    TODO:
    in_channel != out_channelの時のベスト・プラクティスがわからないため，今のところin_channel == out_channelとする
    in_size != out_sizeの時のベスト・プラクティスがわからないため，今の所，in_size == out_sizeとする
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = self.out_channel // 4
        with self.init_scope():
            self.bn1 = L.BatchNormalization(self.in_channel)
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            self.conv1 = L.ConvolutionND(3, self.in_channel, self.hidden_channel, (1, 1, 1), 1, 0)
            self.bn2 = L.BatchNormalization(self.hidden_channel)
            self.conv2 = L.ConvolutionND(3, self.hidden_channel, self.hidden_channel, (3, 3, 3), 1, 1)
            self.bn3 = L.BatchNormalization(self.hidden_channel)
            self.conv3 = L.ConvolutionND(3, self.hidden_channel, self.out_channel, (1, 1, 1), 1, 0)

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        return x + h


class ResBlock3D(chainer.Chain):
    """ResBlock w/ bottleneck and w/o residual convolution
    TODO:
    in_channel != out_channelの時のベスト・プラクティスがわからないため，今のところin_channel == out_channelとする
    in_size != out_sizeの時のベスト・プラクティスがわからないため，今の所，in_size == out_sizeとする
    """
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        with self.init_scope():
            self.bn1 = L.BatchNormalization(self.in_channel)
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            self.conv1 = L.ConvolutionND(3, self.in_channel, self.out_channel, (1, 1, 1), 1, 0)
            self.bn2 = L.BatchNormalization(self.out_channel)
            self.conv2 = L.ConvolutionND(3, self.out_channel, self.out_channel, (3, 3, 3), 1, 1)

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        return x + h