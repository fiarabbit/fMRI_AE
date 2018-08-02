"""model_multigpu.py
BatchNormalization対応版
"""

import numpy as np
import chainer
from chainer import Variable, Parameter, Chain
from chainer.backends.cuda import to_gpu
from chainer.reporter import report
from chainer import functions as F
from chainer import links as L
from chainermn import links as L_multi

from typing import Sequence
from warnings import warn
import itertools

from .links.ResNet_multigpu import ResBlockMN
import pdb


# noinspection PyDeprecation
class StackedResBlockMN(Chain):
    @staticmethod
    def get_shapes(mask_shape: tuple, num_layers: int):
        ret = []
        shape = mask_shape
        for i in range(num_layers):
            ret.append(tuple(shape))
            shape = [(i + 1) // 2 for i in shape]
        return tuple(ret)

    @staticmethod
    def chain(*iterables):
        return tuple(itertools.chain.from_iterable(iterables))

    def __init__(self, comm, mask,
                 feature_dim,
                 encoder_channels: Sequence[int], encoder_layers: Sequence[int],
                 decoder_channels: Sequence[int], decoder_layers: Sequence[int]):
        super().__init__()

        assert len(encoder_layers) == len(decoder_layers)
        assert len(encoder_channels) == len(encoder_layers)
        assert len(decoder_channels) == len(decoder_layers)

        self.comm = comm
        self.mask = mask
        self.feature_dim = feature_dim
        self.encoder_channels = encoder_channels
        self.encoder_layers = encoder_layers
        self.decoder_channels = decoder_channels
        self.decoder_layers = decoder_layers
        self.shapes = self.get_shapes(mask.shape, len(encoder_layers))
        self.loss_const = self.mask.size / self.mask.sum()

        with self.init_scope():
            # Encoding Block
            for layer_idx, num_layers in enumerate(encoder_layers):
                for rep_idx in range(num_layers):
                    in_channel = 2 if layer_idx == 0 and rep_idx == 0 else encoder_channels[layer_idx]

                    if rep_idx == num_layers - 1 and layer_idx != len(encoder_layers) - 1:  # 各layerの出力層，ただしfeatureを出力する層は除く
                        out_channel, stride = encoder_channels[layer_idx + 1], 2
                    else:
                        out_channel, stride = encoder_channels[layer_idx], 1
                    """
                     def __init__(self, comm, in_channel, out_channel, stride=1):
                    """
                    self.add_link("conv_{}_{}".format(layer_idx, rep_idx), ResBlockMN(self.comm, in_channel, out_channel, stride))

            # Extract Block
            """
            def __init__(self, in_size, out_size=None, nobias=False, initialW=None, initial_bias=None):
            """
            self.linear_extract = L.Linear(np.prod(self.chain((self.encoder_channels[-1], ), self.shapes[-1])), self.feature_dim)
            self.linear_reconstruct = L.Linear(self.feature_dim, np.prod(self.chain((self.decoder_channels[-1],), self.shapes[-1])))

            # Decoding Block
            for layer_idx, num_layers in reversed(tuple(enumerate(decoder_layers))):
                for rep_idx in range(num_layers):
                    if rep_idx == 0 and layer_idx != len(decoder_layers) - 1:  # 各layerの入力層，ただしfeatureを受け取る層は除く
                        in_channel = decoder_channels[layer_idx + 1]
                    else:
                        in_channel = decoder_channels[layer_idx]

                    out_channel = 1 if layer_idx == 0 and rep_idx == num_layers - 1 else decoder_channels[layer_idx]

                    stride = 1
                    self.add_link("dcnv_{}_{}".format(layer_idx, rep_idx), ResBlockMN(self.comm, in_channel, out_channel, stride))

    def to_gpu(self, device=None):
        super(StackedResBlockMN, self).to_gpu()
        self.mask = chainer.backends.cuda.to_gpu(self.mask)
        self.loss_const = chainer.backends.cuda.to_gpu(self.loss_const)

    def to_cpu(self):
        super(StackedResBlockMN, self).to_cpu()
        self.mask = chainer.backends.cuda.to_cpu(self.mask)
        self.loss_const = chainer.backends.cuda.to_cpu(self.loss_const)

    def extract(self, x):
        batch_size = x.shape[0]
        x = F.reshape(x, self.chain((batch_size, 1), self.shapes[0]))
        mask = F.broadcast_to(self.mask, x.shape)
        h = F.hstack((x, mask))
        for layer_idx, num_layers in enumerate(self.encoder_layers):
            for rep_idx in range(num_layers):
                conv = self.__getattribute__("conv_{}_{}".format(layer_idx, rep_idx))
                h = conv(h)
        h = self.linear_extract(h)
        return h

    def reconstruct(self, f):
        batch_size = f.shape[0]
        h = self.linear_reconstruct(f)
        h = F.reshape(h, self.chain((batch_size, self.decoder_channels[-1]), self.shapes[-1]))
        for layer_idx, num_layers in reversed(tuple(enumerate(self.decoder_layers))):
            for rep_idx in range(num_layers):
                dcnv = self.__getattribute__(("dcnv_{}_{}".format(layer_idx, rep_idx)))
                h = dcnv(h)
            if layer_idx != 0:
                h = F.unpooling_nd(h, 1, 2, outsize=self.shapes[layer_idx - 1], cover_all=False)
        h = F.reshape(h, self.chain((batch_size,), self.shapes[0]))
        return h

    def __call__(self, x):
        f = self.extract(x)
        x_recon = self.reconstruct(f)
        loss = F.mean_absolute_error(F.scale(x, self.mask), F.scale(x_recon, self.mask)) * self.loss_const
        report({'loss': loss}, self)
        return loss


# noinspection PyDeprecation
class StackedResBlockConvStartMN(Chain):
    """
    上のモデルだと，途中にたくさん計算グラフを保存しなければならないため，バッチ数が少なくなりすぎる傾向にある（8GPUでバッチ16しか乗らない）
    そこで，Encoderの第一レイヤーをただのconvolutionに置き換える(ResNetと同じ工夫)
    Decoderの出力レイヤーを書き換えたいのだけれど，Decoderがめちゃくちゃやっている都合もあって，Decoderで良い案が思いつかない
    てかそもそもDecoderとかいう物体，Unpooling は stride 2 kernel 2 の Deconvolutionでいいんだな．愚か
    いや，そもそも

    ほとんどが第一レイヤーに集中しているので第一レイヤーを削除できればいけるか？
     91 * 109 * 91 = 902629 *  32 = 28884128 =  29M * 4bytes
     46 *  55 * 46 = 116380 *  32 =  3724160 = 3.7M * 4bytes
     23 *  28 * 23 =  14812 *  64 =   947968 = 0.9M * 4bytes
     12 *  14 * 12 =   2016 *  64 =   129024 = 0.1M * 4bytes
      6 *   7 *  6 =    252 * 128 =    32256 =  32K * 4bytes
      3 *   4 *  3 =     36 * 128 =     4608 =   4K * 4bytes
      2 *   2 *  2 =      8 * 256 =     2048 =   2K * 4bytes
    """
    @staticmethod
    def get_shapes(mask_shape: tuple, num_layers: int):
        ret = []
        shape = mask_shape
        for i in range(num_layers):
            ret.append(tuple(shape))
            shape = [(i + 1) // 2 for i in shape]
        return tuple(ret)

    @staticmethod
    def chain(*iterables):
        return tuple(itertools.chain.from_iterable(iterables))

    def __init__(self, comm, mask,
                 feature_dim,
                 encoder_channels: Sequence[int], encoder_layers: Sequence[int],
                 decoder_channels: Sequence[int], decoder_layers: Sequence[int]):
        super().__init__()

        assert len(encoder_layers) == len(decoder_layers)
        assert len(encoder_channels) == len(encoder_layers)
        assert encoder_layers[0] == 1
        assert len(decoder_channels) == len(decoder_layers)

        self.comm = comm
        self.mask = mask
        self.feature_dim = feature_dim
        self.encoder_channels = encoder_channels
        self.encoder_layers = encoder_layers
        self.decoder_channels = decoder_channels
        self.decoder_layers = decoder_layers
        self.shapes = self.get_shapes(mask.shape, len(encoder_layers))
        self.loss_const = self.mask.size / self.mask.sum()

        with self.init_scope():
            # Encoding Block
            for layer_idx, num_layers in enumerate(encoder_layers):
                for rep_idx in range(num_layers):
                    in_channel = 2 if layer_idx == 0 and rep_idx == 0 else encoder_channels[layer_idx]
                    if rep_idx == num_layers - 1 and layer_idx != len(encoder_layers) - 1:  # 各layerの出力層，ただしfeatureを出力する層は除く
                        out_channel, stride = encoder_channels[layer_idx + 1], 2
                    else:
                        out_channel, stride = encoder_channels[layer_idx], 1
                    """
                     def __init__(self, comm, in_channel, out_channel, stride=1):
                    """
                    if layer_idx != 0:
                        self.add_link("conv_{}_{}".format(layer_idx, rep_idx), ResBlockMN(self.comm, in_channel, out_channel, stride))
                    else:
                        """
                        def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0, nobias=False, initialW=None, initial_bias=None, cover_all=False):
                        """
                        self.add_link("conv_{}_{}".format(layer_idx, rep_idx), L.ConvolutionND(3, in_channel, out_channel, 5, stride, 2))

            # Extract Block
            """
            def __init__(self, in_size, out_size=None, nobias=False, initialW=None, initial_bias=None):
            """
            self.linear_extract = L.Linear(np.prod(self.chain((self.encoder_channels[-1], ), self.shapes[-1])), self.feature_dim)
            self.linear_reconstruct = L.Linear(self.feature_dim, np.prod(self.chain((self.decoder_channels[-1],), self.shapes[-1])))

            # Decoding Block
            for layer_idx, num_layers in reversed(tuple(enumerate(decoder_layers))):
                for rep_idx in range(num_layers):
                    if rep_idx == 0 and layer_idx != len(decoder_layers) - 1:  # 各layerの入力層，ただしfeatureを受け取る層は除く
                        in_channel = decoder_channels[layer_idx + 1]
                    else:
                        in_channel = decoder_channels[layer_idx]

                    out_channel = 1 if layer_idx == 0 and rep_idx == num_layers - 1 else decoder_channels[layer_idx]

                    stride = 1
                    self.add_link("dcnv_{}_{}".format(layer_idx, rep_idx), ResBlockMN(self.comm, in_channel, out_channel, stride))

    def to_gpu(self, device=None):
        super(StackedResBlockConvStartMN, self).to_gpu()
        self.mask = chainer.backends.cuda.to_gpu(self.mask)
        self.loss_const = chainer.backends.cuda.to_gpu(self.loss_const)

    def to_cpu(self):
        super(StackedResBlockConvStartMN, self).to_cpu()
        self.mask = chainer.backends.cuda.to_cpu(self.mask)
        self.loss_const = chainer.backends.cuda.to_cpu(self.loss_const)

    def extract(self, x):
        batch_size = x.shape[0]
        x = F.reshape(x, self.chain((batch_size, 1), self.shapes[0]))
        mask = F.broadcast_to(self.mask, x.shape)
        h = F.hstack((x, mask))
        for layer_idx, num_layers in enumerate(self.encoder_layers):
            for rep_idx in range(num_layers):
                conv = self.__getattribute__("conv_{}_{}".format(layer_idx, rep_idx))
                h = conv(h)
        h = self.linear_extract(h)
        return h

    def reconstruct(self, f):
        batch_size = f.shape[0]
        h = self.linear_reconstruct(f)
        h = F.reshape(h, self.chain((batch_size, self.decoder_channels[-1]), self.shapes[-1]))
        for layer_idx, num_layers in reversed(tuple(enumerate(self.decoder_layers))):
            for rep_idx in range(num_layers):
                dcnv = self.__getattribute__(("dcnv_{}_{}".format(layer_idx, rep_idx)))
                h = dcnv(h)
            if layer_idx != 0:
                h = F.unpooling_nd(h, 1, 2, outsize=self.shapes[layer_idx - 1], cover_all=False)
        h = F.reshape(h, self.chain((batch_size,), self.shapes[0]))
        return h

    def __call__(self, x):
        f = self.extract(x)
        x_recon = self.reconstruct(f)
        loss = F.mean_absolute_error(F.scale(x, self.mask), F.scale(x_recon, self.mask)) * self.loss_const
        report({'loss': loss}, self)
        return loss