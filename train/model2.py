"""model2.py
model.pyの文量が増えすぎて，よくわからなくなってきたため，よく使うものだけを綺麗に書き直して保存する
"""

import numpy as np
import chainer
from chainer import Variable, Chain
from chainer.backends.cuda import to_cpu, to_gpu
from chainer.reporter import report
from chainer import functions as F
from chainer import links as L

from train.links.PCA import PCA

from copy import copy
from typing import Sequence
from warnings import warn

# noinspection PyDeprecation
class SimpleFCAE(Chain):
    """
    model.pyのSimpleFCAEを改変したもの．
    Pooling層の前でReLUしないようにしている．
    また，ReLUに伴って情報欠損が生じないように，nobias=Falseとしている箇所が増えている．
    """
    # input size
    # extreme: (70, 88, 74) == crop:[(10, 80), (11, 99), (3, 77)]
    def __init__(self, mask, n_channels, n_downsamples, n_conv_per_downsample, debug=False, init_iteration=0, init_epoch=0):
        super().__init__()
        self.mask = mask
        self.n_channels = n_channels
        self.n_downsamples = n_downsamples
        self.n_conv_per_downsample = n_conv_per_downsample
        self.loss_const = self.mask.size / self.mask.sum()
        self.debug = debug
        self.iteration = init_iteration
        self.epoch = init_epoch

        with self.init_scope():
            # ConvolutionND(dim, in_channel, out_channel, kernel, stride, padding)
            for downsample_degree in range(n_downsamples + 1):
                for conv_idx in range(n_conv_per_downsample):
                    in_channel = 1 if downsample_degree == 0 and conv_idx == 0 else self.n_channels
                    out_channel = 1 if downsample_degree == n_downsamples and conv_idx == n_conv_per_downsample - 1 else self.n_channels
                    nobias = True if conv_idx == n_conv_per_downsample - 1 else False  # ReLUしない時はbiasは不要
                    self.add_link(
                        "conv_{}_{}".format(downsample_degree, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )
            for downsample_degree in reversed(range(n_downsamples + 1)):
                for conv_idx in range(n_conv_per_downsample):
                    in_channel = 1 if downsample_degree == n_downsamples and conv_idx == 0 else self.n_channels
                    out_channel = 1 if downsample_degree == 0 and conv_idx == n_conv_per_downsample - 1 else self.n_channels
                    nobias = True if conv_idx == n_conv_per_downsample - 1 and downsample_degree != 0 else False  # ReLUしない時はbiasは不要だが，最終出力には必要
                    self.add_link(
                        "dcnv_{}_{}".format(downsample_degree, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )

    def to_cpu(self):
        super().to_cpu()
        self.mask = to_cpu(self.mask)
        self.loss_const = to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = to_gpu(self.mask, device)
        self.loss_const = to_gpu(self.loss_const, device)

    def calc(self, x, target):
        """
        :param xp.ndarray x:
        :param xp.ndarray target:
        :return: Variable
        """
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]
        x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        padding_history = []
        shape_history = []

        h = x_masked
        for downsample_degree in range(self.n_downsamples + 1):
            for conv_idx in range(self.n_conv_per_downsample):
                conv = self.__getattribute__("conv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("conv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if conv_idx != self.n_conv_per_downsample - 1:  # poolingや特徴抽出の前はReLUしない
                    if self.debug:
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != self.n_downsamples:
                shape = h.shape[2:]
                shape_history.append(shape)
                padding = tuple([x % 2 for x in shape])
                padding_history.append(padding)
                if self.debug:
                    print("average_pooling_nd")
                h = F.average_pooling_nd(h, 2, 2, padding)  # dimensionの0側にpaddingがかかる
                if self.debug:
                    print("\t{}".format(h.shape))
        # この段階でhがfeature
        for downsample_degree in reversed(range(self.n_downsamples + 1)):
            for conv_idx in range(self.n_conv_per_downsample):
                conv = self.__getattribute__("dcnv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("dcnv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if conv_idx != self.n_conv_per_downsample - 1:  # unpoolingの前はReLUしない
                    if self.debug:
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != 0:
                shape = shape_history.pop()
                padding = padding_history.pop()
                if self.debug:
                    print("unpooling_nd")
                h = F.unpooling_nd(h, 2, 2, padding, shape, cover_all=False)
                if self.debug:
                    print("\t{}".format(h.shape))
        out = F.reshape(h, tuple(original_shape))
        out_masked = F.scale(out, self.mask)

        target_masked = F.scale(target, self.mask)
        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x):
        """
        :param xp.ndarray x:
        :return: Variable
        """
        loss = self.calc(x, x)
        report({'loss': loss}, self)
        return loss


# noinspection PyDeprecation
class RawFCAE(Chain):
    """
    model.pyにあるSimpleFCAEとほとんど同じもの．
    conv_feature層だけ省略している．
    """
    # input size
    # extreme: (70, 88, 74) == crop:[(10, 80), (11, 99), (3, 77)]
    def __init__(self, mask, n_channels, n_downsamples, n_conv_per_downsample, debug=False):
        super().__init__()
        self.mask = mask
        self.n_channels = n_channels
        self.n_downsamples = n_downsamples
        self.n_conv_per_downsample = n_conv_per_downsample
        self.loss_const = self.mask.size / self.mask.sum()
        self.debug = debug

        with self.init_scope():
            # ConvolutionND(dim, in_channel, out_channel, kernel, stride, padding)
            for downsample_degree in range(n_downsamples + 1):
                for conv_idx in range(n_conv_per_downsample):
                    in_channel = 1 if downsample_degree == 0 and conv_idx == 0 else self.n_channels
                    out_channel = 1 if downsample_degree == n_downsamples and conv_idx == n_conv_per_downsample - 1 else self.n_channels
                    nobias = False if downsample_degree == 0 and conv_idx == 0 else True  # 入力層はnobias=False
                    self.add_link(
                        "conv_{}_{}".format(downsample_degree, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )
            for downsample_degree in reversed(range(n_downsamples + 1)):
                for conv_idx in range(n_conv_per_downsample):
                    in_channel = 1 if downsample_degree == n_downsamples and conv_idx == 0 else self.n_channels
                    out_channel = 1 if downsample_degree == 0 and conv_idx == n_conv_per_downsample - 1 else self.n_channels
                    nobias = False if downsample_degree == 0 and conv_idx == n_conv_per_downsample - 1 else True  # 出力層はnobias=False
                    self.add_link(
                        "dcnv_{}_{}".format(downsample_degree, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )

    def to_cpu(self):
        super().to_cpu()
        self.mask = to_cpu(self.mask)
        self.loss_const = to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = to_gpu(self.mask, device)
        self.loss_const = to_gpu(self.loss_const, device)

    def calc(self, x, target):
        """
        :param xp.ndarray x:
        :param xp.ndarray target:
        :return: Variable
        """
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]
        x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        padding_history = []
        shape_history = []

        h = x_masked
        for downsample_degree in range(self.n_downsamples + 1):
            for conv_idx in range(self.n_conv_per_downsample):
                conv = self.__getattribute__("conv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("conv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if not (downsample_degree == self.n_downsamples and conv_idx == self.n_conv_per_downsample - 1):  # 特徴抽出層はReLUしない
                    if self.debug:
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != self.n_downsamples:
                shape = h.shape[2:]
                shape_history.append(shape)
                padding = tuple([x % 2 for x in shape])
                padding_history.append(padding)
                if self.debug:
                    print("average_pooling_nd")
                h = F.average_pooling_nd(h, 2, 2, padding)  # dimensionの0側にpaddingがかかる
        # この段階でhがfeature
        for downsample_degree in reversed(range(self.n_downsamples + 1)):
            for conv_idx in range(self.n_conv_per_downsample):
                conv = self.__getattribute__("dcnv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("dcnv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if not (downsample_degree == 0 and conv_idx == self.n_conv_per_downsample - 1):  # 最終出力層はReLUしない
                    if self.debug:
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != 0:
                shape = shape_history.pop()
                padding = padding_history.pop()
                if self.debug:
                    print("unpooling_nd")
                h = F.unpooling_nd(h, 2, 2, padding, shape, cover_all=False)
        out = F.reshape(h, tuple(original_shape))
        out_masked = F.scale(out, self.mask)

        target_masked = F.scale(target, self.mask)
        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x):
        """
        :param xp.ndarray x:
        :return: Variable
        """
        loss = self.calc(x, x)
        report({'loss': loss}, self)
        return loss


# noinspection PyDeprecation
class StackedFCAE(Chain):
    """
    RawFCAEのn_downsamplesをadaptiveに変化させられるようにしたもの．
    Bias項については全省略．
    input->in_augment->relu->(conv->relu)*n->subsample->(conv->relu)*n->(PCA)->(conv->relu)*n->upsample->(conv->relu)*n->out_augment->output

    downsampleが発生するごとにchannel数を増やす(0: 16, 1: 32, 2: 64, 3: 128)

    :param np.ndarray mask: マスク
    :param List[int] channels: conv_i_0, dcnv_i_0のchannel数．
    :param int n_conv_per_block: length of {conv_0_*}
    :param List[bool] decoder_backprop_enabled: {conv_i_*}, {dcnv_i_*}にbackpropするかどうか
    :param int n_blocks: maximum length of {conv_*_0}
    :param int tmp_n_blocks: current length of {conv_*_0}
    :param bool pca_attached: pcaをするかどうか．
    :param bool pca_loss_attached: pca lossを計算するかどうか
    :param bool reconstruction_loss_attached: reconstruction lossを計算するかどうか

    :meth reset_pca: self.pca is Noneならばself.pcaを設定する．そうでなければ，self.pcaをdelして再設定する
    :meth attach_pca:
    :meth increase_downsample: ダウンサンプルの数を学習可能な状態で増やす

    学習プラン
    1. simple auto encoder & lastly add pca
    init(n_downsamples=max_downsamples, with_pca=False, with_pca_loss=False)
    attach_pca() &  attach_pca_loss() & freeze_convolution() & detach_reconstruction_loss()
    detach_pca_loss() & release_convolution() & attach_reconstruction_loss()

    2. stacked auto encoder
    init(n_downsamples = 1, with_pca = True, with_pca_loss=False)
    .. starting from 1 is because starting from 0 is trivial (identity mapping is the global minimum)
    freeze_convolution(0, 1) & add_downsample() & reset_pca()
    .. n_downsamples = 2
    release_convolution(0, 1)
    freeze_convolution(0, 1, 2) & add_downsample() & reset_pca()
    .. n_downsamples = 3
    release_convolution(0, 1, 2)


    """
    # input size
    # extreme: (70, 88, 74) == crop:[(10, 80), (11, 99), (3, 77)]
    def __init__(self, mask, channels: Sequence[int], n_conv_per_block, pca_dim=1, init_n_blocks=None, with_reconstruction_loss=True, with_pca=False, with_pca_loss=False, debug=False):
        super().__init__()
        try:
            assert len(channels) >= 2
        except AssertionError as e:
            print("len(channels) must be equal or more than 2")
            raise e

        if init_n_blocks is None:
            init_n_blocks = len(channels)

        self.mask = mask
        self.channels = channels
        self.tmp_n_blocks = init_n_blocks
        self.n_blocks = len(self.channels)
        self.n_conv_per_block = n_conv_per_block
        self.loss_const = self.mask.size / self.mask.sum()

        self.decoder_backprop_enabled = [True] * self.n_blocks
        self.encoder_backprop_enabled = [True] * self.n_blocks

        self.pca_attached = with_pca
        self.pca_block_idx = self.tmp_n_blocks - 1

        self.pca_dim = pca_dim

        self.pca_loss_attached = with_pca_loss
        self.reconstruction_loss_attached = with_reconstruction_loss

        self.debug = debug

        with self.init_scope():
            # ConvolutionND(dim, in_channel, out_channel, kernel, stride, padding)
            for block_idx in range(self.n_blocks):
                for conv_idx in range(self.n_conv_per_block):
                    if conv_idx == 0:
                        in_channel = 1 if block_idx == 0 else self.channels[block_idx - 1]
                    else:
                        in_channel = self.channels[block_idx]
                    out_channel = self.channels[block_idx]
                    nobias = False if block_idx == 0 and conv_idx == 0 else True  # 入力層はnobias=False
                    self.add_link(
                        "conv_{}_{}".format(block_idx, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )
            # self.pca = PCA(self.channels[self.pca_block_idx], self.pca_dim)
            self.pca = PCA(self.pca_dim)
            for block_idx in reversed(range(self.n_blocks)):
                for conv_idx in range(self.n_conv_per_block):
                    in_channel = self.channels[block_idx]
                    if conv_idx == self.n_conv_per_block -1:
                        out_channel = 1 if block_idx == 0 else self.channels[block_idx - 1]
                    else:
                        out_channel = self.channels[block_idx]
                    nobias = False if block_idx == 0 and conv_idx == self.n_conv_per_block - 1 else True  # 出力層はnobias=False
                    self.add_link(
                        "dcnv_{}_{}".format(block_idx, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )

    def to_cpu(self):
        super().to_cpu()
        self.mask = to_cpu(self.mask)
        self.loss_const = to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = to_gpu(self.mask, device)
        self.loss_const = to_gpu(self.loss_const, device)

    def extract(self, x):
        assert self.reconstruction_loss_attached or self.pca_loss_attached
        assert self.pca_loss_attached or not self.pca_loss_attached

        original_shape = list(x.shape)
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)
        x_masked = F.reshape(F.scale(x, self.mask), new_shape)
        padding_history = []
        shape_history = []

        h = x_masked
        for downsample_degree in range(self.tmp_n_blocks):
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("conv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if self.debug:  # rawと違って任意のconvレイヤーはReLUしてよい
                    print("relu")
                h = F.relu(h)
            if downsample_degree != self.tmp_n_blocks - 1:
                shape = h.shape[2:]
                shape_history.append(shape)
                padding = tuple([x % 2 for x in shape])
                padding_history.append(padding)
                if self.debug:
                    print("average_pooling_nd")
                h = F.average_pooling_nd(h, 2, 2, padding)  # dimensionの0側にpaddingがかかる
                if self.debug:
                    print("\t{}".format(h.shape))
        return h

    def calc(self, x, target):
        """
        :param xp.ndarray x:
        :param xp.ndarray target:
        :return: Variable
        """
        assert self.reconstruction_loss_attached or self.pca_loss_attached
        assert self.pca_attached or not self.pca_loss_attached

        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]
        x_masked = F.reshape(F.scale(x, self.mask), new_shape)
        padding_history = []
        shape_history = []

        h = x_masked
        for downsample_degree in range(self.tmp_n_blocks):
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("conv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if self.debug:  # rawと違って任意のconvレイヤーはReLUしてよい
                    print("relu")
                h = F.relu(h)
            if downsample_degree != self.tmp_n_blocks - 1:
                shape = h.shape[2:]
                shape_history.append(shape)
                padding = tuple([x % 2 for x in shape])
                padding_history.append(padding)
                if self.debug:
                    print("average_pooling_nd")
                h = F.average_pooling_nd(h, 2, 2, padding)  # dimensionの0側にpaddingがかかる
                if self.debug:
                    print("\t{}".format(h.shape))
        # この段階でhがfeature
        pca_loss = None
        if self.pca_attached:
            if self.debug:
                print("pca")
            feature = self.pca(h)

            if self.pca_loss_attached:
                pca_loss = F.mean_absolute_error(feature, h)
                report({'pca_loss': pca_loss}, self)
                if not self.reconstruction_loss_attached:
                    return pca_loss
            h = feature
            if self.debug:
                print("\t{}".format(h.shape))

        for downsample_degree in reversed(range(self.tmp_n_blocks)):
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("dcnv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("dcnv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if not (downsample_degree == 0 and conv_idx == self.n_conv_per_block - 1):  # 最終出力層はReLUしない
                    if self.debug:
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != 0:
                shape = shape_history.pop()
                padding = padding_history.pop()
                if self.debug:
                    print("unpooling_nd")
                h = F.unpooling_nd(h, 2, 2, padding, shape, cover_all=False)
                if self.debug:
                    print("\t{}".format(h.shape))
        out = F.reshape(h, tuple(original_shape))
        out_masked = F.scale(out, self.mask)

        target_masked = F.scale(target, self.mask)

        reconstruction_loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const
        report({'reconstruction_loss': reconstruction_loss}, self)
        if self.pca_loss_attached:
            return reconstruction_loss + pca_loss
        else:
            return reconstruction_loss

    def __call__(self, x):
        """
        :param xp.ndarray x:
        :return: Variable
        """
        loss = self.calc(x, x)
        report({'loss': loss}, self)
        return loss

    def attach_pca(self, overwrite=False):
        if self.debug:
            print("attach_pca")
        # if self.tmp_n_blocks != self.pca_block_idx + 1:
        #     raise RuntimeError("self.tmp_n_blocks: {} != self.pca_block_idx: {} + 1".format(self.tmp_n_blocks, self.pca_block_idx))
        self.pca_attached = True

    def detach_pca(self, detach_pca_loss=True):
        if self.debug:
            print("detach_pca")
        self.pca_attached = False
        if detach_pca_loss:
            self.pca_loss_attached = False

    def attach_pca_loss(self):
        if self.debug:
            print("attach_pca_loss")
        if not self.pca_attached:
            raise RuntimeError("PCA is not attached")
        self.pca_loss_attached = True

    def detach_pca_loss(self):
        if self.debug:
            print("detach_pca_loss")
        self.pca_loss_attached = False

    def attach_reconstruction_loss(self):
        if self.debug:
            print("attach_reconstruction_loss")
        self.reconstruction_loss_attached = True

    def detach_reconstruction_loss(self):
        if self.debug:
            print("detach_reconstruction_loss")
        self.reconstruction_loss_attached = False

    def freeze_convolution(self, blocks="all"):
        if self.debug:
            print("feeeze_convolution {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = False
            self.decoder_backprop_enabled[block_idx] = False
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.disable_update()
                dcnv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                dcnv.disable_update()

    def release_convolution(self, blocks="all"):
        if self.debug:
            print("release_convolution {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = True
            self.decoder_backprop_enabled[block_idx] = True
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()
                conv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()

    def freeze_decoder(self, blocks="all"):
        if self.debug:
            print("freeze_decoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.decoder_backprop_enabled[block_idx] = False
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                conv.disable_update()

    def release_decoder(self, blocks="all"):
        if self.debug:
            print("release_decoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.decoder_backprop_enabled[block_idx] = True
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()

    def freeze_encoder(self, blocks="all"):
        if self.debug:
            print("freeze_encoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = False
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.disable_update()

    def release_encoder(self, blocks="all"):
        if self.debug:
            print("release_encoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = True
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()

    def freeze_pca(self):
        if self.debug:
            print("freeze_pca")
        self.pca.disable_update()

    def release_pca(self):
        if self.debug:
            print("release_pca")
        self.pca.enable_update()

    def reset_pca(self):
        if self.debug:
            print("reset_pca")
        self.pca_block_idx = self.tmp_n_blocks - 1
        # self.pca = PCA(self.channels[self.pca_block_idx], self.pca_dim)
        self.pca = PCA(self.pca_dim)
        self.pca.to_gpu()

    def add_downsample(self, strict=False):
        if self.debug:
            print("add_downsample")
        if self.n_blocks <= self.tmp_n_blocks:
            raise RuntimeError("maximum number of blocks. channel of the next block is not defined.")

        self.tmp_n_blocks += 1

        if self.pca_attached:
            if strict:
                raise RuntimeError("detach PCA before add_downsample()")
            else:
                warn("recommended: detach PCA before add_downsample()")
                print("PCA is automatically reset")
                self.reset_pca()


# noinspection PyDeprecation
class StackedFCAE_feature_before_relu(Chain):
    """
    RawFCAEのn_downsamplesをadaptiveに変化させられるようにしたもの．
    Bias項については全省略．
    input->in_augment->relu->(conv->relu)*n->subsample->(conv->relu)*n->(PCA)->(conv->relu)*n->upsample->(conv->relu)*n->out_augment->output

    downsampleが発生するごとにchannel数を増やす(0: 16, 1: 32, 2: 64, 3: 128)

    :param np.ndarray mask: マスク
    :param List[int] channels: conv_i_0, dcnv_i_0のchannel数．
    :param int n_conv_per_block: length of {conv_0_*}
    :param List[bool] decoder_backprop_enabled: {conv_i_*}, {dcnv_i_*}にbackpropするかどうか
    :param int n_blocks: maximum length of {conv_*_0}
    :param int tmp_n_blocks: current length of {conv_*_0}
    :param bool pca_attached: pcaをするかどうか．
    :param bool pca_loss_attached: pca lossを計算するかどうか
    :param bool reconstruction_loss_attached: reconstruction lossを計算するかどうか

    :meth reset_pca: self.pca is Noneならばself.pcaを設定する．そうでなければ，self.pcaをdelして再設定する
    :meth attach_pca:
    :meth increase_downsample: ダウンサンプルの数を学習可能な状態で増やす

    学習プラン
    1. simple auto encoder & lastly add pca
    init(n_downsamples=max_downsamples, with_pca=False, with_pca_loss=False)
    attach_pca() &  attach_pca_loss() & freeze_convolution() & detach_reconstruction_loss()
    detach_pca_loss() & release_convolution() & attach_reconstruction_loss()

    2. stacked auto encoder
    init(n_downsamples = 1, with_pca = True, with_pca_loss=False)
    .. starting from 1 is because starting from 0 is trivial (identity mapping is the global minimum)
    freeze_convolution(0, 1) & add_downsample() & reset_pca()
    .. n_downsamples = 2
    release_convolution(0, 1)
    freeze_convolution(0, 1, 2) & add_downsample() & reset_pca()
    .. n_downsamples = 3
    release_convolution(0, 1, 2)


    """
    # input size
    # extreme: (70, 88, 74) == crop:[(10, 80), (11, 99), (3, 77)]
    def __init__(self, mask, channels: Sequence[int], n_conv_per_block, pca_dim=1, init_n_blocks=None, with_reconstruction_loss=True, with_pca=False, with_pca_loss=False, debug=False):
        super().__init__()
        try:
            assert len(channels) >= 2
        except AssertionError as e:
            print("len(channels) must be equal or more than 2")
            raise e

        if init_n_blocks is None:
            init_n_blocks = len(channels)

        self.mask = mask
        self.channels = channels
        self.tmp_n_blocks = init_n_blocks
        self.n_blocks = len(self.channels)
        self.n_conv_per_block = n_conv_per_block
        self.loss_const = self.mask.size / self.mask.sum()

        self.decoder_backprop_enabled = [True] * self.n_blocks
        self.encoder_backprop_enabled = [True] * self.n_blocks

        self.pca_attached = with_pca
        self.pca_block_idx = self.tmp_n_blocks - 1

        self.pca_dim = pca_dim

        self.pca_loss_attached = with_pca_loss
        self.reconstruction_loss_attached = with_reconstruction_loss

        self.debug = debug

        with self.init_scope():
            # ConvolutionND(dim, in_channel, out_channel, kernel, stride, padding)
            for block_idx in range(self.n_blocks):
                for conv_idx in range(self.n_conv_per_block):
                    if conv_idx == 0:
                        in_channel = 1 if block_idx == 0 else self.channels[block_idx - 1]
                    else:
                        in_channel = self.channels[block_idx]
                    out_channel = self.channels[block_idx]
                    nobias = False if block_idx == 0 and conv_idx == 0 else True  # 入力層はnobias=False
                    self.add_link(
                        "conv_{}_{}".format(block_idx, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )
            # self.pca = PCA(self.channels[self.pca_block_idx], self.pca_dim)
            self.pca = PCA(self.pca_dim)
            for block_idx in reversed(range(self.n_blocks)):
                for conv_idx in range(self.n_conv_per_block):
                    in_channel = self.channels[block_idx]
                    if conv_idx == self.n_conv_per_block -1:
                        out_channel = 1 if block_idx == 0 else self.channels[block_idx - 1]
                    else:
                        out_channel = self.channels[block_idx]
                    nobias = False if block_idx == 0 and conv_idx == self.n_conv_per_block - 1 else True  # 出力層はnobias=False
                    self.add_link(
                        "dcnv_{}_{}".format(block_idx, conv_idx),
                        L.ConvolutionND(3, in_channel, out_channel, (3, 3, 3), 1, 1, nobias=nobias)
                    )

    def to_cpu(self):
        super().to_cpu()
        self.mask = to_cpu(self.mask)
        self.loss_const = to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = to_gpu(self.mask, device)
        self.loss_const = to_gpu(self.loss_const, device)

    def extract(self, x):
        assert self.reconstruction_loss_attached or self.pca_loss_attached
        assert self.pca_loss_attached or not self.pca_loss_attached

        original_shape = list(x.shape)
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)
        x_masked = F.reshape(F.scale(x, self.mask), new_shape)
        padding_history = []
        shape_history = []

        h = x_masked
        for downsample_degree in range(self.tmp_n_blocks):
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("conv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if not (downsample_degree == self.tmp_n_blocks - 1 and conv_idx == self.n_conv_per_block - 1):
                    if self.debug:  # 特徴抽出層でReLUせず，
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != self.tmp_n_blocks - 1:
                shape = h.shape[2:]
                shape_history.append(shape)
                padding = tuple([x % 2 for x in shape])
                padding_history.append(padding)
                if self.debug:
                    print("average_pooling_nd")
                h = F.average_pooling_nd(h, 2, 2, padding)  # dimensionの0側にpaddingがかかる
                if self.debug:
                    print("\t{}".format(h.shape))
        return h

    def calc(self, x, target):
        """
        :param xp.ndarray x:
        :param xp.ndarray target:
        :return: Variable
        """
        assert self.reconstruction_loss_attached or self.pca_loss_attached
        assert self.pca_attached or not self.pca_loss_attached

        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]
        x_masked = F.reshape(F.scale(x, self.mask), new_shape)
        padding_history = []
        shape_history = []

        h = x_masked
        for downsample_degree in range(self.tmp_n_blocks):
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("conv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if not (downsample_degree == self.tmp_n_blocks - 1 and conv_idx == self.n_conv_per_block - 1):
                    if self.debug:  # rawなので，特徴抽出層でReLUしない
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != self.tmp_n_blocks - 1:
                shape = h.shape[2:]
                shape_history.append(shape)
                padding = tuple([x % 2 for x in shape])
                padding_history.append(padding)
                if self.debug:
                    print("average_pooling_nd")
                h = F.average_pooling_nd(h, 2, 2, padding)  # dimensionの0側にpaddingがかかる
                if self.debug:
                    print("\t{}".format(h.shape))
        # この段階でhがfeature
        pca_loss = None
        if self.pca_attached:
            if self.debug:
                print("pca")
            feature = self.pca(h)
            if self.pca_loss_attached:
                pca_loss = F.mean_absolute_error(feature, h)
                report({'pca_loss': pca_loss}, self)
                if not self.reconstruction_loss_attached:
                    return pca_loss
            h = feature
            if self.debug:
                print("\t{}".format(h.shape))

        h = F.relu(h)
        if self.debug:
            print("relu")

        for downsample_degree in reversed(range(self.tmp_n_blocks)):
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("dcnv_{}_{}".format(downsample_degree, conv_idx))
                if self.debug:
                    print("dcnv_{}_{}".format(downsample_degree, conv_idx), conv.W.shape)
                h = conv(h)
                if self.debug:
                    print("\t{}".format(h.shape))
                if not (downsample_degree == 0 and conv_idx == self.n_conv_per_block - 1):  # 最終出力層はReLUしない
                    if self.debug:
                        print("relu")
                    h = F.relu(h)
            if downsample_degree != 0:
                shape = shape_history.pop()
                padding = padding_history.pop()
                if self.debug:
                    print("unpooling_nd")
                h = F.unpooling_nd(h, 2, 2, padding, shape, cover_all=False)
                if self.debug:
                    print("\t{}".format(h.shape))
        out = F.reshape(h, tuple(original_shape))
        out_masked = F.scale(out, self.mask)

        target_masked = F.scale(target, self.mask)

        reconstruction_loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const
        report({'reconstruction_loss': reconstruction_loss}, self)
        if self.pca_loss_attached:
            return reconstruction_loss + pca_loss
        else:
            return reconstruction_loss

    def __call__(self, x):
        """
        :param xp.ndarray x:
        :return: Variable
        """
        loss = self.calc(x, x)
        report({'loss': loss}, self)
        return loss

    def attach_pca(self, overwrite=False):
        if self.debug:
            print("attach_pca")
        # if self.tmp_n_blocks != self.pca_block_idx + 1:
        #     raise RuntimeError("self.tmp_n_blocks: {} != self.pca_block_idx: {} + 1".format(self.tmp_n_blocks, self.pca_block_idx))
        self.pca_attached = True

    def detach_pca(self, detach_pca_loss=True):
        if self.debug:
            print("detach_pca")
        self.pca_attached = False
        if detach_pca_loss:
            self.pca_loss_attached = False

    def attach_pca_loss(self):
        if self.debug:
            print("attach_pca_loss")
        if not self.pca_attached:
            raise RuntimeError("PCA is not attached")
        self.pca_loss_attached = True

    def detach_pca_loss(self):
        if self.debug:
            print("detach_pca_loss")
        self.pca_loss_attached = False

    def attach_reconstruction_loss(self):
        if self.debug:
            print("attach_reconstruction_loss")
        self.reconstruction_loss_attached = True

    def detach_reconstruction_loss(self):
        if self.debug:
            print("detach_reconstruction_loss")
        self.reconstruction_loss_attached = False

    def freeze_convolution(self, blocks="all"):
        if self.debug:
            print("feeeze_convolution {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = False
            self.decoder_backprop_enabled[block_idx] = False
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.disable_update()
                dcnv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                dcnv.disable_update()

    def release_convolution(self, blocks="all"):
        if self.debug:
            print("release_convolution {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = True
            self.decoder_backprop_enabled[block_idx] = True
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()
                conv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()

    def freeze_decoder(self, blocks="all"):
        if self.debug:
            print("freeze_decoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.decoder_backprop_enabled[block_idx] = False
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                conv.disable_update()

    def release_decoder(self, blocks="all"):
        if self.debug:
            print("release_decoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.decoder_backprop_enabled[block_idx] = True
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("dcnv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()

    def freeze_encoder(self, blocks="all"):
        if self.debug:
            print("freeze_encoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = False
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.disable_update()

    def release_encoder(self, blocks="all"):
        if self.debug:
            print("release_encoder {}".format(blocks))
        if blocks == "all":
            blocks = tuple(range(self.n_blocks))
        assert isinstance(blocks, Sequence[int])
        for block_idx in blocks:
            self.encoder_backprop_enabled[block_idx] = True
            for conv_idx in range(self.n_conv_per_block):
                conv = self.__getattribute__("conv_{}_{}".format(block_idx, conv_idx))
                conv.enable_update()

    def freeze_pca(self):
        if self.debug:
            print("freeze_pca")
        self.pca.disable_update()

    def release_pca(self):
        if self.debug:
            print("release_pca")
        self.pca.enable_update()

    def reset_pca(self):
        if self.debug:
            print("reset_pca")
        self.pca_block_idx = self.tmp_n_blocks - 1
        # self.pca = PCA(self.channels[self.pca_block_idx], self.pca_dim)
        self.pca = PCA(self.pca_dim)
        self.pca.to_gpu()

    def add_downsample(self, strict=False):
        if self.debug:
            print("add_downsample")
        if self.n_blocks <= self.tmp_n_blocks:
            raise RuntimeError("maximum number of blocks. channel of the next block is not defined.")

        self.tmp_n_blocks += 1

        if self.pca_attached:
            if strict:
                raise RuntimeError("detach PCA before add_downsample()")
            else:
                warn("recommended: detach PCA before add_downsample()")
                print("PCA is automatically reset")
                self.reset_pca()