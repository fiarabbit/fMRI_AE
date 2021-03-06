from copy import copy

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.backends.cuda
from .links.PixelShuffler import PixelShuffler3D
from .links.ResNet import ResBlock3D

from train.links.Reorg import Reorg

import numpy as np

import pdb


class SimplestFCAE(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    # costcut: (, 88, )
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        s1 = F.average_pooling_nd(x_masked, 2, 2, 0)
        s2 = F.average_pooling_nd(s1, 2, 2, 0)
        s3 = F.average_pooling_nd(s2, 2, 2, 0)
        s4 = F.unpooling_nd(s3, 2, 2, 0, tuple([x * 2 for x in s3.shape[2:]]), False)
        s5 = F.unpooling_nd(s4, 2, 2, 0, tuple([x * 2 for x in s4.shape[2:]]), False)
        s6 = F.unpooling_nd(s5, 2, 2, 0, tuple([x * 2 for x in s5.shape[2:]]), False)
        out = F.reshape(s6, tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E64D64_feature64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    # costcut: (, 88, )
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 64, 64, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            # self.pixel_shuffler = PixelShuffler3D(self._r)
            # self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target
        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = self.dcnv8(b7)
        out = F.reshape(b8, tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_EVD64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 128, 256, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 256, 256, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 256, 512, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 512, 512, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 512, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            # self.pixel_shuffler = PixelShuffler3D(self._r)
            # self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = self.dcnv8(b7)
        out = F.reshape(b8, tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_EVD64_Small(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 32, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 64, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 128, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            # self.pixel_shuffler = PixelShuffler3D(self._r)
            # self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = self.dcnv8(b7)
        out = F.reshape(b8, tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_EVD64_Small_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(32)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(32)
            self.conv5 = L.ConvolutionND(3, 32, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(64)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(64)
            self.conv7 = L.ConvolutionND(3, 64, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(128)
            self.conv8 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(128)
            self.conv_extract = L.ConvolutionND(3, 128, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(64)
            self.dcnv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(64)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(64)
            self.dcnv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(64)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(64)
            self.dcnv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(64)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(64)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            # self.pixel_shuffler = PixelShuffler3D(self._r)
            # self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = self.bnc3(F.relu(self.conv3(s1)))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = self.bnc5(F.relu(self.conv5(s2)))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = self.bnc7(F.relu(self.conv7(s3)))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.dcnv2(b1)))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = self.bnd3(F.relu(self.dcnv3(s4)))
        b4 = self.bnd4(F.relu(self.dcnv4(b3)))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = self.bnd5(F.relu(self.dcnv5(s5)))
        b6 = self.bnd6(F.relu(self.dcnv6(b5)))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = self.bnd7(F.relu(self.dcnv7(s6)))
        b8 = self.dcnv8(b7)
        out = F.reshape(b8, tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class PixelShufflerFCAE_E64D64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 64, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            # self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class PixelShufflerFCAE_EVD64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 128, 256, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 256, 256, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 256, 512, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 512, 512, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 512, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            # self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_E64D64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64 * self._r ** 3, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 64 * self._r ** 3, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 64 * self._r ** 3, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 64, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(self.reorg(c2)))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(self.reorg(c4)))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(self.reorg(c6)))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_EVD64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64 * self._r ** 3, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 128 * self._r ** 3, 256, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 256, 256, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 256 * self._r ** 3, 512, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 512, 512, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 512, 1, (1, 1, 1), 1, 0, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 64, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(self.reorg(c2)))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(self.reorg(c4)))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(self.reorg(c6)))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_EVD64_Small(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 32 * self._r ** 3, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 64 * self._r ** 3, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 128, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(self.reorg(c2)))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(self.reorg(c4)))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(self.reorg(c6)))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_EVD64_Small_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(32)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(32)
            self.conv5 = L.ConvolutionND(3, 32 * self._r ** 3, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(64)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(64)
            self.conv7 = L.ConvolutionND(3, 64 * self._r ** 3, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(128)
            self.conv8 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(128)
            self.conv_extract = L.ConvolutionND(3, 128, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(64)
            self.dcnv2 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(64)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(64)
            self.dcnv4 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(64)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(64)
            self.dcnv6 = L.ConvolutionND(3, 64, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(64)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(64)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        c3 = self.bnc3(F.relu(self.conv3(self.reorg(c2))))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        c5 = self.bnc5(F.relu(self.conv5(self.reorg(c4))))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        c7 = self.bnc7(F.relu(self.conv7(self.reorg(c6))))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.pixel_shuffler(self.dcnv2(b1))))
        b3 = self.bnd3(F.relu(self.dcnv3(b2)))
        b4 = self.bnd4(F.relu(self.pixel_shuffler(self.dcnv4(b3))))
        b5 = self.bnd5(F.relu(self.dcnv5(b4)))
        b6 = self.bnd6(F.relu(self.pixel_shuffler(self.dcnv6(b5))))
        b7 = self.bnd7(F.relu(self.dcnv7(b6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_EVDV_Small_BN_feature128(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(32)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(32)
            self.conv5 = L.ConvolutionND(3, 32 * self._r ** 3, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(64)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(64)
            self.conv7 = L.ConvolutionND(3, 64 * self._r ** 3, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(128)
            self.conv8 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(128)
            self.conv_extract = L.ConvolutionND(3, 128, 128, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 128, 128, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(128)
            self.dcnv2 = L.ConvolutionND(3, 128, 64 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(64)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(64)
            self.dcnv4 = L.ConvolutionND(3, 64, 32 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(32)
            self.dcnv5 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(32)
            self.dcnv6 = L.ConvolutionND(3, 32, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(16)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(16)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        c3 = self.bnc3(F.relu(self.conv3(self.reorg(c2))))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        c5 = self.bnc5(F.relu(self.conv5(self.reorg(c4))))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        c7 = self.bnc7(F.relu(self.conv7(self.reorg(c6))))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.pixel_shuffler(self.dcnv2(b1))))
        b3 = self.bnd3(F.relu(self.dcnv3(b2)))
        b4 = self.bnd4(F.relu(self.pixel_shuffler(self.dcnv4(b3))))
        b5 = self.bnd5(F.relu(self.dcnv5(b4)))
        b6 = self.bnd6(F.relu(self.pixel_shuffler(self.dcnv6(b5))))
        b7 = self.bnd7(F.relu(self.dcnv7(b6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E16D16(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E16D16_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(16)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(16)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(16)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(16)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(16)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(16)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(16)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(16)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(16)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(16)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(16)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(16)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(16)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = self.bnc3(F.relu(self.conv3(s1)))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = self.bnc5(F.relu(self.conv5(s2)))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = self.bnc7(F.relu(self.conv7(s3)))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.dcnv2(b1)))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = self.bnd3(F.relu(self.dcnv3(s4)))
        b4 = self.bnd4(F.relu(self.dcnv4(b3)))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = self.bnd5(F.relu(self.dcnv5(s5)))
        b6 = self.bnd6(F.relu(self.dcnv6(b5)))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = self.bnd7(F.relu(self.dcnv7(s6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class PixelShufflerFCAE_E16D16(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class PixelShufflerFCAE_E32D32(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 32, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 32, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 32, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 32, 32 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 32, 32 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 32, 32 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 32, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class PixelShufflerFCAE_E16D16_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(16)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(16)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(16)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(16)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(16)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(16)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(16)
            self.dcnv2 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(16)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(16)
            self.dcnv4 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(16)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(16)
            self.dcnv6 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(16)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(16)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = self.bnc3(F.relu(self.conv3(s1)))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = self.bnc5(F.relu(self.conv5(s2)))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = self.bnc7(F.relu(self.conv7(s3)))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.pixel_shuffler(self.dcnv2(b1))))
        b3 = self.bnd3(F.relu(self.dcnv3(b2)))
        b4 = self.bnd4(F.relu(self.pixel_shuffler(self.dcnv4(b3))))
        b5 = self.bnd5(F.relu(self.dcnv5(b4)))
        b6 = self.bnd6(F.relu(self.pixel_shuffler(self.dcnv6(b5))))
        b7 = self.bnd7(F.relu(self.dcnv7(b6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgFCAE_E16D16(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(self.reorg(c2)))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(self.reorg(c4)))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(self.reorg(c6)))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgFCAE_E16D16_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(16)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(16)
            self.conv5 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(16)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(16)
            self.conv7 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(16)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(16)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(16)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(16)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(16)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(16)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(16)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(16)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(16)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        c3 = self.bnc3(F.relu(self.conv3(self.reorg(c2))))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        c5 = self.bnc5(F.relu(self.conv5(self.reorg(c4))))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        c7 = self.bnc7(F.relu(self.conv7(self.reorg(c6))))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.dcnv2(b1)))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = self.bnd3(F.relu(self.dcnv3(s4)))
        b4 = self.bnd4(F.relu(self.dcnv4(b3)))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = self.bnd5(F.relu(self.dcnv5(s5)))
        b6 = self.bnd6(F.relu(self.dcnv6(b5)))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = self.bnd7(F.relu(self.dcnv7(s6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_E16D16(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(self.reorg(c2)))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(self.reorg(c4)))
        c6 = F.relu(self.conv6(c5))
        c7 = F.relu(self.conv7(self.reorg(c6)))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.pixel_shuffler(self.dcnv2(b1)))
        b3 = F.relu(self.dcnv3(b2))
        b4 = F.relu(self.pixel_shuffler(self.dcnv4(b3)))
        b5 = F.relu(self.dcnv5(b4))
        b6 = F.relu(self.pixel_shuffler(self.dcnv6(b5)))
        b7 = F.relu(self.dcnv7(b6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_E16D16_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(16)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(16)
            self.conv5 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(16)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(16)
            self.conv7 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(16)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(16)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(16)
            self.dcnv2 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(16)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(16)
            self.dcnv4 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(16)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(16)
            self.dcnv6 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(16)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(16)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        c3 = self.bnc3(F.relu(self.conv3(self.reorg(c2))))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        c5 = self.bnc5(F.relu(self.conv5(self.reorg(c4))))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        c7 = self.bnc7(F.relu(self.conv7(self.reorg(c6))))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.pixel_shuffler(self.dcnv2(b1))))
        b3 = self.bnd3(F.relu(self.dcnv3(b2)))
        b4 = self.bnd4(F.relu(self.pixel_shuffler(self.dcnv4(b3))))
        b5 = self.bnd5(F.relu(self.dcnv5(b4)))
        b6 = self.bnd6(F.relu(self.pixel_shuffler(self.dcnv6(b5))))
        b7 = self.bnd7(F.relu(self.dcnv7(b6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class ReorgPixelShufflerFCAE_E16D16_feature16_BN(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc2 = L.BatchNormalization(16)
            self.conv3 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc3 = L.BatchNormalization(16)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc4 = L.BatchNormalization(16)
            self.conv5 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc5 = L.BatchNormalization(16)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc6 = L.BatchNormalization(16)
            self.conv7 = L.ConvolutionND(3, 16 * self._r ** 3, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc7 = L.BatchNormalization(16)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnc8 = L.BatchNormalization(16)
            self.conv_extract = L.ConvolutionND(3, 16, 16, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd1 = L.BatchNormalization(16)
            self.dcnv2 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd2 = L.BatchNormalization(16)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd3 = L.BatchNormalization(16)
            self.dcnv4 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd4 = L.BatchNormalization(16)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd5 = L.BatchNormalization(16)
            self.dcnv6 = L.ConvolutionND(3, 16, 16 * self._r ** 3, (3, 3, 3), 1, 1, nobias=True)
            self.bnd6 = L.BatchNormalization(16)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.bnd7 = L.BatchNormalization(16)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)
            self.pixel_shuffler = PixelShuffler3D(self._r)
            self.reorg = Reorg(self._r)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = self.bnc1(F.relu(self.conv1(x_masked)))
        c2 = self.bnc2(F.relu(self.conv2(c1)))
        c3 = self.bnc3(F.relu(self.conv3(self.reorg(c2))))
        c4 = self.bnc4(F.relu(self.conv4(c3)))
        c5 = self.bnc5(F.relu(self.conv5(self.reorg(c4))))
        c6 = self.bnc6(F.relu(self.conv6(c5)))
        c7 = self.bnc7(F.relu(self.conv7(self.reorg(c6))))
        c8 = self.bnc8(F.relu(self.conv8(c7)))
        feature = self.conv_extract(c8)
        b1 = self.bnd1(F.relu(self.dcnv1(feature)))
        b2 = self.bnd2(F.relu(self.pixel_shuffler(self.dcnv2(b1))))
        b3 = self.bnd3(F.relu(self.dcnv3(b2)))
        b4 = self.bnd4(F.relu(self.pixel_shuffler(self.dcnv4(b3))))
        b5 = self.bnd5(F.relu(self.dcnv5(b4)))
        b6 = self.bnd6(F.relu(self.pixel_shuffler(self.dcnv6(b5))))
        b7 = self.bnd7(F.relu(self.dcnv7(b6)))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E8D8(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 8, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 8, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 8, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E32D32(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 32, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 32, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 32, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 32, 32, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 32, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def extract(self, x: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        return feature

    def reconstruct(self, x):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out
        return out_masked


    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E64D64(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 64, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 64, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 64, 64, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 64, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E16D16_ResBlock(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            self.bias = L.Bias()
            if self.in_mask == "concat":
                self.conv0 = L.ConvolutionND(3, 2, 16, (1, 1, 1), 1, 0, nobias=False)
            else:
                self.conv0 = L.ConvolutionND(3, 1, 16, (1, 1, 1), 1, 0, nobias=False)
            self.conv1 = ResBlock3D(16, 16)
            self.conv2 = ResBlock3D(16, 16)
            self.conv3 = ResBlock3D(16, 16)
            self.conv4 = ResBlock3D(16, 16)
            self.conv5 = ResBlock3D(16, 16)
            self.conv6 = ResBlock3D(16, 16)
            self.conv7 = ResBlock3D(16, 16)
            self.conv8 = ResBlock3D(16, 16)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv0 = L.ConvolutionND(3, 1, 16, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = ResBlock3D(16, 16)
            self.dcnv2 = ResBlock3D(16, 16)
            self.dcnv3 = ResBlock3D(16, 16)
            self.dcnv4 = ResBlock3D(16, 16)
            self.dcnv5 = ResBlock3D(16, 16)
            self.dcnv6 = ResBlock3D(16, 16)
            self.dcnv7 = ResBlock3D(16, 16)
            self.dcnv8 = ResBlock3D(16, 16)
            self.dcnv_out = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c0 = F.relu(self.conv0(x_masked))
        c1 = F.relu(self.conv1(c0))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b0 = F.relu(self.dcnv0(feature))
        b1 = F.relu(self.dcnv1(b0))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = F.relu(self.dcnv8(b7))
        out = F.reshape(self.dcnv_out(b8), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E16D16_small(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    # symmetry: (72, 88, 80)
    # this: (80, 96, 80)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv9 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv10 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv9 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv10 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        s4 = F.average_pooling_nd(c8, 2, 2, 0)
        c9 = F.relu(self.conv9(s4))
        c10 = F.relu(self.conv10(c9))
        feature = self.conv_extract(c10)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = F.relu(self.dcnv8(b7))
        s7 = F.unpooling_nd(b8, 2, 2, 0, tuple([x * 2 for x in b8.shape[2:]]), False)
        b9 = F.relu(self.dcnv9(s7))
        out = F.reshape(self.dcnv10(b9), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E8D8_L1(chainer.Chain):
    # unlearnable

    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str, l=0.01):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        assert (isinstance(mask, np.ndarray))
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.l = np.asarray(l, dtype=float)
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = np.asarray(1, dtype=float)
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 8, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 8, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 8, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)
        self.l = chainer.cuda.to_cpu(self.l)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)
        self.l = chainer.cuda.to_gpu(self.l, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const
        loss_l1 = F.mean_absolute_error(feature, self.xp.zeros(feature.shape, dtype=self.xp.float32))

        chainer.report({'loss_l1': loss_l1}, self)

        return loss, loss_l1

    def __call__(self, x: chainer.Variable):
        loss, loss_l1 = self.calc(x, x)
        chainer.report({'loss': loss, 'loss_l1': loss_l1}, self)
        return loss + self.l * loss_l1


class SimpleFCAE_E4D4_small(chainer.Chain):
    # unlearnable

    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    # symmetry: (72, 88, 80)
    # this: (80, 96, 80)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 4, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 4, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv9 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv10 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 4, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv9 = L.ConvolutionND(3, 4, 4, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv10 = L.ConvolutionND(3, 4, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        s4 = F.average_pooling_nd(c8, 2, 2, 0)
        c9 = F.relu(self.conv9(s4))
        c10 = F.relu(self.conv10(c9))
        feature = self.conv_extract(c10)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = F.relu(self.dcnv8(b7))
        s7 = F.unpooling_nd(b8, 2, 2, 0, tuple([x * 2 for x in b8.shape[2:]]), False)
        b9 = F.relu(self.dcnv9(s7))
        out = F.reshape(self.dcnv10(b9), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E8D8_ReLU(chainer.Chain):
    """結果：学習が進まない
    featureにReLUを入れてはいけないようだ．
    """
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 8, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 8, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 8, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(F.relu(feature)))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E8D8_small(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    # symmetry: (72, 88, 80)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 8, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv0 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv00 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 8, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv00 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv0 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 8, 1, (3, 3, 3), 1, 1, nobias=False)
        self.conv1.disable_update()
        self.conv2.disable_update()
        self.conv3.disable_update()
        self.conv4.disable_update()
        self.conv5.disable_update()
        self.conv6.disable_update()
        self.conv7.disable_update()
        self.conv8.disable_update()
        self.dcnv1.disable_update()
        self.dcnv2.disable_update()
        self.dcnv5.disable_update()
        self.dcnv4.disable_update()
        self.dcnv5.disable_update()
        self.dcnv6.disable_update()
        self.dcnv7.disable_update()
        self.dcnv8.disable_update()

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        s4 = F.average_pooling_nd(c8, 2, 2, 0)
        c9 = F.relu(self.conv0(s4))
        c10 = F.relu(self.conv00(c9))
        feature = self.conv_extract(c10)
        b1 = F.relu(self.dcnv00(feature))
        b2 = F.relu(self.dcnv0(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv1(s4))
        b4 = F.relu(self.dcnv2(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv4(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv5(s6))
        b8 = F.relu(self.dcnv6(b7))
        s7 = F.unpooling_nd(b8, 2, 2, 0, tuple([x * 2 for x in b8.shape[2:]]), False)
        b9 = F.relu(self.dcnv7(s7))
        out = F.reshape(self.dcnv8(b9), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class Linear(chainer.Chain):
    def __init__(self, mask: np.ndarray):
        super(Linear, self).__init__()
        assert mask.shape == (72, 88, 80)
        self.mask = mask
        with self.init_scope():
            self.l1 = L.Linear(self.xp.count_nonzero(self.mask), 990)
            self.l2 = L.Linear(990, self.xp.count_nonzero(self.mask))

    def to_cpu(self):
        super(Linear, self).to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)

    def to_gpu(self, device=None):
        super(Linear, self).to_gpu()
        self.mask = chainer.cuda.to_gpu(self.mask, device)

    def calc(self, x):
        x_masked = x[[Ellipsis] + list(self.xp.nonzero(self.mask))]
        l1 = F.relu(self.l1(x_masked))
        y = self.l2(l1)
        loss = F.mean_absolute_error(x_masked, y)
        return loss

    def __call__(self, x: np.ndarray):
        loss = self.calc(x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E16D16_ResBlock_small(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            self.bias = L.Bias()
            if self.in_mask == "concat":
                self.conv0 = L.ConvolutionND(3, 2, 16, (1, 1, 1), 1, 0, nobias=False)
            else:
                self.conv0 = L.ConvolutionND(3, 1, 16, (1, 1, 1), 1, 0, nobias=False)
            self.conv1 = ResBlock3D(16, 16)
            self.conv2 = ResBlock3D(16, 16)
            self.conv3 = ResBlock3D(16, 16)
            self.conv4 = ResBlock3D(16, 16)
            self.conv5 = ResBlock3D(16, 16)
            self.conv6 = ResBlock3D(16, 16)
            self.conv7 = ResBlock3D(16, 16)
            self.conv8 = ResBlock3D(16, 16)
            self.conv9 = ResBlock3D(16, 16)
            self.conv10 = ResBlock3D(16, 16)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv0 = L.ConvolutionND(3, 1, 16, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = ResBlock3D(16, 16)
            self.dcnv2 = ResBlock3D(16, 16)
            self.dcnv3 = ResBlock3D(16, 16)
            self.dcnv4 = ResBlock3D(16, 16)
            self.dcnv5 = ResBlock3D(16, 16)
            self.dcnv6 = ResBlock3D(16, 16)
            self.dcnv7 = ResBlock3D(16, 16)
            self.dcnv8 = ResBlock3D(16, 16)
            self.dcnv9 = ResBlock3D(16, 16)
            self.dcnv10 = ResBlock3D(16, 16)
            self.dcnv_out = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c0 = F.relu(self.conv0(x_masked))
        c1 = F.relu(self.conv1(c0))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        s4 = F.average_pooling_nd(c8, 2, 2, 0)
        c9 = F.relu(self.conv9(s4))
        c10 = F.relu(self.conv10(c9))
        feature = self.conv_extract(c10)
        b0 = F.relu(self.dcnv0(feature))
        b1 = F.relu(self.dcnv1(b0))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        b8 = F.relu(self.dcnv8(b7))
        s7 = F.unpooling_nd(b8, 2, 2, 0, tuple([x * 2 for x in b8.shape[2:]]), False)
        b9 = F.relu(self.dcnv9(s7))
        b10 = F.relu(self.dcnv10(b9))
        out = F.reshape(self.dcnv_out(b10), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E8D8_linear(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 8, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.Linear(9 * 11 * 10 * 8, 9 * 11 * 10)
            self.dcnv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 8, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        feature = F.reshape(feature, c8.shape)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss

class SimpleFCAE_E16D16_wo_BottleNeck(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        b1 = F.relu(self.dcnv1(c8))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss

class SimpleFCAE_E8D8_wo_BottleNeck(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 8, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 8, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 8, 8, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 8, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        b1 = F.relu(self.dcnv1(c8))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x: chainer.Variable):
        loss = self.calc(x, x)
        chainer.report({'loss': loss}, self)
        return loss

class SimpleFCAE_E16D16_Denoise(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv_extract = L.ConvolutionND(3, 16, 1, (1, 1, 1), 1, 0, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        feature = self.conv_extract(c8)
        b1 = F.relu(self.dcnv1(feature))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x_noise: chainer.Variable, x_truth: chainer.Variable):
        loss = self.calc(x_noise, x_truth)
        chainer.report({'loss': loss}, self)
        return loss


class SimpleFCAE_E16D16_wo_BottleNeck_Denoise(chainer.Chain):
    # input size
    # None: (91, 109, 91)
    # cubic: (88, 88, 88)
    # extreme: (70, 88, 74)
    def __init__(self, mask, r: int, in_mask: str, out_mask: str):
        assert in_mask in ("mask", "concat", "none")
        assert out_mask in ("mask", "none")
        # in_mask <- ["mask", "concat", "none"]
        # out_mask <- ["mask", "none"]
        super().__init__()
        self.mask = mask
        self._r = r
        self.in_mask = in_mask
        self.out_mask = out_mask
        if self.out_mask == "mask":
            self.loss_const = self.mask.size / self.mask.sum()
        elif self.out_mask == "none":
            self.loss_const = 1
        with self.init_scope():
            # ConvolutionND(dim, inchannel, outchannel, kernel, stride, padding)
            if self.in_mask == "concat":
                self.conv1 = L.ConvolutionND(3, 2, 16, (3, 3, 3), 1, 1, nobias=False)
            else:
                self.conv1 = L.ConvolutionND(3, 1, 16, (3, 3, 3), 1, 1, nobias=False)
            self.conv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.conv8 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv1 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv2 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv3 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv4 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv5 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv6 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv7 = L.ConvolutionND(3, 16, 16, (3, 3, 3), 1, 1, nobias=True)
            self.dcnv8 = L.ConvolutionND(3, 16, 1, (3, 3, 3), 1, 1, nobias=False)

    def to_cpu(self):
        super().to_cpu()
        self.mask = chainer.cuda.to_cpu(self.mask)
        self.loss_const = chainer.cuda.to_cpu(self.loss_const)

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask = chainer.cuda.to_gpu(self.mask, device)
        self.loss_const = chainer.cuda.to_gpu(self.loss_const, device)

    def calc(self, x: chainer.Variable, target: chainer.Variable):
        original_shape = list(x.shape)  # [batch, dim1, dim2, dim3]
        new_shape = copy(original_shape)
        new_shape.insert(1, 1)  # [batch, 1, dim1, dim2, dim3]

        if self.in_mask == "concat":
            x = x.reshape(x, tuple(new_shape))
            x_masked = F.hstack((F.reshape(x, new_shape), F.broadcast_to(self.mask, x.shape)))
        else:
            x_masked = F.reshape(F.scale(x, self.mask), new_shape)

        if self.out_mask == "mask":
            target_masked = F.scale(target, self.mask)
        else:
            target_masked = target

        c1 = F.relu(self.conv1(x_masked))
        c2 = F.relu(self.conv2(c1))
        s1 = F.average_pooling_nd(c2, 2, 2, 0)
        c3 = F.relu(self.conv3(s1))
        c4 = F.relu(self.conv4(c3))
        s2 = F.average_pooling_nd(c4, 2, 2, 0)
        c5 = F.relu(self.conv5(s2))
        c6 = F.relu(self.conv6(c5))
        s3 = F.average_pooling_nd(c6, 2, 2, 0)
        c7 = F.relu(self.conv7(s3))
        c8 = F.relu(self.conv8(c7))
        b1 = F.relu(self.dcnv1(c8))
        b2 = F.relu(self.dcnv2(b1))
        s4 = F.unpooling_nd(b2, 2, 2, 0, tuple([x * 2 for x in b2.shape[2:]]), False)
        b3 = F.relu(self.dcnv3(s4))
        b4 = F.relu(self.dcnv4(b3))
        s5 = F.unpooling_nd(b4, 2, 2, 0, tuple([x * 2 for x in b4.shape[2:]]), False)
        b5 = F.relu(self.dcnv5(s5))
        b6 = F.relu(self.dcnv6(b5))
        s6 = F.unpooling_nd(b6, 2, 2, 0, tuple([x * 2 for x in b6.shape[2:]]), False)
        b7 = F.relu(self.dcnv7(s6))
        out = F.reshape(self.dcnv8(b7), tuple(original_shape))

        if self.out_mask == "mask":
            out_masked = F.scale(out, self.mask)
        else:
            out_masked = out

        loss = F.mean_absolute_error(out_masked, target_masked) * self.loss_const

        return loss

    def __call__(self, x_noise: chainer.Variable, x_truth: chainer.Variable):
        loss = self.calc(x_noise, x_truth)
        chainer.report({'loss': loss}, self)
        return loss
