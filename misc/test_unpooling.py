from chainer.functions import unpooling_nd
import numpy as np

B = 3
C = 2
H = 11
W = 11
D = 11

unpooling_nd(np.arange(B * C * H * W * D, dtype=np.float32).reshape((B, C, H, W, D)), 2, 2, 0, (22, 22, 22), False)