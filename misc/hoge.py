from chainer import Chain, Variable
from chainer import links as L
from chainer.serializers import save_npz, load_npz

class CustomChain(Chain):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self._persistent.add("mask")
        with self.init_scope():
            self.bn1 = L.BatchNormalization(64)


c1 = CustomChain(1)
c2 = CustomChain(2)
save_npz("save", c1)
load_npz("save", c2)
print(c2.mask)