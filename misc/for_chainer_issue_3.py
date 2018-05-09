from chainer import Link, Chain, Parameter, Variable
from chainer import links as L
import numpy as np


class CustomChain(Chain):
    def __init__(self, some_parameter):
        super().__init__()
        if isinstance(some_parameter, Link) or isinstance(some_parameter, Parameter):
            with self.init_scope():
                self.some_parameter = some_parameter
        else:
            self.add_persistent("some_parameter", some_parameter)


def print_children_params_persistent(c: Chain):
    print(c._children, c._params, c._persistent)


print_children_params_persistent(CustomChain(np.array([1, 2, 3])))
print_children_params_persistent(CustomChain(Variable(np.array([1, 2, 3]))))
print_children_params_persistent(CustomChain(L.Bias()))
print_children_params_persistent(CustomChain(Parameter(np.array([1, 2, 3]))))