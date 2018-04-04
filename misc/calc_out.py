def calc_out(in_size=None, kernel=None, stride=None, padding=None,):
    def get(o, q):
        if o is None:
            o = int(input(q))
        return o
    in_size = get(in_size, "in_size: ")
    kernel = get(kernel, "kernel: ")
    stride = get(stride, "stride: ")
    padding = get(padding, "padding: ")
    return (in_size + 2 * padding - kernel) / stride + 1