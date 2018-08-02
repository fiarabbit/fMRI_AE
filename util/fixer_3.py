from os import path, listdir

import re

import numpy as np


def rm_persistent(root_dir):
    model_dir = path.join(root_dir, "model")
    files = listdir(model_dir)
    regex = re.compile("^(?P<mode>model|snapshot)_iter_[0-9]*$")
    for i, file in enumerate(files):
        matchobj = regex.match(file)
        if not matchobj or not path.isfile(path.join(model_dir, file)):
            print("skipping {}".format(file))
            continue
        else:
            print("processing {}".format(file))
            d = dict(np.load(path.join(model_dir, file)))
            mode = matchobj.group("mode")
            keys = ["mask", "_r", "out_mask", "loss_const", "in_mask"]
            if mode == "snapshot":
                keys = ["updater/model:main/" + x for x in keys]
            for key in keys:
                if key in d.keys():
                    del d[key]
            np.savez_compressed(path.join(model_dir, file), **d)


def main():
    target_list = [
        "PixelShufflerFCAE_E16D16",
        "PixelShufflerFCAE_E16D16_BN",
        "PixelShufflerFCAE_E32D32",
        "ReorgFCAE_E16D16",
        "ReorgPixelShufflerFCAE_E16D16",
        "ReorgPixelShufflerFCAE_E16D16_BN",
        "ReorgPixelShufflerFCAE_E16D16_feature16_BN",
        "SimpleFCAE_E16D16",
        "SimpleFCAE_E16D16_BN",
        "SimpleFCAE_E16D16_small",
        "SimpleFCAE_E32D32",
        "SimpleFCAE_E64D64",
        "SimpleFCAE_E8D8",
        "SimpleFCAE_E8D8_small"
    ]
    target_list = sorted(target_list)
    for target in target_list:
        root_dir = path.join("/efs/fMRI_AE", target)
        print("processing {}".format(root_dir))
        rm_persistent(root_dir)