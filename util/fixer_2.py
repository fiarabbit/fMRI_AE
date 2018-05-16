"""rename (model|snapshot)_iter_[0-9]*.npz -> (model|snapshot)_iter_[0-9]*
fixer.pyで何故かどうあがいても.npz拡張子がついちゃうので，一斉リネームを行う
"""
from os import path, listdir, remove
from shutil import move

import re


def rename(root_dir: str):
    directory = path.join(root_dir, "model")
    files = sorted(listdir(directory))
    regex = re.compile("(?P<stem>^(?P<mode>model|snapshot)_iter_[0-9]]*).npz$")
    for i, file in enumerate(files):
        matchobj = regex.match(file)
        if not matchobj or not path.isfile(path.join(directory, file)):
            continue
        else:
            mode = matchobj.group("mode")
            stem = matchobj.group("stem")
            if stem in files:
                remove(path.join(directory, stem))
            move(path.join(directory, file), path.join(directory, stem))


def main():
    root_dir = "/efs/fMRI_AE"
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
    for i, target in enumerate(target_list):
        rename(path.join(root_dir, target))
