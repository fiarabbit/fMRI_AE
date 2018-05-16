"""this file was created to fix bug in serialization.
in chainer, all variables in self._persistent must not be Variable.
however, I mistook that principle and many fucking objects was created by the mistake.
とにかく，model.maskの持ち方をVariable->ndarrayにする必要がある．
"""

import numpy as np

from os import path, mkdir, listdir, chdir
from shutil import copy2, rmtree

import re
import nibabel as nib
import yaml

import warnings


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f)
        return path.join(config["additional information"]["mask"]["directory"], config["additional information"]["mask"]["file"]), config["additional information"]["crop"],


def load_mask_nib(mask_path: str, crop: list):
    mask = np.asarray(nib.load(mask_path).get_data(), dtype=np.float32)
    if mask.shape[0] != crop[0][1] - crop[0][0] or mask.shape[1] != crop[1][1] - crop[1][0] or mask.shape[2] != crop[2][1] - crop[2][0]:
        slice_ = [slice(*crop[i]) for i in range(3)]
        mask = mask[slice_]
    try:
        assert (1 == mask[mask.nonzero()]).all()
    except AssertionError:
        warnings.warn("Non-bool mask")
        print("converting to boolean...")
        mask[mask.nonzero()] = 1
    return mask


def save_original(directory):
    original_dir = path.join(directory, "original")
    if not path.exists(original_dir):
        mkdir(original_dir)
    files = sorted(listdir(directory))
    for i, file in enumerate(files):
        print("{:3d}/{:3d}".format(i, len(files)), end=" ")
        if path.isfile(path.join(directory, file)):
            if not path.isfile(path.join(original_dir, file)):
                print("copying {} -> {}".format(path.join(directory, file), path.join(original_dir, file)))
                copy2(path.join(directory, file), path.join(original_dir, file))
            else:
                print("skipping {}".format(path.join(directory, file)))


def fix_experiment(root_dir):
    model_dir = path.join(root_dir, "model")
    save_original(model_dir)

    config_dir = path.join(root_dir, "log")
    config_path = path.join(config_dir, "config.yml")
    mask_path, crop = load_config(config_path)
    mask = load_mask_nib(mask_path, crop)

    files = sorted(listdir(model_dir))
    regex = re.compile("^(?P<mode>model|snapshot)_iter_[0-9]*$")
    for i, file in enumerate(files):
        print("{:3d}/{:3d}".format(i, len(files)), end=" ")
        match_obj = regex.match(file)
        if not match_obj:
            print("skipping {}".format(file))
            continue
        else:
            if match_obj.group("mode") == "model":
                key = "mask"
            else:
                key = "updater/model:main/mask"
            print("processing {}".format(file))
            _d = np.load(path.join(model_dir, file))
            _mask = _d[key]
            if isinstance(_mask, np.ndarray) and _mask.dtype == np.dtype('f'):
                print("skipping {}".format(file))
                continue
            elif isinstance(_mask, np.ndarray) and _mask.dtype == np.dtype('O'):
                d = dict(_d)
                d[key] = mask
                with open(path.join(model_dir, file), "wb") as f:
                    np.savez_compressed(f, **d)
            else:
                raise TypeError("unexpected type of {}".format(_d[key].dtype))


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
        print("{:3d}/{:3d}".format(i, len(target_list)), end=" ")
        fix_experiment(path.join(root_dir, target))
