import uuid
from datetime import datetime
from os import listdir, path, mkdir, walk, makedirs
from shutil import copy2, rmtree

from . import awsutil

from .util import assert_dir, assert_file, yaml_dump_log

from collections import OrderedDict
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        d = yaml.load(f)
    return d


def get_config():
    training_package = "train"
    config = {
        "general":
            {
                "name": "",
                "date": datetime.now().strftime("%Y/%m/%d_%H:%M:%S"),
                "hash": str(uuid.uuid4())[0:8]
            },
        "dataset": {
            "package": training_package,
            "module": ".dataset",  # == dataset.py
            "class": "NibDataset",
            "train": {
                "params": {
                    "directory": "/data/train",
                    "crop": None  # to be automatically configured
                },
                "file": None  # to be automatically configured
            },
            "valid": {
                "params": {
                    "directory": "/data/valid",
                    "crop": None  # to be automatically configured
                },
                "file": None  # to be automatically configured
            }
        },
        "model": {
            "package": training_package,
            "module": ".model_multigpu",  # == model_multigpu.py
            "class": "StackedResBlockMN",
            "params": {
                "feature_dim": 1000,
                # "encoder_channels": [32, 64, 128, 256, 512, 1024, 2048],
                "encoder_channels": [32, 32, 64, 64, 128, 128, 256],
                "encoder_layers": [3, 3, 3, 3, 3, 3, 3],
                # "decoder_channels": [32, 64, 128, 256, 512, 1024, 2048],
                "decoder_channels": [32, 32, 64, 64, 128, 128, 256],
                "decoder_layers": [3, 3, 3, 3, 3, 3, 3]
            }
        },
        "optimizer": {
            "package": "chainer.optimizers",
            "module": ".adam",
            "class": "Adam",
            "params": {
            },
            "hook":
                [
                    {
                        "package": "chainer",
                        "module": ".optimizer",
                        "class": "WeightDecay",
                        "params": {
                            "rate": 0.0001
                        }
                    }
                ]
        },
        "trainer": {
            "params": {
                "stop_trigger": [200000, "iteration"],
                "out": None  # to be automatically configured
            },
            "model_interval": [1000, "iteration"],
            "log_interval": [10, "iteration"],
            "eval_interval": [100, "iteration"],
            "snapshot_interval": [1000, "iteration"]
        },
        "batch": {
            "train": 2,
            "valid": 2
        },
        "save": {
            "root": "/out",
            "directory": None,  # to be automatically configured
            "program": {
                "directory": None  # to be automatically configured
            },
            "log": {
                "directory": None,  # to be automatically configured
                "file": "config.yml"
            },
            "model": {
                "directory": None  # to be automatically configured
            }
        },
        "additional information":
            {
                # "crop": [[9, 81], [11, 99], [0, 80]],
                # "crop": [[5, 85], [7, 103], [0, 80]],
                # "crop": [[10, 80], [11, 99], [3, 77]],
                "crop": [[0, 91], [0, 109], [0, 91]],
                "mask": {
                    "directory": "/data/mask",
                    "file": "average_optthr.nii",
                    "loader": {
                        "package": training_package,
                        "module": ".mask_loader",
                        "function": "load_mask_nib",
                        "params": {
                            "mask_path": None,  # to be automatically configured
                            "crop": None  # to be automatically configured
                        }
                    }
                },
                "ec2": {
                    "instance-id": awsutil.get_instanceid(),
                    "volume-id": awsutil.get_volumeids()
                }
            }
    }

    try:
        assert_dir(config["dataset"]["train"]["params"]["directory"])
        assert_dir(config["dataset"]["valid"]["params"]["directory"])
        assert_dir(config["save"]["root"])
    except FileNotFoundError as e:
        print(e)
        exit(1)

    config["dataset"]["train"]["file"] = sorted(listdir(config["dataset"]["train"]["params"]["directory"]))
    config["dataset"]["valid"]["file"] = sorted(listdir(config["dataset"]["valid"]["params"]["directory"]))

    config["dataset"]["train"]["params"]["crop"] = config["additional information"]["crop"]
    config["dataset"]["valid"]["params"]["crop"] = config["additional information"]["crop"]

    # TODO: write some code to share experiment name across workers
    config["general"]["name"] = "karioki"

    config["save"]["directory"] = path.join(config["save"]["root"], config["general"]["name"])
    try:
        mkdir(config["save"]["directory"])
    except Exception:
        pass
    config["save"]["program"]["directory"] = path.join(config["save"]["directory"], "program")
    try:
        mkdir(config["save"]["program"]["directory"])
    except Exception:
        pass
    config["save"]["log"]["directory"] = path.join(config["save"]["directory"], "log")
    try:
        mkdir(config["save"]["log"]["directory"])
    except Exception:
        pass
    config["save"]["model"]["directory"], config["trainer"]["params"]["out"] = [path.join(config["save"]["directory"], "model")] * 2
    try:
        mkdir(config["save"]["model"]["directory"])
    except Exception:
        pass

    for root, dirs, files in walk("."):
        for file in files:
            body, ext = path.splitext(file)
            if ext == ".py":
                src = path.join(root, file)
                dest = path.join(config["save"]["program"]["directory"], path.join(root, file))
                try:
                    makedirs(path.dirname(dest))
                except FileExistsError:
                    pass
                copy2(src, dest)

    config["additional information"]["mask"]["loader"]["params"]["mask_path"] = path.join(config["additional information"]["mask"]["directory"], config["additional information"]["mask"]["file"])
    config["additional information"]["mask"]["loader"]["params"]["crop"] = config["additional information"]["crop"]

    with open(path.join(config["save"]["log"]["directory"], config["save"]["log"]["file"]), "a") as f:
        print(yaml_dump_log(config), file=f)


    return config


def destroy_config(config_):
    rmtree(config_["save"]["directory"])
    raise RuntimeError("configure destroyed")


def log_config(config_, status):
    log_path = "/efs/fMRI_AE.config.log"
    try:
        assert_file(log_path)
    except FileNotFoundError as e:
        raise e

    d = {
        "status": status,
        "name": config_["general"]["name"],
        "hash": config_["general"]["hash"],
        "instance": config_["additional information"]["ec2"]["instance-id"],
        "volume": config_["additional information"]["ec2"]["volume-id"]
    }

    with open(log_path, "a") as f:
        print(yaml_dump_log(d), file=f)

