from util import assert_dir, assert_file, yaml_dump, yaml_dump_log
from helper.config_helper import get_savedir
import awsutil

from datetime import datetime
from os import listdir, path, mkdir, walk, makedirs
from shutil import copy2, rmtree
import uuid


def get_config():
    config = {
        "general":
            {
                "name": "",
                "date": datetime.now().strftime("%Y/%m/%d_%H:%M:%S"),
                "hash": str(uuid.uuid4())[0:8]
            },
        "device": [0],  # 0, if using GPUs
        "dataset": {
            "module": "dataset",  # == dataset.py
            "class": "NibDataset",
            "train": {
                "params": {
                    "directory": "/data/train"
                },
                "file": None  # to be automatically configured
            },
            "valid": {
                "params": {
                    "directory": "/data/valid"
                },
                "file": None  # to be automatically configured
            }
        },
        "model": {
            "module": "model",  # == model.py
            "class": "ReorgPixelShufflerFCAE_E64D64",
            "params": {
                # "mask" parameter is NOT to be configured in config.py
                "r": 2,
                "in_mask": "mask",
                "out_mask": "mask"
            }
        },
        "optimizer": {
            "module": "chainer.optimizers",
            "class": "Adam",
            "params": {
            },
            "hook":
                [
                    {
                        "module": "chainer.optimizer",
                        "class": "WeightDecay",
                        "params": {
                            "rate": 0.0001
                        }
                    }
                ]
        },
        "trainer": {
            "params": {
                "stop_trigger": [1000, "epoch"],
                "out": None  # to be automaticaly configured
            },
            "model_interval": [1, "epoch"],
            "log_interval": [100, "iteration"],
            "eval_interval": [1, "epoch"]
        },
        "batch": {
            "train": 32,
            "valid": 32
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
                "mask": {
                    "directory": "/data/npy/mask",
                    "file": "average_optthr.npy",
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

    config["general"]["name"] = get_savedir(config["save"]["root"], config["general"])

    config["save"]["directory"] = path.join(config["save"]["root"], config["general"]["name"])
    mkdir(config["save"]["directory"])
    config["save"]["program"]["directory"] = path.join(config["save"]["directory"], "program")
    mkdir(config["save"]["program"]["directory"])
    config["save"]["log"]["directory"] = path.join(config["save"]["directory"], "log")
    mkdir(config["save"]["log"]["directory"])
    config["save"]["model"]["directory"], config["trainer"]["params"]["out"] = [path.join(config["save"]["directory"], "model")] * 2
    mkdir(config["save"]["model"]["directory"])

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

