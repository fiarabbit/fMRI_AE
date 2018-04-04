from util import assert_dir, assert_file, yaml_dump, yaml_dump_log
import awsutil

from datetime import datetime
from os import listdir, path, mkdir
from shutil import copy2, rmtree
import uuid


def get_config(experiment_name=None):
    config = {
        "general":
            {
                "name": "",
                "date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                "hash": str(uuid.uuid4())[0:8]
            },
        "dataset":
            {
                "directory": {
                    "train": None,
                    "valid": None,
                },
                "file": {
                    "train": [],
                    "valid": [],
                },
                "module": "dataset",
                "class": "NpyCroppedDataset",
                "train": {
                    "params": {
                        "directory": "/data/npy/train",
                    }
                },
                "valid": {
                    "params": {
                        "directory": "/data/npy/valid",
                    }
                }

            },
        "model":
            {
                "module": "model",
                "class": "SimpleFCAE_E16D16"
            },
        "optimizer": {
            "class": "MomentumSGD",
            "params": {
                "lr": 1,
                "momentum": 0.9
            },
            "WeightDecay": 0.0001,
            "ExponentialShift": {
                "params": {
                    "attr": "lr",
                    "rate": 0.1,
                }
            }
        },
        "trainer": {
            "stop_trigger": [1000, "epoch"],
            "model_interval": [1, "epoch"],
            "log_interval": [100, "iteration"],
            "eval_interval": [1, "epoch"],
            "ExponentialShift": [10, "epoch"]
        },
        "batch":
            {
                "train": 20,
                "valid": 20,
                "test": 10
            },
        "device": [0, 1, 2, 3, 4, 5, 6, 7],
        "save": {
            "root": "/out",
            "directory": None,
            "log": {
                "directory": None,
                "file": "config.yml"
            },
            "model": {
                "directory": None
            }
        },
        "additional information":
            {
                "mask": {
                    "directory": "/data/mask",
                    "file": "average_optthr.nii",
                    "crop": [[9, 81], [11, 99], [0, 80]]
                },
                "ec2": {
                    "instance-id": awsutil.get_instanceid(),
                    "volume-id": awsutil.get_volumeids()
                },
                "model_params": {
                    "r": 2,
                    "in_mask": "mask",
                    "out_mask": "mask",
                }
            }
    }
    # assert file existence
    config["dataset"]["directory"]["train"] = config["dataset"]["train"]["params"]["directory"]
    config["dataset"]["directory"]["valid"] = config["dataset"]["valid"]["params"]
    try:
        assert_dir(config["dataset"]["directory"]["train"])
        assert_dir(config["dataset"]["directory"]["valid"])
    except FileNotFoundError as e:
        print(e)
        exit(1)

    config["dataset"]["file"]["train"].extend(sorted(listdir(config["dataset"]["directory"]["train"])))
    config["dataset"]["file"]["valid"].extend(sorted(listdir(config["dataset"]["directory"]["valid"])))

    try:
        assert_dir(config["save"]["root"])
    except FileNotFoundError as e:
        raise e

    try:
        assert_file(path.join(config["additional information"]["mask"]["directory"], config["additional information"]["mask"]["file"]))
    except FileNotFoundError as e:
        raise e

    if experiment_name is None:
        experiment_name = input("name your experiment: ")
        config["general"]["name"] = experiment_name

    if config["general"]["name"] == "" or path.exists(path.join(config["save"]["root"], experiment_name)):
        config["general"]["name"] = config["general"]["name"] + config["general"]["hash"]
        print("your experiment was renamed to {}".format(config["general"]["name"]))

    try:
        config["save"]["directory"] = path.join(config["save"]["root"], config["general"]["name"])
        config["save"]["log"]["directory"] = path.join(config["save"]["directory"], "log")
        config["save"]["model"]["directory"] = path.join(config["save"]["directory"], "model")
        mkdir(config["save"]["directory"])
        mkdir(config["save"]["log"]["directory"])
        mkdir(config["save"]["model"]["directory"])

    except FileExistsError as e:
        print("hash collision happened")
        raise e

    try:
        copy2(config["model"]["module"] + ".py", path.join(config["save"]["directory"], 'model.py'))
        copy2(config["dataset"]["module"] + ".py", path.join(config["save"]["directory"], 'dataset.py'))
    except FileNotFoundError as e:
        print("While copying model.py and dataset.py, the file not found.")
        raise e

    try:
        copy2("train.py", path.join(config["save"]["directory"], "train.py"))
    except FileNotFoundError:
        print("While copying train.py, the file not found")
        print("Continue configuration...")

    with open(path.join(config["save"]["log"]["directory"], config["save"]["log"]["file"]), "w") as f:
        print(yaml_dump(config), file=f)

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

