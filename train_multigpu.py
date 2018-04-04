from config import get_config, destroy_config, log_config
from util import yaml_dump
from mask_loader import load_mask

from dataset import NibDataset

import chainer
from chainer.iterators import MultiprocessIterator as Iterator
import chainer.optimizers
from chainer.optimizer import WeightDecay
from chainer.training.updater import StandardUpdater as Updater
from chainer.training.updaters import MultiprocessParallelUpdater
from chainer.training import Trainer
from chainer.training.extensions import snapshot_object, LogReport, observe_lr, Evaluator, PrintReport, ExponentialShift, ProgressBar

from os import path
from importlib import import_module


def main():
    config = get_config()
    print("configured as follows:")
    print(yaml_dump(config))
    while True:
        s = input("ok? (y/n):")
        if s == 'y' or s == 'Y':
            log_config(config, "training start")
            break
        elif s == 'n' or s == 'N':
            destroy_config(config)
            exit(1)
    try:
        try:
            mask = load_mask(path.join(config["additional information"]["mask"]["directory"], config["additional information"]["mask"]["file"]))
        except FileNotFoundError as e:
            raise e
        try:
            if config["additional information"]["mask"]["crop"]:
                slice_t = config["additional information"]["mask"]["crop"]
                slice_ = [slice(*slice_t[i]) for i in range(3)]
                mask = mask[slice_]
                print("mask was cropped to {}".format(slice_t))
        except KeyError:
            print("mask was not cropped")

        model_module = import_module(config["model"]["module"])
        Model = getattr(model_module, config["model"]["class"])
        model = Model(mask=mask, **config["additional information"]["model_params"])

        train_dataset = NibDataset(config["dataset"]["directory"]["train"], crop=config["additional information"]["mask"]["crop"])
        valid_dataset = NibDataset(config["dataset"]["directory"]["valid"], crop=config["additional information"]["mask"]["crop"])

        # train_iterator = Iterator(train_dataset, config["batch"]["train"], True, True, 16, 2)
        train_iterators = [Iterator(i, config["batch"]["train"], True, True, 16) for i in chainer.datasets.split_dataset_n_random(train_dataset, len(config["device"]))]
        # valid_iterator = Iterator(valid_dataset, config["batch"]["valid"], False, False, 16, 2)
        valid_iterator = Iterator(valid_dataset, config["batch"]["valid"], False, False, 16)
        Optimizer = getattr(chainer.optimizers, config["optimizer"]["class"])
        optimizer = Optimizer(**config["optimizer"]["params"])

        optimizer.setup(model)
        try:
            optimizer.add_hook(WeightDecay(config["optimizer"]["WeightDecay"]))
            print("Weight Decay was set to {}".format(config["optimizer"]["WeightDecay"]))
        except KeyError:
            print("Weight Decay was not set")

        # updater = Updater(train_iterator, optimizer, device=gpu)
        updater = MultiprocessParallelUpdater(train_iterators, optimizer, devices=config["device"])

        trainer = Trainer(updater, config["trainer"]["stop_trigger"], config["save"]["model"]["directory"])
        trainer.extend(snapshot_object(model, "model_iter_{.updater.iteration}"), trigger=config["trainer"]["model_interval"])
        trainer.extend(observe_lr(), trigger=config["trainer"]["log_interval"])
        trainer.extend(LogReport(["epoch", "iteration", "main/loss", "validation/main/loss", "lr", "elapsed_time"], trigger=config["trainer"]["log_interval"]))
        trainer.extend(Evaluator(valid_iterator, model, device=config["device"][0]), trigger=config["trainer"]["eval_interval"])
        trainer.extend(PrintReport(["epoch", "iteration", "main/loss", "validation/main/loss", "lr", "elapsed_time"]), trigger=config["trainer"]["log_interval"])
        trainer.extend(ProgressBar(update_interval=10))
        try:
            trainer.extend(ExponentialShift(**config["optimizer"]["ExponentialShift"]["params"]), trigger=config["trainer"]["ExponentialShift"])
            print("Exponential Shift was set")
        except KeyError:
            print("Exponential Shift was not set")

        trainer.run()
        log_config(config, "succeeded")

    except Exception as e:
        log_config(config, "unintentional termination")
        raise e


if __name__ == '__main__':
    main()
