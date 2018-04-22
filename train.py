from config import get_config, destroy_config, log_config
from util import yaml_dump

import chainer
from chainer.iterators import SerialIterator as Iterator
import chainer.optimizers
from chainer.optimizer import WeightDecay
from chainer.training.updater import StandardUpdater as Updater
from chainer.training import Trainer
from chainer.training.extensions import snapshot_object, LogReport, observe_lr, Evaluator, PrintReport, ExponentialShift, ProgressBar

from os import path
from importlib import import_module


def main():
    config = get_config()
    # print("configured as follows:")
    # print(yaml_dump(config))
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
            load_mask_module = import_module(config["additional information"]["mask"]["loader"]["module"])
            load_mask = getattr(load_mask_module, config["additional information"]["mask"]["loader"]["function"])
            mask = load_mask(**config["additional information"]["mask"]["loader"]["params"])
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
        model = Model(mask=mask, **config["model"]["params"])
        try:
            chainer.cuda.get_device_from_id(0).use()
            gpu = 0
            model.to_gpu()
            print("GPU enabled")
        except RuntimeError:
            gpu = -1
            print("GPU disabled")

        dataset_module = import_module(config["dataset"]["module"])
        Dataset = getattr(dataset_module, config["dataset"]["class"])
        train_dataset = Dataset(**config["dataset"]["train"]["params"])
        valid_dataset = Dataset(**config["dataset"]["valid"]["params"])

        train_iterator = Iterator(train_dataset, config["batch"]["train"], True, True)
        valid_iterator = Iterator(valid_dataset, config["batch"]["valid"], False, False)

        Optimizer = getattr(chainer.optimizers, config["optimizer"]["class"])
        optimizer = Optimizer(**config["optimizer"]["params"])

        optimizer.setup(model)

        for hook_config in config["optimizer"]["hook"]:
            hook_module = import_module(hook_config["module"])
            Hook = getattr(hook_module, hook_config["class"])
            hook = Hook(**hook_config["params"])
            optimizer.add_hook(hook)


        updater = Updater(train_iterator, optimizer, device=gpu)

        trainer = Trainer(updater, **config["trainer"]["params"])
        trainer.extend(snapshot_object(model, "model_iter_{.updater.iteration}"), trigger=config["trainer"]["model_interval"])
        trainer.extend(observe_lr(), trigger=config["trainer"]["log_interval"])
        trainer.extend(LogReport(["epoch", "iteration", "main/loss", "validation/main/loss", "lr", "elapsed_time"], trigger=config["trainer"]["log_interval"]))
        trainer.extend(Evaluator(valid_iterator, model, device=gpu), trigger=config["trainer"]["eval_interval"])
        trainer.extend(PrintReport(["epoch", "iteration", "main/loss", "validation/main/loss", "lr", "elapsed_time"]), trigger=config["trainer"]["log_interval"])
        trainer.extend(ProgressBar(update_interval=1))
        trainer.run()
        log_config(config, "succeeded")

    except Exception as e:
        log_config(config, "unintentional termination")
        raise e


if __name__ == '__main__':
    main()
