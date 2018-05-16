from importlib import import_module

import chainer
import chainer.optimizers
from chainer.iterators import SerialIterator as Iterator
from chainer.training import Trainer
from chainer.training.extensions import snapshot, snapshot_object, LogReport, observe_lr, Evaluator, PrintReport, ProgressBar
from chainer.training.updater import ParallelUpdater as Updater
from chainer.serializers import load_npz

from train.config import get_config, destroy_config, log_config

from os import path

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
            print("mask loading...")
            load_mask_module = import_module(config["additional information"]["mask"]["loader"]["module"],
                                             config["additional information"]["mask"]["loader"]["package"])
            load_mask = getattr(load_mask_module, config["additional information"]["mask"]["loader"]["function"])
            mask = load_mask(**config["additional information"]["mask"]["loader"]["params"])
            print("done.")
            print("mask.shape: {}".format(mask.shape))
        except FileNotFoundError as e:
            raise e

        model_module = import_module(config["model"]["module"], config["model"]["package"])
        Model = getattr(model_module, config["model"]["class"])
        model = Model(mask=mask, **config["model"]["params"])
        finetune_config = config["additional information"]["finetune"]
        if finetune_config is not None:
            load_npz(path.join(finetune_config["directory"], finetune_config["file"]), model)

        dataset_module = import_module(config["dataset"]["module"], config["dataset"]["package"])
        Dataset = getattr(dataset_module, config["dataset"]["class"])
        train_dataset = Dataset(**config["dataset"]["train"]["params"])
        valid_dataset = Dataset(**config["dataset"]["valid"]["params"])

        train_iterator = Iterator(train_dataset, config["batch"]["train"], True, True)
        valid_iterator = Iterator(valid_dataset, config["batch"]["valid"], False, False)

        Optimizer = getattr(chainer.optimizers, config["optimizer"]["class"])
        optimizer = Optimizer(**config["optimizer"]["params"])

        optimizer.setup(model)

        for hook_config in config["optimizer"]["hook"]:
            hook_module = import_module(hook_config["module"], hook_config["package"])
            Hook = getattr(hook_module, hook_config["class"])
            hook = Hook(**hook_config["params"])
            optimizer.add_hook(hook)


        updater = Updater(train_iterator, optimizer, devices=gpu)

        trainer = Trainer(updater, **config["trainer"]["params"])
        trainer.extend(snapshot(), trigger=config["trainer"]["snapshot_interval"])
        trainer.extend(snapshot_object(model, "model_iter_{.updater.iteration}"), trigger=config["trainer"]["model_interval"])
        trainer.extend(observe_lr(), trigger=config["trainer"]["log_interval"])
        trainer.extend(LogReport(["epoch", "iteration", "main/loss", "main/loss_l1", "validation/main/loss", "validation/main/loss_l1", "lr", "elapsed_time"], trigger=config["trainer"]["log_interval"]))
        trainer.extend(Evaluator(valid_iterator, model, device=gpu), trigger=config["trainer"]["eval_interval"])
        trainer.extend(PrintReport(["epoch", "iteration", "main/loss", "main/loss_l1", "validation/main/loss", "validation/main/loss_l1", "lr", "elapsed_time"]), trigger=config["trainer"]["log_interval"])
        trainer.extend(ProgressBar(update_interval=1))
        trainer.run()
        log_config(config, "succeeded")

    except Exception as e:
        log_config(config, "unintentional termination")
        raise e

