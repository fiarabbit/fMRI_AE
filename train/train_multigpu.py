from importlib import import_module

import chainer
import chainer.backends.cuda
import chainer.optimizers
from chainer.iterators import SerialIterator as Iterator
from chainer.training import Trainer
from chainer.training.extensions import LogReport, observe_lr, Evaluator, PrintReport, ProgressBar
from chainer.training.updater import StandardUpdater as Updater

import chainermn as mn

from train.config_multigpu import get_config, destroy_config, log_config

from os import path

import numpy as np


def main():
    comm = mn.create_communicator("pure_nccl")
    device = comm.intra_rank

    config = get_config()

    print("pid {}: mask loading...".format(comm.rank))
    load_mask_module = import_module(config["additional information"]["mask"]["loader"]["module"],
                                     config["additional information"]["mask"]["loader"]["package"])
    load_mask = getattr(load_mask_module, config["additional information"]["mask"]["loader"]["function"])
    mask = load_mask(**config["additional information"]["mask"]["loader"]["params"])
    print("pid {}: done.".format(comm.rank))
    if comm.rank == 0:
        print("mask.shape: {}".format(mask.shape))

    model_module = import_module(config["model"]["module"], config["model"]["package"])
    Model = getattr(model_module, config["model"]["class"])
    model = Model(comm=comm, mask=mask, **config["model"]["params"])

    optimizer_module = import_module(config["optimizer"]["module"], config["optimizer"]["package"])
    Optimizer = getattr(optimizer_module, config["optimizer"]["class"])
    optimizer = mn.create_multi_node_optimizer(Optimizer(**config["optimizer"]["params"]), comm)
    optimizer.setup(model)

    if device >= 0:
        chainer.backends.cuda.get_device_from_id(device).use()
        model.to_gpu()
        print("pid {}: GPU {} enabled".format(comm.rank, device))

    if comm.rank == 0:
        dataset_module = import_module(config["dataset"]["module"], config["dataset"]["package"])
        Dataset = getattr(dataset_module, config["dataset"]["class"])
        train_dataset = Dataset(**config["dataset"]["train"]["params"])
        valid_dataset = Dataset(**config["dataset"]["valid"]["params"])
    else:
        train_dataset = None
        valid_dataset = None

    train_dataset = mn.datasets.scatter_dataset(train_dataset, comm, shuffle=True)
    valid_dataset = mn.datasets.scatter_dataset(valid_dataset, comm, shuffle=True)

    train_iterator = Iterator(train_dataset, config["batch"]["train"])
    valid_iterator = Iterator(valid_dataset, config["batch"]["valid"], False, False)

    updater = Updater(train_iterator, optimizer, device=device)
    trainer = Trainer(updater, **config["trainer"]["params"])

    checkpointer = mn.create_multi_node_checkpointer(config["general"]["name"], comm)
    checkpointer.maybe_load(trainer, optimizer)
    trainer.extend(checkpointer, trigger=tuple(config["trainer"]["snapshot_interval"]))

    evaluator = Evaluator(valid_iterator, model, device=device)
    evaluator = mn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    trainer.extend(observe_lr(), trigger=config["trainer"]["log_interval"])
    if comm.rank == 0:
        trainer.extend(LogReport(trigger=config["trainer"]["log_interval"]))
        trainer.extend(PrintReport(["epoch", "iteration", "main/loss", "validation/main/loss"]), trigger=config["trainer"]["log_interval"])
        trainer.extend(ProgressBar(update_interval=1))

    trainer.run()



