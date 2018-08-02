import chainer


def IncreaseDownsample(trainer: chainer.training.Trainer):
    trainer.updater.get_optimizer("main").target.n_downsamples += 1