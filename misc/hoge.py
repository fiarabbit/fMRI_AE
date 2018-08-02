import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import updaters

# Network definition


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    batchsize = 32
    gpus = [0, 1]
    unit = 1000
    loaderjob = 2


    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(unit, 10))

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    devices = tuple(gpus)

    kf = chainer.datasets.get_cross_validation_datasets_random(train, n_fold=5)
    for i, (train, val) in enumerate(kf):

        train_iters = [
            chainer.iterators.MultiprocessIterator(i,
                                                   batchsize,
                                                   n_processes=loaderjob)
            for i in chainer.datasets.split_dataset_n_random(train, len(devices))]

        # Set up a trainer
        updater = updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                       devices=devices)


if __name__ == '__main__':
    main()
