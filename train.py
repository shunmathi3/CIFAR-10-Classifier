import argparse
import numpy as np
import matplotlib.pyplot as plt

from data import load_cifar10, DataSampler
from linear_classifier import LinearClassifier
from two_layer_net import TwoLayerNet
from optim import SGD
from layers import softmax_loss, l2_regularization
from utils import check_accuracy


parser = argparse.ArgumentParser()
parser.add_argument(
    '--plot-file',
    default='plot.pdf',
    help='File where loss and accuracy plot should be saved')
parser.add_argument(
    '--checkpoint-file',
    default='checkpoint.pkl',
    help='File where trained model weights should be saved')
parser.add_argument(
    '--print-every',
    type=int,
    default=25,
    help='How often to print losses during training')


def main(args):
    # data to use for training
    num_train = 20000

    # model architecture hyperparameters.
    hidden_dim = 16

    # optimization hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-4
    reg = 1.0

    data = load_cifar10(num_train=num_train)
    train_sampler = DataSampler(data['X_train'], data['y_train'], batch_size)
    val_sampler = DataSampler(data['X_val'], data['y_val'], batch_size)

    # Set up the model and optimizer
    model = TwoLayerNet(hidden_dim=hidden_dim)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)

    stats = {
        't': [],
        'loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    for epoch in range(1, num_epochs + 1):
        print(f'Starting epoch {epoch} / {num_epochs}')
        for i, (X_batch, y_batch) in enumerate(train_sampler):
            loss, grads = training_step(model, X_batch, y_batch, reg)
            optimizer.step(grads)
            if i % args.print_every == 0:
                print(f'  Iteration {i} / {len(train_sampler)}, loss = {loss}')
            stats['t'].append(i / len(train_sampler) + epoch - 1)
            stats['loss'].append(loss)

        print('Checking accuracy')
        train_acc = check_accuracy(model, train_sampler)
        print(f'  Train: {train_acc:.2f}')
        val_acc = check_accuracy(model, val_sampler)
        print(f'  Val:   {val_acc:.2f}')
        stats['train_acc'].append(train_acc)
        stats['val_acc'].append(val_acc)

    print(f'Saving plot to {args.plot_file}')
    plot_stats(stats, args.plot_file)
    print(f'Saving model checkpoint to {args.checkpoint_file}')
    model.save(args.checkpoint_file)


def training_step(model, X_batch, y_batch, reg):
    loss, grads = None, None
    scores, cache = model.forward(X_batch)
    loss, grad_scores = softmax_loss(scores, y_batch)

    for param_name, param_value in model.parameters().items():
        if param_name.startswith('W'):
            l2_loss, l2_grad = l2_regularization(param_value, reg)
            loss += l2_loss

    grads = model.backward(grad_scores, cache)
    return loss, grads


def plot_stats(stats, filename):
    plt.subplot(1, 2, 1)
    plt.plot(stats['t'], stats['loss'], 'o', alpha=0.5, ms=4)
    plt.title('Loss')
    plt.xlabel('Epoch')
    loss_xlim = plt.xlim()

    plt.subplot(1, 2, 2)
    epoch = np.arange(1, 1 + len(stats['train_acc']))
    plt.plot(epoch, stats['train_acc'], '-o', label='train')
    plt.plot(epoch, stats['val_acc'], '-o', label='val')
    plt.xlim(loss_xlim)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(12, 4)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main(parser.parse_args())
