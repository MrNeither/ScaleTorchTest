import argparse
import logging

import numpy as np
import scaletorch as st
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from nets.Net1 import Net1

st.init(verbose=None)

logger = logging.getLogger('exp1')
logger.setLevel(logging.DEBUG)


def load_dataset():
    files = st.list_files('https://disk.yandex.com/d/ONfjkcdy7dpRtw', pattern="*.npy")
    print(files)
    assert len(files) == 2
    with st.open('https://disk.yandex.com/d/mQO53oNnYOpJEA') as f:
        X = np.load(f)

    with st.open('https://disk.yandex.com/d/Z8zKEhEYdZGUTA') as f:
        Y = np.load(f)
    return X, Y


def train(args):
    assert args.tesst == 10

    X, Y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    train_dataset = TensorDataset(torch.tensor(X_train).cuda(),
                                  torch.tensor(y_train, dtype=torch.long).cuda())  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # create your dataloader

    test_dataset = TensorDataset(torch.tensor(X_test).cuda(),
                                 torch.tensor(y_test, dtype=torch.long).cuda())  # create your datset
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)  # cr

    model = Net1().cuda()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    errs = []
    accs = []
    softmax = torch.nn.Softmax(dim=1)
    correct = 0
    for x, y in test_dataloader:
        y_ = model(x)
        y_ = softmax(y_).argmax(dim=1)
        correct += (y_ == y).float().sum().cpu()

    accuracy = 100 * correct / len(test_dataset)
    print("Accuracy = {}".format(accuracy))

    for step in range(args.epochs):
        tot_err = 0
        for x, y in train_dataloader:
            optimizer.zero_grad()
            y_ = model(x)
            err: torch.Tensor = loss(y_, y)
            tot_err += err.detach().cpu().item()
            err.backward()
            optimizer.step()

        errs.append(tot_err / len(train_dataset))
        print('Error = {}'.format(errs[-1]), end='    ')
        st.track(epoch=step, metrics={'train_loss': errs[-1]}, tuner_default='train_loss')
        if step % 1 == 0:
            model.eval()
            correct = 0
            for x, y in test_dataloader:
                y_ = model(x)
                y_ = softmax(y_).argmax(dim=1)
                correct += (y_ == y).float().sum()

            accuracy = 100 * correct / len(test_dataset)
            print("Accuracy = {}".format(accuracy))
            accs.append(accuracy.item())
            model.train()
            st.track(epoch=step, metrics={'accuracy': accuracy.item()}, tuner_default='accuracy')


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--tesst",
            type=int,
            help="test arg",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=5,
            metavar="N",
            help="number of epochs to train (default: 10)",
        )

        train(parser.parse_args())

    except Exception as exception:
        logger.exception(exception)
        raise
