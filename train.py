import torch
import torch.nn as nn
from preprocessing import data_preprocessing, dataset
from models import my_RNN
from evaluation import model_evaluation, epoch_evaluation


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    num_correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # acc
        num_correct += (torch.round(pred) == y).type(torch.float).sum().item()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f'train accuracy: {(100*num_correct/len(dataloader.dataset)):>0.01f}')


if __name__ == '__main__':
    #create/tune your own model here...

    pass
