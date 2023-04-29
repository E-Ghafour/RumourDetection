import torch
import itertools
import numpy as np
import pandas as pd

def count_parameters(model):
    f_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of trainable features: {f_num}')

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

def predict_validation_label(model, dataloader, device):
    model.eval()
    Y = []
    preds = []
    with torch.no_grad():
        for x, y in dataloader:
            Y += y.tolist()
            pred = model(x.to(device))
            preds += torch.round(pred).tolist()
    preds = list(itertools.chain.from_iterable(preds))
    y = list(itertools.chain.from_iterable(y))
    return preds, Y

def save_submit_dataset(dataloader, model, device, csv_path):
    model.eval()
    with torch.no_grad():
        preds = []
        for x in dataloader:
            preds += model(x.to(device))
    labels = torch.round(torch.tensor(preds)).tolist()
    input = np.array([range(20800, 26000), labels]).T
    pd.DataFrame(input , columns = ['id', 'label'], dtype=int).to_csv(csv_path, index= False)
