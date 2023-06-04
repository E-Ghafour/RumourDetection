import torch
from torch.utils.data import DataLoader
import itertools
import numpy as np
import pandas as pd
import argparse
from models import my_RNN, my_GRU, my_LSTM
from preprocessing import my_dataset
import pickle
from evaluation import model_evaluation
from configparser import ConfigParser


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    num_correct = 0
    for batch, (X, y) in enumerate(dataloader):
        if isinstance(X, list):
            X = (X[0].to(device), X[1].to(device))
            y = y.to(device)
        else:
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

def save_submit_dataset(dataloader, model, device = 'cuda', csv_path = 'submit.csv'):
    model.eval()
    with torch.no_grad():
        preds = []
        for x in dataloader:
            preds += model(x.to(device))
    labels = torch.round(torch.tensor(preds)).tolist()
    input = np.array([range(20800, 26000), labels]).T
    pd.DataFrame(input , columns = ['id', 'label'], dtype=int).to_csv(csv_path, index= False)

def count_parameters(model):
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
    print(f'The model has {trainable:,} trainable parameters')
    print(f'The model has {total:,} parameters')

def evaluate_best_model(model_path = None, device = 'cuda', validation_dataset_path = 'validation_dataset.pkl'):
    config = ConfigParser()
    config.read('/content/RomourDetection/config.ini')
    x_test_path = config.get('DATA', 'x_test_path')
    pad_len = config.getint('MODEL_INFO', 'pad_len')
    submit_path = config.get('GENERAL', 'submit_model_path')
    report_evaluation = config.getboolean('GENERAL', 'report_evaluation')
    batch_size = config.getint('MODEL_INFO', 'batch_size')
    embedding_type = config.get('MODEL_INFO', 'embedding_type')
    best_model_path = config.get('GENERAL', 'best_model_path')


    model_path = best_model_path if model_path == None else model_path
    model = torch.load(model_path)
    model.to(device)

    useen_dataset = my_dataset.RumorDataset(x_test_path, [], pad_len, have_label=False, embedding_type = embedding_type)

    unseen_dataloader = DataLoader(useen_dataset, 64)

    save_submit_dataset(dataloader = unseen_dataloader,
                    model = model,
                    device = device,
                    csv_path = submit_path)

    with open(validation_dataset_path, 'rb') as ff:
        validation_dataset = pickle.load(ff)

    validation_dataLoader = DataLoader(dataset=validation_dataset,
                                   batch_size = batch_size,
                                   shuffle = True)

    if(report_evaluation):
        y_pred, y_validation = predict_validation_label(model=model,
                                                          dataloader=validation_dataLoader,
                                                          device=device
                                                          )
        model_evaluation.report_model_evaluation(y_pred=y_pred,
                                             y=y_validation
                                             )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file_name', type=str)
    # args = parser.parse_args()
    # file_name = args.file_name
    evaluate_best_model()

    