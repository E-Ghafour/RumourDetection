import torch
import torch.nn as nn
from preprocessing import data_preprocessing, dataset
from models import my_RNN
from evaluation import model_evaluation, epoch_evaluation
from configparser import ConfigParser


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

    config = ConfigParser()
    config.read('config.ini')

    #create/tune your own model here...

    #preprocess the data and save it to the file
    fake_news_train_path = config.get('DATA', 'fake_news_train_path')
    remove_punc = config.getboolean('PREPROCESSING', 'remove_punc')
    lower_case = config.getboolean('PREPROCESSING', 'lower_case')
    remove_stopwords = config.getboolean('PREPROCESSING', 'remove_stopwords')
    steam = config.getboolean('PREPROCESSING', 'steam')

    X, y = data_preprocessing.preprocess(fake_news_train_path,
                                        fillna='unk',
                                        remove_punc=remove_punc,
                                        lower_case=lower_case,
                                        remove_stopwords=remove_stopwords,
                                        steam=steam)
    
    data_preprocessing.save_data(X, 'x_train.data')
    data_preprocessing.save_data(y, 'y_train.data')
















    model_type = config.get('MODEL_INFO', 'type')
    input_size = config.getint('MODEL_INFO', 'input_size')
    hidden_size = config.getint('MODEL_INFO', 'hidden_size')
    output_size = config.getint('MODEL_INFO', 'output_size')
    learning_rate = config.getfloat('MODEL_INFO', 'learning_rate')
    batch_size = config.getint('MODEL_INFO', 'batch_size')
    n_layer = config.getint('MODEL_INFO', 'n_layer')
    bidirectional = config.getboolean('MODEL_INFO', 'bidirectional')
    inner_dropout = config.getfloat('MODEL_INFO', 'inner_dropout')
    dropout = config.getfloat('MODEL_INFO', 'dropout')
    epochs = config.getint('MODEL_INFO', 'epochs')
    pad_len = config.getint('MODEL_INFO', 'pad_len')
    trainable_embedding = config.getboolean('MODEL_INFO', 'trainable_embedding')
    embedding_type = config.get('MODEL_INFO', 'embedding_type')



    


    
    
    
