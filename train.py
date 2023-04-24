import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocessing import data_preprocessing, my_dataset
from models import my_RNN
from evaluation import model_evaluation, epoch_evaluation
from configparser import ConfigParser


config = ConfigParser()
config.read('config.ini')

skip_preprocessing = config.getboolean('GENERAL', 'skip_preprocessing')

if(not skip_preprocessing):
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


#training info
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



x_train_path = config.get('DATA', 'x_train_path')
y_train_path = config.get('DATA', 'y_train_path')

dataset = my_dataset.RumorDataset(tokenize_data_path=x_train_path,
                                  labels_path=y_train_path,
                                  pad_len=pad_len,
                                  have_label=True,
                                  embedding_type=embedding_type,
                                  )

train_dataLoader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              )

# validation_dataLoader = DataLoader(dataset=)






















    


    
    
    
