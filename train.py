import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from preprocessing import data_preprocessing, my_dataset
from models import my_RNN, my_GRU, my_LSTM, my_Bert_LSTM, my_Bert_GRU, my_Bert_RNN
from evaluation import model_evaluation, epoch_evaluation
from configparser import ConfigParser
import utils
from utils import save_submit_dataset
import gc
import pickle

SEED = 119
torch.random.seed = SEED
np.random.seed(SEED)

config = ConfigParser()
# config.read('config.ini')
config.read('/content/RomourDetection/config.ini')


skip_preprocessing = config.getboolean('GENERAL', 'skip_preprocessing')

if(not skip_preprocessing):
    #preprocess the data and save it to the file
    title_col_name = config.get('DATA', 'title_col_name')
    text_col_name = config.get('DATA', 'text_col_name')
    label_col_name = config.get('DATA', 'label_col_name')
    fake_news_train_path = config.get('DATA', 'fake_news_train_path')
    remove_punc = config.getboolean('PREPROCESSING', 'remove_punc')
    lower_case = config.getboolean('PREPROCESSING', 'lower_case')
    remove_stopwords = config.getboolean('PREPROCESSING', 'remove_stopwords')
    steam = config.getboolean('PREPROCESSING', 'steam')

    X, y = data_preprocessing.preprocess(fake_news_train_path,
                                        title_col_name=title_col_name,
                                        text_col_name=text_col_name,
                                        label_col_name=label_col_name,
                                        fillna='<unk>',
                                        remove_punc=remove_punc,
                                        lower_case=lower_case,
                                        remove_stopwords=remove_stopwords,
                                        steam=steam)

    data_preprocessing.save_data(X, 'x_train2.data')
    data_preprocessing.save_data(y, 'y_train2.data')
    print('the data is saved...')
    print('sent the skip_preprocessing to True and run again to have more resources')
    exit()

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
validation_size = config.getfloat('MODEL_INFO', 'validation_size')
trainable_last_encoder = config.getboolean('MODEL_INFO', 'trainable_last_encoder')
bert_type = config.get('MODEL_INFO', 'bert_type')
report_evaluation = config.getboolean('GENERAL', 'report_evaluation')

if(torch.cuda.is_available()):
    device = 'cuda'
elif(torch.backends.mps.is_available()):
    device = 'mps'
else:
    device = 'cpu'
print(f'the available device is {device}')

x_train_path = config.get('DATA', 'x_train_path')
x_test_path = config.get('DATA', 'x_test_path')
y_train_path = config.get('DATA', 'y_train_path')
submit_path = config.get('GENERAL', 'submit_model_path')
best_model_path = config.get('GENERAL', 'best_model_path')

accepted_embeddings = ['fasttext', 'glove', 'bert', 'distilbert']
assert embedding_type in accepted_embeddings, f'your embedding model should be one of thease: {accepted_embeddings}'

is_bert = 'bert' in embedding_type

if(is_bert):
    dataset = my_dataset.RumourDatasetBert(tokenize_data_path=x_train_path,
                                           labels_path=y_train_path,
                                           pad_len=pad_len,
                                           have_label=True,
                                           bert_type=bert_type
                                           )
else:
    dataset = my_dataset.RumorDataset(tokenize_data_path=x_train_path,
                                    labels_path=y_train_path,
                                    pad_len=pad_len,
                                    have_label=True,
                                    embedding_type=embedding_type,
                                    )

data_size = len(dataset)
validation_size = int(validation_size * data_size)
train_size = data_size - validation_size

train_dataset, validation_dataset = random_split(dataset=dataset,
                                                 lengths=[train_size, validation_size]
                                                 )

with open('validation_dataset.pkl', 'wb') as ff:
    pickle.dump(validation_dataset, ff)

train_dataLoader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
validation_dataLoader = DataLoader(dataset=validation_dataset,
                                   batch_size = batch_size,
                                   shuffle = True)

accepted_models = ['RNN', 'GRU', 'LSTM']
assert model_type in accepted_models, f'your model_type should be one of thease: {accepted_models}'

if(is_bert):
    if(model_type == 'RNN'):
        model = my_Bert_RNN.myBertRNN(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    num_layers = n_layer,
                    bidirectional = bidirectional,
                    inner_dropout = inner_dropout,
                    dropout = dropout,
                    bert_type=bert_type
                    ).to(device)
        
    elif(model_type == 'GRU'):
        model = my_Bert_GRU.myBertGRU(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    num_layers = n_layer,
                    bidirectional = bidirectional,
                    inner_dropout = inner_dropout,
                    dropout = dropout,
                    bert_type= bert_type
                    ).to(device) 
        
    elif(model_type == 'LSTM'):
        model = my_Bert_LSTM.myBertLSTM(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    num_layers = n_layer,
                    bidirectional = bidirectional,
                    inner_dropout = inner_dropout,
                    dropout = dropout,
                    bert_type= bert_type
                    ).to(device) 
        
else:
    if(model_type == 'RNN'):
        model = my_RNN.myRNN(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    num_layers = n_layer,
                    bidirectional = bidirectional,
                    inner_dropout = inner_dropout,
                    dropout = dropout,
                    vocab = dataset.vocab,
                    ).to(device)
    elif(model_type == 'GRU'):
        model = my_GRU.myGRU(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    num_layers = n_layer,
                    bidirectional = bidirectional,
                    inner_dropout = inner_dropout,
                    dropout = dropout,
                    vocab = dataset.vocab
                    ).to(device) 
    elif(model_type == 'LSTM'):
        model = my_LSTM.myLSTM(input_size = input_size,
                    hidden_size = hidden_size,
                    output_size = output_size,
                    num_layers = n_layer,
                    bidirectional = bidirectional,
                    inner_dropout = inner_dropout,
                    dropout = dropout,
                    vocab = dataset.vocab
                    ).to(device) 
        
print(model)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = nn.BCELoss().to(device)
gc.collect()

if(not is_bert):
    if(not trainable_embedding):
        model.embedding.weight.requires_grad = False
    else:
        model.embedding.weight.requires_grad = False

if(is_bert and trainable_last_encoder):
    for param in model.bert.encoder.layer[-1].parameters():
        param.requires_grad = True

utils.count_parameters(model = model)

max_acc = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    utils.train(train_dataLoader, model, loss_fn, optimizer, device)
    acc = epoch_evaluation.evaluate(dataloader=validation_dataLoader,
                                    model=model,
                                    loss_fn=loss_fn,
                                    device=device)
    if(acc > max_acc):
        max_acc = acc
        print(f'model saved with the validation accuracy of {acc}')
        torch.save(model, best_model_path)

print("Done!")