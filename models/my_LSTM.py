import torch
import torch.nn as nn
from torchtext.vocab import GloVe, FastText

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, inner_dropout, dropout, vocab = None, word2vec = 'fasttext'):
        super(myLSTM, self).__init__()
        if(vocab == None):
            if(word2vec == 'fasttext'):
                vocab = FastText(language='en') if vocab == None else vocab
            else:
                vocab = GloVe(name = '6B', dim = 100)
        vocab = FastText(language='en') if vocab == None else vocab
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, num_layers = num_layers, bidirectional = bidirectional, dropout = inner_dropout )
        self.dropout = nn.Dropout(dropout)
        bidirectional = 2 if bidirectional else 1
        self.fc = nn.Linear(num_layers * hidden_size * bidirectional , output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, (hidden, _) = self.lstm(x)
        hidden = torch.cat(([h for h in hidden]), dim = 1)
        hidden = self.dropout(hidden)
        hidden = self.relu(hidden)
        out = self.fc(hidden)
        return self.sigmoid(out)
