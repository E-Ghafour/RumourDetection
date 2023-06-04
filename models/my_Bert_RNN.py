import torch.nn as nn
from transformers import BertModel
import torch


class myBertRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, inner_dropout, dropout, bert_type = 'bert-base-uncased'):
        super(myBertRNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_type)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True, num_layers = num_layers, bidirectional = bidirectional, dropout = inner_dropout )
        self.dropout = nn.Dropout(dropout)
        bidirectional = 2 if bidirectional else 1
        self.fc = nn.Linear(num_layers * hidden_size * bidirectional , output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.bert(input_ids = x[0], attention_mask = x[1])
        x = self.dropout(x[0])
        out, hidden = self.rnn(x)
        hidden = torch.cat(([h for h in hidden]), dim = 1)
        hidden = self.dropout(hidden)
        hidden = self.relu(hidden)
        out = self.fc(hidden)
        return self.sigmoid(out)