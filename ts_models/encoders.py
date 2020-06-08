import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNInitEncoder(nn.Module):
    def __init__(self, embed_sizes, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False, device='cpu'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.embeds = nn.ModuleList([nn.Embedding(num_classes, output_size) for num_classes, output_size in embed_sizes])
        self.embed_to_ht = nn.Linear(sum([s[1] for s in embed_sizes]), self.hidden_size)
        self.gru = nn.GRU(
            num_layers = rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.device = device

    def forward(self, input_seq, input_cat):
        embeds = [e(input_cat[:, i]) for i, e in enumerate(self.embeds)]
        embeds = torch.cat(embeds, 1)
        ht = self.embed_to_ht(embeds)
        ht.unsqueeze_(0)
        if (self.num_layers * self.rnn_directions) > 1:
            ht = ht.repeat(self.rnn_directions * self.num_layers, 1, 1)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        return gru_out, hidden.squeeze(0)


class RNNConcatEncoder(nn.Module):
    def __init__(self, embed_sizes, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False, device='cpu'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.embeds = nn.ModuleList([nn.Embedding(num_classes, output_size) for num_classes, output_size in embed_sizes])
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers = rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.output_linear = nn.Linear(hidden_size + sum([s[1] for s in embed_sizes]), hidden_size)
        self.device = device

    def forward(self, input_seq, input_cat):
        embeds = [e(input_cat[:, i]) for i, e in enumerate(self.embeds)]
        embeds = torch.cat(embeds, 1)
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0) , self.hidden_size, device=self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        if self.rnn_directions > 1:
            gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            gru_out = torch.sum(gru_out, axis=2)
        encoder_concat_hidden = self.output_linear(torch.cat((hidden.squeeze(0), embeds), axis=1))
        return gru_out, encoder_concat_hidden


#output shape
# bidirectional output is summed
# gru_out - (batch, sequence_len, hidden_size)
# hidden - (batch, hidden_size) only the last layer for multi-layer
class RNNEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False, device='cpu', rnn_dropout=0.2):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout
        )
        self.device = device

    def forward(self, input_seq):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device=self.device)
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        gru_out, hidden = self.gru(input_seq, ht)
        print(gru_out.shape)
        print(hidden.shape)
        if self.rnn_directions * self.num_layers > 1:
            num_layers = self.rnn_directions * self.num_layers
            if self.rnn_directions > 1:
                gru_out = gru_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
                gru_out = torch.sum(gru_out, axis=2)
            hidden = hidden.view(self.num_layers, self.rnn_directions, input_seq.size(0), self.hidden_size)
            if self.num_layers > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        else:
            hidden.squeeze_(0)
        return gru_out, hidden
