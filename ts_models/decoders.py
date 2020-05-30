import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, positive_pred, dropout=0.2):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=input_feature_len,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        self.attention = False
        self.positive_pred = positive_pred
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_hidden, y):
        rnn_hidden = self.decoder_rnn_cell(y, prev_hidden)
        output = self.out(rnn_hidden)
        if self.positive_pred:
            output = F.relu(output)
        return output, self.dropout(rnn_hidden)

    
class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, hidden_size, sequence_len):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, 1)
        
    def forward(self, encoder_output, prev_hidden, y):
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input)).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden = self.decoder_rnn_cell(attention_combine, prev_hidden)
        output = F.relu(self.out(rnn_hidden))
        return output, rnn_hidden
