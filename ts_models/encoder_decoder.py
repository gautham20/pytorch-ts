import torch
import torch.nn as nn
import torch.nn.functional as F


## TODO - support multiple RNN layers in encoder

class EncoderDecoderWrapper():
    def __init__(self, encoder, decoder_cell, output_size=3, teacher_forcing=0.3, sequence_len=336, decoder_input=True):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_length = sequence_len
        self.mode = 'train'
        self.decoder_input = decoder_input

    def train(self):
        self.mode = 'train'
        self.encoder.train()
        self.decoder_cell.train()

    def eval(self):
        self.mode = 'eval'
        self.encoder.eval()
        self.decoder_cell.eval()

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'decoder_cell': self.decoder_cell.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder_cell.load_state_dict(state_dict['decoder_cell'])

    def __call__(self, xb, yb=None):
        if self.decoder_input:
            decoder_input = xb[-1]
            input_seq = xb[0]
            if len(xb) > 2:
                encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
            else:
                encoder_output, encoder_hidden = self.encoder(input_seq)
        else:
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(input_seq.size(0), self.output_size, device='cuda')
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        return outputs