import torch
import torch.nn as nn

class PInSoRoRNN(nn.Module):
    def __init__(self, batch_size, seq_size, hidden_dim, output_dim):
        """
        :param seq_size: length of the datapoints sequence that we feed to the LSTM unit at each step
        """
        super(PInSoRoRNN, self).__init__()

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.lstm_poses = nn.LSTM(seq_size, hidden_dim)

        self.i2o_poses = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden = self.init_hidden()

    def forward(self, input):
        """
        :param input: a 3D tensor - 1st dim is the sequence, 2nd dim is the batch, 3rd dim is the input dim
        """

        lstm_out, self.hidden = self.lstm_poses(self.hidden)

        output_poses = self.i2o_poses(lstm_out)

        output = self.softmax(output_poses)
        return output

    def initHidden(self):

        return (torch.zeros(1, self.batch_size, self.hidden_dim, requires_grad=True),
                torch.zeros(1, self.batch_size, self.hidden_dim, requires_grad=True))
