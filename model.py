import torch
import torch.nn as nn
import torch.nn.functional as F

class PInSoRoRNN(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, output_dim, device, num_layers=1):
        """
        """
        super(PInSoRoRNN, self).__init__()

        self.device=device

        self.num_layers=1
        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.lstm_poses = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=1,
                                  batch_first=True, # the input and output tensors are provided as (batch, seq, feature)
                                  dropout=0
                                  )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden = self.init_hidden()

    def forward(self, x):
        """
        :param input: a 3D tensor - because of batch_first=True in LSTM ctor,
                                        1st dim is the batch, 
                                        2nd dim is the sequence, 
                                        3rd dim is the input dim
        """

        x, self.hidden = self.lstm_poses(x, self.hidden)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        output = self.softmax(x)

        # return the prediction for the last datapoint in the sequence
        # ie: can we predict the state of the interaction after observing the sequence?
        return output[:,-1,:]

    def init_hidden(self):
        """
        returns (h_0, c_0):
            - h_0 of shape (num_layers * num_directions, batch, hidden_size):
              tensor containing the initial hidden state for each element in
              the batch.
            - c_0 of shape (num_layers * num_directions, batch, hidden_size):
              tensor containing the initial cell state for each element in the
              batch.
        :see: https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, requires_grad=True, device=self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, requires_grad=True, device=self.device))
