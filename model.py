import torch
import torch.nn as nn
import torch.nn.functional as F

class PInSoRoRNN(nn.Module):
    version="v3"

    def __init__(self, input_dim, hidden_dim, output_dim, device, num_layers=1, dropout=0):
        """
        """
        super(PInSoRoRNN, self).__init__()

        self.device=device

        self.num_layers=num_layers

        self.hidden_dim = hidden_dim

        self.lstm_poses = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=self.num_layers,
                                  batch_first=True, # the input and output tensors are provided as (batch, seq, feature)
                                  dropout=dropout
                                  )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)



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
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc_out(x))

        output = self.softmax(x)

        # return the prediction for the last datapoint in the sequence
        # ie: can we predict the state of the interaction after observing the sequence?
        return output[:,-1,:]

    def init_hidden(self, input_shape):
        """

        :param input_shape: the shape of the network's input tensor. Only the
        1st member of the shape tuple (the batch_size) is used.

        returns (h_0, c_0):
            - h_0 of shape (num_layers * num_directions, batch, hidden_size):
              tensor containing the initial hidden state for each element in
              the batch.
            - c_0 of shape (num_layers * num_directions, batch, hidden_size):
              tensor containing the initial cell state for each element in the
              batch.
        :see: https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        """
        return (torch.zeros(self.num_layers, input_shape[0], self.hidden_dim, requires_grad=True, device=self.device),
                torch.zeros(self.num_layers, input_shape[0], self.hidden_dim, requires_grad=True, device=self.device))
