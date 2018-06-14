import torch
import torch.nn as nn

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

        self.i2o_poses = nn.Linear(hidden_dim, output_dim)

        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden = self.init_hidden()

    def forward(self, input):
        """
        :param input: a 3D tensor - because of batch_first=True in LSTM ctor,
                                        1st dim is the batch, 
                                        2nd dim is the sequence, 
                                        3rd dim is the input dim
        """

        #import pdb;pdb.set_trace()
        lstm_out, self.hidden = self.lstm_poses(input, self.hidden)

        output_poses = self.i2o_poses(lstm_out)

        output = self.softmax(output_poses)

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
