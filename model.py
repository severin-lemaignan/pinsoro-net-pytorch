import torch
import torch.nn as nn
from torch.autograd import Variable

class PInSoRoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(PInSoRoRNN, self).__init__()

        self.device = device

        self.hidden_size = hidden_size

        self.i2h_poses = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o_poses = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined_poses = torch.cat((input, hidden), 1)
        hidden_poses = self.i2h_poses(combined_poses)
        output_poses = self.i2o_poses(combined_poses)
        output = self.softmax(output)
        return output, hidden_poses

    def initHidden(self):

        return Variable(torch.zeros(1, self.hidden_size), device=self.device)
