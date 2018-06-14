import torch
import torch.nn as nn
from torch.autograd import Variable

class PInSoRoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PInSoRoRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h_poses = nn.Linear(input_size + hidden_size, hidden_size)

        self.i2o_poses = nn.Linear(input_size + hidden_size, hidden_size)

        self.fc1_poses = nn.Linear(hidden_size, hidden_size)
        self.fc2_poses = nn.Linear(hidden_size, hidden_size)
        self.fc3_poses = nn.Linear(hidden_size, hidden_size)
        self.fc4_poses = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #import pdb;pdb.set_trace()

        combined_poses = torch.cat((input, hidden), 1)
        hidden_poses = self.i2h_poses(combined_poses)
        output_poses0 = self.i2o_poses(combined_poses)

        output_poses1 = self.fc1_poses(output_poses0)
        output_poses2 = self.fc2_poses(output_poses1)
        output_poses3 = self.fc3_poses(output_poses2)

        output_poses = self.fc4_poses(output_poses3)

        output = self.softmax(output_poses)
        return output, hidden_poses

    @staticmethod
    def initHidden(batch_size, hidden_size):

        return torch.zeros(batch_size, hidden_size, requires_grad=True)
