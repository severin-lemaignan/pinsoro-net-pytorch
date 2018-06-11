import logging
logging.basicConfig(level=logging.INFO)

import sys
import random
import time
import math


import torch

from data import PInSoRoDataset, train_validation_loaders, collate_minibatch
from model import *

#device = torch.device("cuda") 
device = torch.device("cpu") 


poses_input_size = 140
annotations_output_size = 28 

n_hidden = 116 # 140 + 116 = 256
n_epochs = 100000
print_every = 500
plot_every = 100
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


d = PInSoRoDataset(sys.argv[1], device=device, chunksize=1000)
train_loader, validation_loader = train_validation_loaders(d,
                                                           batch_size=10, 
                                                           num_workers=1,
                                                           shuffle=False)


rnn = PInSoRoRNN(poses_input_size, n_hidden, annotations_output_size, device=device)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(input_tensor, annotations_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i], hidden)

    loss = criterion(output, annotations_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data[0]

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



################################################################################
################################################################################
################################################################################


start = time.time()

logging.info("Starting training on %d epochs" % n_epochs)

from torch.utils.data import DataLoader
loader = DataLoader(d, batch_size=10, num_workers=1, shuffle=False, collate_fn=collate_minibatch)

for epoch in range(1, n_epochs + 1):


    for poses_tensor, annotations_tensor in loader:

        output, loss = train(poses_tensor, annotations_tensor)
        current_loss += loss

        # Print epoch number, loss, name and guess
        if epoch % print_every == 0:
            #guess, guess_i = categoryFromOutput(output)
            #correct = '✓' if guess == category else '✗ (%s)' % category
            #print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))
            logger.info('%d %d%% (%s) %.4f' % (epoch, epoch / n_epochs * 100, timeSince(start), loss))

        # Add current loss avg to list of losses
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

torch.save(rnn, 'pinsoro-rnn-classification.pt')

