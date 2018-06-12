import logging
logging.basicConfig(level=logging.INFO)

import traceback
import os.path
import sys
import random
import time
from datetime import datetime
import math


import torch
import torch.nn as nn

from data import PInSoRoDataset, train_validation_loaders, collate_minibatch
from data import TASK_ENGAGEMENT, SOCIAL_ENGAGEMENT, SOCIAL_ATTITUDE
from model import PInSoRoRNN

MODELS_PATH="models"

device = torch.device("cuda") 
#device = torch.device("cpu") 


batch_size=300

n_hidden = 116 # 140 + 116 = 256

n_epochs = 10

print_every_iteration = 50
plot_every_iteration = 50

save_every_iteration = 1000

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


d = PInSoRoDataset(sys.argv[1], device=device, constructs_class=SOCIAL_ATTITUDE, chunksize=1000)
train_loader, validation_loader = train_validation_loaders(d,
                                                           batch_size=batch_size, 
                                                           num_workers=1,
                                                           shuffle=False)


rnn = PInSoRoRNN(d.POSES_INPUT_SIZE, n_hidden, d.ANNOTATIONS_OUTPUT_SIZE)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


# NLLLoss does not calculate loss on a one-hot-vector
# cf discussion: https://discuss.pytorch.org/t/feature-request-nllloss-crossentropyloss-that-accepts-one-hot-target/2724
#criterion = nn.NLLLoss()
criterion = nn.MultiLabelSoftMarginLoss()

def train(input_tensor, annotations_tensor):
    hidden = PInSoRoRNN.initHidden(batch_size, n_hidden, device)
    optimizer.zero_grad()

    # pass minibatches of data to the RNN
    output, hidden = rnn(input_tensor, hidden)

    #import pdb;pdb.set_trace()
    loss = criterion(output, annotations_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()

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

timestamp = "{:%Y-%m-%d-%H:%M}".format(datetime.now())
start = time.time()

logging.info("Starting training on %d epochs (batch size: %d, %d iterations per epoch)" % (n_epochs, batch_size, int(len(d)/batch_size)))

from torch.utils.data import DataLoader
loader = DataLoader(d, batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_minibatch)

epoch = 1
iteration = 0

try:
    for epoch in range(1, n_epochs + 1):

        iteration = 0

        logging.info('******** EPOCH %d/%d **********' % (epoch, n_epochs))

        for poses_tensor, annotations_tensor in loader:

            output, loss = train(poses_tensor, annotations_tensor)
            current_loss += loss

            iteration += 1

            # Print epoch number, loss, name and guess
            if iteration % print_every_iteration == 0:
                #guess, guess_i = categoryFromOutput(output)
                #correct = '✓' if guess == category else '✗ (%s)' % category
                #print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))
                logging.info('iteration %d (%d%% of epoch) (%s) %.4f' % (iteration, iteration*batch_size / len(d) * 100, timeSince(start), loss))

            # Add current loss avg to list of losses
            if iteration % plot_every_iteration == 0:
                all_losses.append(current_loss / plot_every_iteration)
                current_loss = 0

            if iteration % save_every_iteration == 0:
                torch.save(rnn, os.path.join(MODELS_PATH, 'pinsoronet-%s-epoch-%d-iteration-%d.pt' % (timestamp, epoch, iteration)))

    torch.save(rnn, os.path.join(MODELS_PATH, 'pinsoronet-%s-epoch-%d-iteration-%d.pt' % (timestamp, epoch, iteration)))

except Exception as e:
    logging.error(traceback.format_exc())
    logging.fatal("Exception! Saving the model to %s/pinsoronet-<...>-INTERRUPTED-<...>.pt" % MODELS_PATH)
    torch.save(rnn, os.path.join(MODELS_PATH, 'pinsoronet-%s-INTERRUPTED-epoch-%d-iteration-%d.pt' % (timestamp, epoch, iteration)))

