# coding: utf-8

import logging
logging.basicConfig(level=logging.INFO)

import argparse

import traceback
import os.path
import sys
import random
import time
from datetime import datetime
import math


import torch
import torch.nn as nn

try:
    from tensorboardX import SummaryWriter 
except ImportError:
    logging.error("Install tensorboard for pytorch with pip3 install tensorboard-pytorch")
    sys.exit(1)


from data import PInSoRoDataset, train_validation_loaders, collate_minibatch
from data import TASK_ENGAGEMENT, SOCIAL_ENGAGEMENT, SOCIAL_ATTITUDE
from model import PInSoRoRNN

MODELS_PATH="models"


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(model, optimizer, criterion, input_tensor, annotations_tensor, cuda=False):
    hidden = PInSoRoRNN.initHidden(batch_size, n_hidden)

    if cuda:
        hidden = hidden.cuda()
        input_tensor = input_tensor.cuda()
        annotations_tensor = annotations_tensor.cuda()

    optimizer.zero_grad()

    # pass minibatches of data to the RNN
    output, hidden = model(input_tensor, hidden)

    #import pdb;pdb.set_trace()
    loss = criterion(output, annotations_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



################################################################################
################################################################################
################################################################################

parser = argparse.ArgumentParser(description='PInSoRo-net -- PyTorch implementation')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=300, metavar='N', help='batch size')
parser.add_argument('--chunk-size', type=int, default=0, metavar='N', help='chunk size (default: load the whole dataset in one go)')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--resume', help='partially trained model to reuse as starting point')
parser.add_argument("dataset", help="path to the PInSoRo CSV dataset")

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


batch_size = args.batch_size

n_workers = 8
n_hidden = 116 # 140 + 116 = 256

n_epochs = args.epochs

print_every_iteration = 50
plot_every_iteration = 50

save_every_iteration = 1000

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

timestamp = "{:%Y-%m-%d-%H:%M}".format(datetime.now())

writer = SummaryWriter('runs/%s' % timestamp)
# Keep track of losses for plotting
current_loss = 0


d = PInSoRoDataset(args.dataset, device=device, batch_size=batch_size, constructs_class=SOCIAL_ATTITUDE, chunksize=args.chunk_size)
train_loader, validation_loader = train_validation_loaders(d,
                                                           batch_size=batch_size, 
                                                           num_workers=1,
                                                           shuffle=False)


best_prec1 = 0

model = PInSoRoRNN(d.POSES_INPUT_SIZE, n_hidden, d.ANNOTATIONS_OUTPUT_SIZE)

# log the graph of the network
dummy_hidden = PInSoRoRNN.initHidden(1, n_hidden)
dummy_input = torch.rand(1, d.POSES_INPUT_SIZE, requires_grad=True)
writer.add_graph(model, (dummy_input, dummy_hidden ))


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# NLLLoss does not calculate loss on a one-hot-vector
# cf discussion: https://discuss.pytorch.org/t/feature-request-nllloss-crossentropyloss-that-accepts-one-hot-target/2724
#criterion = nn.NLLLoss()
criterion = nn.MultiLabelSoftMarginLoss()

if args.cuda:
    model.cuda()
    criterion.cuda()


start_epoch = 1
start_iteration = 0

if args.resume:
    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        logging.info("...loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("Loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        logging.info("Continuing training from epoch %d/%d, iteration %d (batch size: %d, %d iterations per epoch)" % (start_epoch, n_epochs, start_iteration, batch_size, int(len(d)/batch_size)))
    else:
        logging.warning("No checkpoint found at '{}'".format(args.resume))

if start_epoch == 1 and start_iteration == 0:
    logging.info("Starting training on %d epochs (batch size: %d, %d iterations per epoch)" % (n_epochs, batch_size, int(len(d)/batch_size)))


start = time.time()


from torch.utils.data import DataLoader
loader = DataLoader(d, batch_size=batch_size, num_workers=n_workers, shuffle=False, collate_fn=collate_minibatch)

prec1 = 0

iteration = start_iteration
tot_iteration = start_iteration

try:
    for epoch in range(start_epoch, n_epochs + 1):


        logging.info('******** EPOCH %d/%d **********' % (epoch, n_epochs))

        for poses_tensor, annotations_tensor in loader:

            #import pdb;pdb.set_trace()
            output, loss = train(model, optimizer, criterion, poses_tensor, annotations_tensor, args.cuda)
            current_loss += loss

            iteration += 1
            tot_iteration += 1

            # Print epoch number, loss, name and guess
            if iteration % print_every_iteration == 0:
                #guess, guess_i = categoryFromOutput(output)
                #correct = '✓' if guess == category else '✗ (%s)' % category
                #print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))
                logging.info('iteration %d (%d%% of epoch) (%s) %.4f' % (iteration, iteration*batch_size / len(d) * 100, timeSince(start), loss))

            # Add current loss avg to list of losses
            if iteration % plot_every_iteration == 0:
                avg_loss = current_loss / plot_every_iteration
                writer.add_scalar('loss', avg_loss, tot_iteration)
                current_loss = 0


            if iteration % save_every_iteration == 0:
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch,
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                    }, 
                    is_best,
                    os.path.join(MODELS_PATH, 'pinsoronet-%s-epoch-%d-iteration-%d.pt' % (timestamp, epoch, iteration)))

        iteration = 0

#####################################################################

except (Exception, KeyboardInterrupt) as e:
    logging.error(traceback.format_exc())
    logging.fatal("Exception! Saving the model...")
finally:
    if epoch > 1 or iteration > 0: # only save if not a crash during first iteration (ie a bug)
        is_best = prec1 > best_prec1
        save_checkpoint({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, 
            False,
            os.path.join(MODELS_PATH, 'pinsoronet-%s-epoch-%d-iteration-%d.pt' % (timestamp, epoch, iteration)))


