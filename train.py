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

import cProfile, pstats, io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

try:
    from tensorboardX import SummaryWriter 
except ImportError:
    logging.error("Install tensorboard for pytorch with pip3 install tensorboard-pytorch")
    sys.exit(1)


from data import make_train_test_datasets, collate_minibatch
from data import TASK_ENGAGEMENT, SOCIAL_ENGAGEMENT, SOCIAL_ATTITUDE, CONSTRUCT_CLASSES
from model import PInSoRoRNN

MODELS_PATH="models"

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(model, optimizer, criterion, input_tensor, annotations_tensor, cuda=False):

    model.train()

    if cuda:
        input_tensor = input_tensor.cuda()
        annotations_tensor = annotations_tensor.cuda()

    # Pytorch accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad()
    optimizer.zero_grad()

    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden(input_tensor.shape)

    # pass minibatches of data to the RNN
    output = model(input_tensor)

    #import pdb;pdb.set_trace()

    loss = criterion(output, annotations_tensor)
    loss.backward()

    optimizer.step()
    
    acc = accuracy(output, annotations_tensor, cuda)

    return loss.item(), acc



def evaluate(model, criterion, input_tensor, annotations_tensor, cuda=False):

    # Turn on evaluation mode which disables dropout.
    model.eval()

    if cuda:
        input_tensor = input_tensor.cuda()
        annotations_tensor = annotations_tensor.cuda()

    model.hidden = model.init_hidden(input_tensor.shape)

    with torch.no_grad():
        # pass minibatches of data to the RNN
        output = model(input_tensor)

        loss = criterion(output, annotations_tensor)

        acc = accuracy(output, annotations_tensor, cuda)

    return loss, acc

def accuracy(output, target, cuda=False):
    """
    Compute the average classification accuracy

    :param output: the output of the network (dim NxC, N=batch size, C=nb of classes)
    :param target: one-hot vectors (dim NxC) of the target annotations
    """

    nb_active_classes = int(sum(target[0]).item())

    # code to efficiently convert output into one-hot vectors with the k first
    # classes selected:
    hotoutput = torch.zeros(output.shape)
    if cuda:
        hotoutput=hotoutput.cuda()

    hotoutput.scatter_(1, output.topk(nb_active_classes)[1],1.)

    # for each sample, multiply the output one-hot vector by the target one-hot vector, sum, and divide by the nb of classes.
    # result is 1 if the 2 vectors match, 0<x<1 if only some classes match, 0 if no class match
    per_sample_accuracy = (hotoutput*target).sum(dim=1) / nb_active_classes
    
    # return the average over the whole mini batch
    return sum(per_sample_accuracy)/output.shape[0]

def compute_chance(dim1, dim2, k=2, n=2000):

    def make_rand_one_hot(dim1, dim2,k=2):
         a=torch.zeros(dim1,dim2)
         tmp=torch.rand(dim1, dim2)
         a.scatter_(1, tmp.topk(k)[1],1.)
         return a

    tot=0
    for i in range(n):
        tot+=accuracy(torch.rand(dim1,dim2),make_rand_one_hot(dim1,dim2,k))
    return tot/n




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
parser.add_argument('--constructs', default="social-attitude", help='type of social constructs to train against. One of task-engagement, social-engagement, social-attitude or special keyword "all"')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--seq-size', type=int, default=300, help='length of the sequence fed to the RNN (default: 300 datapoints, ie 10s at 30FPS)')
parser.add_argument('--batch-size', type=int, default=300, metavar='N', help='batch size')
parser.add_argument('--num-workers', type=int, default=4, metavar='N', help='number of workers to load the data')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--sanity-check', action='store_true', help='if set, only load a small subset of the dataset, for quick testing')
parser.add_argument('--profile', action='store_true', help='Profile the training. Press Ctrl+C to stop.')
parser.add_argument('--resume', help='partially trained model to reuse as starting point')
parser.add_argument("datasets_root", help="path to the root of PInSoRo CSV datasets")

args = parser.parse_args()

if args.profile:
    pr = cProfile.Profile()

if args.constructs == "all":
    constructs = None
    logging.info("Training against all social construct classes")
else:
    constructs = CONSTRUCT_CLASSES[args.constructs]
    logging.info("Training against %s constructs" % args.constructs)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


batch_size = args.batch_size

dataset_test_fraction=0.2

n_workers = args.num_workers
n_hidden = 256

seq_size = args.seq_size

n_epochs = args.epochs

eval_every_iteration = 50

save_every_iteration = 1000

learning_rate = args.lr

timestamp = "{:%Y-%m-%d-%H:%M}".format(datetime.now())
model_id = "%s-%s-lr-%f-seq-size-%d" % (timestamp, args.constructs, learning_rate, seq_size)

writer = SummaryWriter('runs/%s' % model_id)


train_dataset, test_dataset = make_train_test_datasets(path=args.datasets_root,
                                                       test_fraction=dataset_test_fraction,
                                                       device=device, 
                                                       seq_size=seq_size, 
                                                       constructs_class=constructs,
                                                       sanity_check=args.sanity_check)



chance = compute_chance(batch_size,len(constructs)*2, 2)
logging.info("Chance accuracy at %.2f" % chance)

best_prec1 = 0

model = PInSoRoRNN(train_dataset.POSES_INPUT_SIZE, 
                   n_hidden, 
                   train_dataset.ANNOTATIONS_OUTPUT_SIZE, 
                   device=device)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# NLLLoss does not calculate loss on a one-hot-vector
# cf discussion: https://discuss.pytorch.org/t/feature-request-nllloss-crossentropyloss-that-accepts-one-hot-target/2724
#criterion = nn.NLLLoss()
#criterion = nn.MultiLabelSoftMarginLoss()

# https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

if args.cuda:
    model.cuda()
    criterion.cuda()

# log the graph of the network
dummy_input = torch.rand(batch_size, seq_size, train_dataset.POSES_INPUT_SIZE, requires_grad=True, device=device)
model.hidden = model.init_hidden(dummy_input.shape)
writer.add_graph(model, (dummy_input, ))



#####################################################################################
#####################################################################################
#########  DATA LOADERS


train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset,
                          sampler=train_sampler,
                          collate_fn=collate_minibatch,
                          batch_size=batch_size,
                          num_workers=n_workers)

test_sampler = RandomSampler(test_dataset)
test_loader = DataLoader(test_dataset,
                          sampler=test_sampler,
                          collate_fn=collate_minibatch,
                          batch_size=batch_size,
                          num_workers=n_workers)


test_loader_iterator = iter(test_loader)
#####################################################################################

start_epoch = 1
start_iteration = 0

current_loss = 0
current_accuracy = 0

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
        logging.info("Continuing training from epoch %d/%d, iteration %d (batch size: %d, %d iterations per epoch)" % (start_epoch, n_epochs, start_iteration, batch_size, len(train_loader)))
    else:
        logging.warning("No checkpoint found at '{}'".format(args.resume))

if start_epoch == 1 and start_iteration == 0:
    logging.info("Starting training on %d epochs (batch size: %d, %d iterations per epoch)" % (n_epochs, batch_size, len(train_loader)))


start = time.time()

prec1 = 0

iteration = start_iteration
tot_iteration = start_iteration

if args.profile:
    logging.warning("STARTING PROFILING -- press Ctrl+C to stop")
    pr.enable()

try:
    for epoch in range(start_epoch, n_epochs + 1):


        logging.info('******** EPOCH %d/%d **********' % (epoch, n_epochs))

        for poses_tensor, annotations_tensor in train_loader:

            #import pdb;pdb.set_trace()
            loss, acc = train(model, optimizer, criterion, poses_tensor, annotations_tensor, args.cuda)
            current_loss += loss
            current_accuracy += acc

            iteration += 1
            tot_iteration += 1

            # Evaluate the model on the validation dataset and record the losses
            if iteration % eval_every_iteration == 0:
                avg_loss = current_loss / eval_every_iteration
                avg_accuracy = current_accuracy / eval_every_iteration

                logging.info('iteration %d (%d%% of epoch) (%s) -- avg loss over the last %d iterations: %.4f (accuracy: %.4f)' % (iteration, iteration / len(train_loader) * 100, timeSince(start), eval_every_iteration, avg_loss, avg_accuracy))

                current_loss = 0
                current_accuracy = 0

                test_poses, test_annotations = next(test_loader_iterator)
                test_loss, test_accuracy = evaluate(model, criterion, 
                                                    test_poses, test_annotations,
                                                    args.cuda)
                writer.add_scalars('cross-entropy', {'train': avg_loss,
                                                    'eval': test_loss}, tot_iteration)
                writer.add_scalars('accuracy', {'train': avg_accuracy,
                                                'eval': test_accuracy,
                                                'chance': chance}, tot_iteration)



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
                    os.path.join(MODELS_PATH, 'pinsoronet-%s-epoch-%d-iteration-%d.pt' % (model_id, epoch, iteration)))

        iteration = 0

#####################################################################

except (Exception, KeyboardInterrupt) as e:
    logging.error(traceback.format_exc())
    logging.fatal("Exception at epoch %d, iteration %d!" % (epoch, iteration))
finally:
    if args.profile:
        logging.warning("STOPPING PROFILING")
        pr.disable()

    if epoch > 1 or iteration > 0: # only save if not a crash during first iteration (ie a bug)
        logging.info("Saving the model...")
        is_best = prec1 > best_prec1
        save_checkpoint({
            'epoch': epoch,
            'iteration': iteration,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            }, 
            False,
            os.path.join(MODELS_PATH, 'pinsoronet-%s-epoch-%d-iteration-%d.pt' % (model_id, epoch, iteration)))

    if args.profile:
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20)
        logging.warning("PROFILING RESULTS")
        print(s.getvalue())

