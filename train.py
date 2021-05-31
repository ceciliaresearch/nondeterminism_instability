"""Trains a model."""

import argparse
import csv
import math
import os
import random
import struct

import numpy as np
import torch
import torch.nn as nn

import data
import models
import utils

parser = argparse.ArgumentParser()
# General training args
parser.add_argument('--dataset', default='cifar10', type=str,
    help='Dataset name')
parser.add_argument('--model_type', default='resnet14', type=str,
    help='Model name, from models.py')
parser.add_argument('--model_dir', default='tmp_model', type=str,
    help='Directory to save model and logs in')
parser.add_argument('--resume_from', default='', type=str,
    help='Path to saved model to initialize all state from')
# Optimizer args
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--num_epochs', default=500, type=int,
    help='Total number of epochs to train for')
parser.add_argument('--max_lr', default=.40, type=float,
    help='Max learning rate')
parser.add_argument('--lr_schedule', default='cosine', type=str,
    help='Name of learning rate schedule')
parser.add_argument('--warmup_epochs', default=3., type=float,
    help='Number of epochs for learning rate warmup')
# Seeds
parser.add_argument('--data_aug_seed', default=1, type=int,
    help='Seed for data augmentation')
parser.add_argument('--cudnn_seed', default=1, type=int,
    help='Seed for cuDNN (1=deterministic)')
parser.add_argument('--shuffle_train_seed', default=1, type=int,
    help='Seed for training data shuffling')
parser.add_argument('--init_seed', default=1, type=int,
    help='Seed for weight initialization')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--random_bit_change', default=0, type=int,
    help='Type of random bit change to do (see code), 0=no change.')
parser.add_argument('--random_bit_change_seed', default=0, type=int,
    help='Seed for random bit changes')

args = parser.parse_args()
args.dataset = args.dataset.lower()

################################################################################
# Function definitions.
################################################################################

def random_bit_change(net, weight_name, seed):
  """Changes a random parameter in the given layer by a single bit."""
  old_rng_state = torch.get_rng_state()
  torch.manual_seed(seed)
  # Pick a random parameter and change it by the smallest amount possible.
  found_name = False
  for name, param in sorted(net.named_parameters()):
    if name != weight_name:
      continue
    found_name = True
    with torch.no_grad():
      old_param = param.data.clone()
      # Pick a random index.
      view = param.data.view(-1)
      print('shape: %s' % (view.shape[0],))
      index = torch.randint(0, view.shape[0], (1,)).item()
      print('index: %d' % index)
      oldval = view[index].cpu().clone()
      if torch.rand(1).item() > .5:
        newval = np.nextafter(oldval, oldval + 1)
      else:
        newval = np.nextafter(oldval, oldval - 1)
      def binary(num):
        return ''.join(bin(c).replace('0b', '').rjust(8, '0')
                       for c in struct.pack('!f', num))
      diff = newval - oldval
      view[index] = newval
      print('oldval: %.32f' % oldval)
      print('oldval: %s' % binary(oldval))
      print('newval: %.32f' % view[index].cpu())
      print('newval: %s' % binary(view[index].cpu()))
      assert not torch.all(torch.eq(param.data, old_param))
    break
  if not found_name:
    raise ValueError('Couldn\'t find weight with name %s' % weight_name)
  torch.set_rng_state(old_rng_state)


def train(epoch):
  """Trains for an epoch."""
  print('\nEpoch: %d' % epoch)
  net.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    if use_cuda:
      inputs, targets = inputs.cuda(), targets.cuda()

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = cel(outputs, targets)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    train_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).float().cpu().sum()
    acc = 100.*float(correct)/float(total)

    utils.progress_bar(batch_idx, len(trainloader),
        'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), acc,
          correct, total))

  return (train_loss/batch_idx, acc)


def test(epoch):
  """Tests at the given epoch."""
  net.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
      if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
      outputs = net(inputs)
      loss = cel(outputs, targets)

      test_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += predicted.eq(targets.data).cpu().sum()

      utils.progress_bar(batch_idx, len(testloader),
          'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
          (test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct,
           total))

    # Save checkpoint.
    acc = 100.*float(correct)/float(total)

    if args.lr_schedule == 'snapshot5':
      cycle_length = args.num_epochs // 5
      if epoch % cycle_length == (cycle_length - 1):
        lr = max(lr_scheduler.get_lr())
        print(f'Snapshot, learning rate is {lr:g}')
        assert lr <= 1e-7  # Should be at the end of a snapshot cycle.
        checkpoint(model_dir, acc, epoch,
            checkpoint_filename=f'model-{epoch:05d}.ckpt')
    if epoch >= args.num_epochs - 1:
      checkpoint(model_dir, acc, epoch)
  return (test_loss/batch_idx, acc)


def checkpoint(model_dir, acc, epoch, checkpoint_filename='model.ckpt'):
  """Saves a model checkpoint."""
  print('Saving..')
  state = {
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    'acc': acc,
    'epoch': epoch,
    'rng_state': torch.get_rng_state(),
    'train_shuffle_state': trainloader.sampler.state,
  }
  torch.save(state, os.path.join(model_dir, checkpoint_filename))


def learning_rate_fn(step):
  """Learning rate function, given a global step."""
  steps_per_epoch = int(math.ceil(len(trainloader)))
  epoch = step / float(steps_per_epoch)

  if args.lr_schedule == 'cosine':
    if epoch <= args.warmup_epochs:
      # Linear warmup.
      lr = step / (args.warmup_epochs * steps_per_epoch) * args.max_lr
    else:
      # Cosine decay
      lr_min = 0.
      lr = lr_min + .5 * (args.max_lr - lr_min) * (
          1. + math.cos(min(epoch / args.num_epochs, 1.) * math.pi))
    return lr
  elif args.lr_schedule == 'snapshot5':
    if args.num_epochs % 5:
      raise ValueError(
          'Number of epochs for snapshot5 learning rate schedule must be '
          'divisible by 5.')
    cycle_length = args.num_epochs // 5
    start_epochs = range(0, args.num_epochs, cycle_length)
    for start_epoch in start_epochs:
      if epoch < start_epoch + cycle_length:
        lr_min = 0.
        if epoch <= args.warmup_epochs + start_epoch:
          # Linear warmup.
          lr = (epoch - start_epoch) / args.warmup_epochs * args.max_lr
        else:
          # Cosine decay
          lr = lr_min + .5 * (args.max_lr - lr_min) * (1. +
              math.cos(min((epoch - start_epoch) / cycle_length, 1.) * math.pi))
        return min(max(lr, 0.), args.max_lr)
    return 0.
  else:
    raise ValueError(f'Unsupported learning rate: {args.lr_schedule}')


def resume(checkpoint_path, model):
  """Restores network and rng state from a given checkpoint path."""
  print(f'Resuming from {checkpoint_path}')
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
  epoch = checkpoint['epoch']
  # Restore rng state.
  rng_state = checkpoint['rng_state']
  torch.set_rng_state(rng_state)
  trainloader.sampler.state = checkpoint['train_shuffle_state']
  return epoch


################################################################################
# Main script code.
################################################################################

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
if use_cuda and torch.cuda.device_count() != 1:
  raise NotImplementedError('Only 1 gpu supported')


# Set seeds.
# Only data augmentation is controlled by this seed.
torch.manual_seed(args.data_aug_seed)  

# For the experiments we do (e.g. nothing with dropout or nondeterministic
# layers), these seeds don't matter. Set them to 0 arbitrarily.
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

torch.backends.cudnn.deterministic = bool(args.cudnn_seed == 1)
torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic


# Set up data loaders.
trainloader = data.get_trainloader(args.dataset, args.batch_size,
    args.shuffle_train_seed)
testloader = data.get_testloader(args.dataset, args.batch_size)

# Make the network.
net = getattr(models, args.model_type)(args)
if use_cuda:
  net.cuda()
  print('Using', torch.cuda.device_count(), 'GPUs.')

# Optionally resume from a checkpoint.
start_epoch = 0  # Start from epoch 0 or last checkpoint epoch
if args.resume_from:
  epoch = resume(args.resume_from, net)
  start_epoch = epoch + 1

# Do a random bit change, only after checkpoint loading.
if args.random_bit_change > 0:
  weight_name = {1: 'conv1.weight',
                 2: 'fc.weight',
                 3: 'fc.bias',
                 4: 'fc1.weight',
                 5: 'features.0.weight'}[args.random_bit_change]
  random_bit_change(net, weight_name, args.random_bit_change_seed)

model_dir = os.path.join(args.dataset + '_models', args.model_dir)
if os.path.exists(model_dir):
  print('model_dir already exists: %s' % model_dir)
  print('Exiting...')
  exit()
os.makedirs(model_dir)
print('model_dir: %s' % model_dir)

# Set up the optimizer.
parameters_bias = sorted([p[0] for p in net.named_parameters()
                          if 'bias' in p[0]])
parameters_bnscale = sorted([p[0] for p in net.named_parameters()
                             if 'bn' in p[0] and 'weight' in p[0]])
parameters_others = sorted([p[0] for p in net.named_parameters()
                            if p[0] not in parameters_bias and
                               p[0] not in parameters_bnscale])

def tensor_params(name_list):
  names = set(name_list)
  return [p[1] for p in sorted(net.named_parameters()) if p[0] in name_list]

optimizer = torch.optim.SGD(
    [{'params': tensor_params(parameters_bias), 'lr': .1},
    {'params': tensor_params(parameters_bnscale), 'lr': .1},
    {'params': tensor_params(parameters_others)}], 
    lr=1.,  # This is set by the scheduler.
    momentum=args.momentum,
    weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lr_lambda=learning_rate_fn)

cel = nn.CrossEntropyLoss()

logname = os.path.join(model_dir, 'log.txt')
if not os.path.exists(logname):
  with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss',
                        'test acc'])

# Main training loop.
for epoch in range(start_epoch, args.num_epochs):
  train_loss, train_acc = train(epoch)
  test_loss, test_acc = test(epoch)
  with open(logname, 'a') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    lr = lr_scheduler.get_lr()
    if isinstance(lr, float):
      lr_str = '%g' % lr
    elif isinstance(lr, list):
      lr_str = '[' + ', '.join(['%g' % x for x in lr]) + ']'
    logwriter.writerow([epoch, lr_str, train_loss, train_acc, test_loss,
                        test_acc])
  if np.isnan(train_loss):
    print('Detected NaN train loss, terminating early...')
    exit()
