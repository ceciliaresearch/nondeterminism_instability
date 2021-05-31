"""Analyzes model variability for image classification."""

import os
import random

import numpy as np
import torch

import data
import models

# Setup
use_cuda = torch.cuda.is_available()
if not use_cuda:
  raise NotImplementedError("evaluate.py requires a GPU to use.")
device = torch.device('cuda')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


_cached_testloaders = {}  # Maps (dataset_name, batch_size) to testloader
def get_testloader(dataset_name, batch_size=128):
  key = (dataset_name, batch_size)
  if key in _cached_testloaders:
    return _cached_testloaders[key]
  else:
    testloader = data.get_testloader(dataset_name, batch_size=batch_size)
    _cached_testloaders[key] = testloader
    return testloader


def aug_rep_factor(tta_type):
  """Returns the number of augmented images per input image"""
  return {
      'flip': 1,
      'crop25_cifar': 25,  # padding 4, stride 2 -> 5x5
      'crop81_cifar': 81,  # padding 4 -> (2*4+1) * (2*4+1)
      'flip_crop25_cifar': 50,
      'flip_crop81_cifar': 162,
      'crop_mnist': 9,  # padding 1 -> (2*1+1) * (2*1+1)
      'crop_imagenet': 9,
      'flip_crop_imagenet': 18,
  }[tta_type]


def hflip(images):
  return torch.flip(images, dims=[3])


def crop_cifar(images, crop_stride=1):
  """images: [N, C, H, W]"""
  # Make the padded version
  padding = 4
  n, c, h, w = images.shape
  # Fillling with zero then normalizing with
  # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  # ...produces these values. e.g. (0 - 0.4914) / 0.2023
  fill_val = torch.tensor([-2.4290657439446366, -2.418254764292879,
      -2.2213930348258706]).cuda()
  padded_images = fill_val.reshape([1, 3, 1, 1]).repeat(
      [n, 1, h + 2 * padding, w + 2 * padding])
  padded_images[:, :, padding:padding+h, padding:padding+w] = images
  cropped_images = []
  # Go through each possible crop.
  for row in range(0, 2 * padding + 1, crop_stride):
    for col in range(0, 2 * padding + 1, crop_stride):
      cropped_images.append(padded_images[:, :, row:row+h, col:col+w])
  cropped_images = torch.cat(cropped_images, dim=0)
  return cropped_images


def flip_crop_cifar(images, crop_stride=1):
  """Do both flipping and cropping."""
  images = torch.cat([images, hflip(images)], dim=0)
  return crop_cifar(images, crop_stride=crop_stride)


def crop_mnist(images):
  """images: [N, C, H, W]"""
  # Make the padded version
  padding = 1
  n, c, h, w = images.shape
  # Fillling with zero then normalizing with
  # transforms.Normalize((0.1307,), (0.3081,)),
  # ...produces these values.
  fill_val = torch.tensor([-0.424212917883804]).cuda()
  padded_images = fill_val.reshape([1, 1, 1, 1]).repeat(
      [n, 1, h + 2 * padding, w + 2 * padding])
  padded_images[:, :, padding:padding+h, padding:padding+w] = images
  cropped_images = []
  # Go through each possible crop.
  for row in range(2 * padding + 1):
    for col in range(2 * padding + 1):
      cropped_images.append(padded_images[:, :, row:row+h, col:col+w])
  cropped_images = torch.cat(cropped_images, dim=0)
  return cropped_images


def crop_imagenet(images):
  """images: [N, C, H, W]"""
  # Extract 9 224x224 images (left/middle/right x top/middle/bottom).
  cropped_images = []
  for row in [0, 16, 32]:
    for col in [0, 16, 32]:
      cropped_images.append(images[:, :, row:row+224, col:col+224])
  cropped_images = torch.cat(cropped_images, dim=0)
  return cropped_images


def flip_crop_imagenet(images):
  images = torch.cat([images, hflip(images)], dim=0)
  return crop_imagenet(images)


def apply_tta(images, tta_type='', tta_strategy=''):
  """images: [N, C, H, W]"""

  if tta_strategy in ['', 'ignore']:
    return images

  # Apply the augmentation.
  if tta_type == 'flip':
    aug_images = hflip(images)
  elif tta_type == 'crop81_cifar':
    aug_images = crop_cifar(images, crop_stride=1)
  elif tta_type == 'crop25_cifar':
    aug_images = crop_cifar(images, crop_stride=2)
  elif tta_type == 'flip_crop81_cifar':
    aug_images = flip_crop_cifar(images, crop_stride=1)
  elif tta_type == 'flip_crop25_cifar':
    aug_images = flip_crop_cifar(images, crop_stride=2)
  elif tta_type == 'crop_mnist':
    aug_images = crop_mnist(images)
  elif tta_type == 'crop_imagenet':
    aug_images = crop_imagenet(images)
  elif tta_type == 'flip_crop_imagenet':
    aug_images = flip_crop_imagenet(images)
  else:
    assert False

  # Figure out which images to return.
  if tta_strategy == 'ignore':
    return images
  elif tta_strategy == 'overwrite':
    # Assumes aug_images is replicated in the same order as images.
    # Interleave in batch dimension: [1, 1, 2, 2, 3, 3, 4, 4, ...]
    images_tuple = torch.chunk(aug_images, aug_rep_factor(tta_type), dim=0)
    return torch.flatten(torch.stack(images_tuple, dim=1),
                         start_dim=0, end_dim=1)
  elif tta_strategy == 'moredata':
    # Assumes aug_images is replicated in the same order as images.
    # Interleave in batch dimension: [1, 1, 2, 2, 3, 3, 4, 4, ...]
    images_tuple = torch.chunk(torch.cat((images, aug_images), dim=0),
                               1+aug_rep_factor(tta_type), dim=0)
    return torch.flatten(torch.stack(images_tuple, dim=1),
                         start_dim=0, end_dim=1)
  elif tta_strategy == 'tta':
    # Assumes aug_images is replicated in the same order as images.
    # Do *not* interleave: [1, 2, 3, 4, 1, 2, 3, 4, ...], since logits will be
    # aggregated later and this ordering is easier to make that work for.
    return torch.cat((images, aug_images), dim=0)
  elif tta_strategy == 'tta_noorig':
    # Assumes aug_images is replicated in the same order as images.
    # Do *not* interleave: [1, 2, 3, 4, 1, 2, 3, 4, ...], since logits will be
    # aggregated later and this ordering is easier to make that work for.
    return aug_images
  else:
    assert False


def tta_agg_logits(logits, tta_type, tta_strategy):
  """Aggregates logits for the given tta_strategy."""
  if tta_strategy in ['', 'ignore']:
    return logits
  elif tta_strategy == 'overwrite':
    return logits
  elif tta_strategy == 'moredata':
    return logits
  elif tta_strategy == 'tta':
    # Aggregate logits
    logits_tuple = torch.chunk(logits, 1+aug_rep_factor(tta_type), dim=0)
    return torch.mean(torch.stack(logits_tuple, dim=0), dim=0)
  elif tta_strategy == 'tta_noorig':
    logits_tuple = torch.chunk(logits, aug_rep_factor(tta_type), dim=0)
    return torch.mean(torch.stack(logits_tuple, dim=0), dim=0)


_cached_testlabels = {}  # Maps (dataset_name,) to test labels
def get_testlabels(dataset_name, tta_type, tta_strategy, batch_size=128):
  """Gets test labels for the given dataset."""
  key = (dataset_name,)
  if key not in _cached_testlabels:
    testloader = get_testloader(dataset_name, batch_size=batch_size)
    test_labels = []
    for _, targets in testloader:
      test_labels.append(targets.numpy())
    noaug_testlabels = np.concatenate(test_labels)
    _cached_testlabels[key] = noaug_testlabels
  noaug_testlabels = _cached_testlabels[key]

  # Potentially replicate based on tta_strategy.
  if tta_strategy in ['', 'ignore']:
    return noaug_testlabels
  elif tta_strategy == 'overwrite':
    return np.repeat(noaug_testlabels, aug_rep_factor(tta_type), axis=0)
  elif tta_strategy == 'moredata':
    return np.repeat(noaug_testlabels, 1+aug_rep_factor(tta_type), axis=0)
  elif tta_strategy == 'tta':
    return noaug_testlabels
  elif tta_strategy == 'tta_noorig':
    return noaug_testlabels
  else:
    assert False


def predict(checkpoint_paths, model_type, dataset_name='cifar10', tta_type='',
    tta_strategy='', batch_size=128):
  """Returns logits from ensembling the given checkpoint_paths together."""
  for checkpoint_path in checkpoint_paths:
    if not os.path.exists(checkpoint_path):
      return []
  ensemble_preds = []
  for checkpoint_path in checkpoint_paths:
    with torch.no_grad():
      net = getattr(models, model_type)()
      checkpoint = torch.load(checkpoint_path)
      if 'net' in checkpoint:
        net = checkpoint['net']
      elif 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
      else:
        raise ValueError(f'unable to load model from checkpoint with keys: '
                         f'{list(checkpoint.keys())}')
      net.cuda()
      net.eval()
      model_outputs = []
      testloader = get_testloader(dataset_name, batch_size=batch_size)
      for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = apply_tta(inputs, tta_type, tta_strategy)
        outputs = net(inputs)  # Logits
        outputs = tta_agg_logits(outputs, tta_type, tta_strategy)
        model_outputs.append(outputs.detach().cpu().numpy())
      model_outputs = np.concatenate(model_outputs)
      ensemble_preds.append(model_outputs)
  if len(ensemble_preds) == 1:
    return ensemble_preds[0]
  else:
    return np.mean(np.stack(ensemble_preds, axis=0), axis=0)


def _get_ranks(x):
  tmp = x.argsort()
  ranks = torch.zeros_like(tmp).cuda()
  ranks[tmp] = torch.arange(len(x)).cuda()
  return ranks


def spearman_gpu(x, y):
  """Computes the Spearman correlation between 2 1-D vectors.

  Args:
    x: Shape (N, )
    y: Shape (N, )
  """
  x_rank = _get_ranks(x)
  y_rank = _get_ranks(y)
  
  n = x.size(0)
  upper = 6 * torch.sum((x_rank - y_rank).type(torch.cuda.FloatTensor).pow(2))
  down = n * (n ** 2 - 1.0)
  return 1.0 - (upper / down)


def pearson_gpu(x, y):
  """Computes the Pearson correlation between 2 1-D vectors.

  Args:
    x: Shape (N, )
    y: Shape (N, )
  """
  vx = x - torch.mean(x)
  vy = y - torch.mean(y)
  cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) *
                               torch.sqrt(torch.sum(vy ** 2)))
  return cost


def compute_metrics(all_logits, test_labels, verbose=False):
  """Computes and prints all variability metrics.

  Args:
    all_logits: List of np.array. Logits for each model.
    test_labels: np.array of test labels.
    verbose: Print progress as we go along.
  """
  # Get baseline accs/CEs.
  accs = []
  ces = []
  cel = torch.nn.CrossEntropyLoss()
  for logits in all_logits:
    preds = logits.argmax(axis=1)
    accs.append(100 * np.mean(preds == test_labels))
    ces.append(cel(torch.tensor(logits), torch.tensor(test_labels)).item())

  # Mean/std stats.
  bootstrap_n = 10000
  np.random.seed(1)
  for (stat_name, stat_list) in [('Test Cross-Entropy', ces),
                                 ('Test Accuracy', accs)]:
    mean = np.mean(stat_list)
    std = np.std(stat_list)
    # Bootstrap the std
    bs_vals = np.random.choice(stat_list, size=(bootstrap_n, len(stat_list)),
                               replace=True)
    bs_std = np.std(bs_vals, axis=1)
    assert len(bs_std) == bootstrap_n
    bs_std = np.std(bs_std)
    print(f'{stat_name}: {mean:g} +- {std:g} (+- {bs_std:g})')

  # Put everything on the GPU once at the beginning if the size is small enough.
  if len(all_logits) > 0:
    bytes_used = (4 * len(all_logits) *
                  all_logits[0].shape[0] * all_logits[0].shape[1])
  else:
    bytes_used = 0
  already_on_gpu = False
  if bytes_used < 3e9:
    for i in range(len(all_logits)):
      all_logits[i] = torch.tensor(all_logits[i]).cuda()
    already_on_gpu = True

  # Compute pairwise metrics.
  acc_deltas = []
  ce_deltas = []
  spearmans = []
  pearsons = []
  disagreements = []
  labels_tensor = torch.tensor(test_labels).cuda()
  for i1 in range(len(all_logits)):
    if already_on_gpu:
      logits1 = all_logits[i1]
    else:
      logits1 = torch.tensor(all_logits[i1]).cuda()
    flat_logits1 = logits1.flatten()
    for i2 in range(i1+1, len(all_logits)):
      if verbose:
        print(f'Pairwise {i1} {i2}')
      if already_on_gpu:
        logits2 = all_logits[i2]
      else:
        logits2 = torch.tensor(all_logits[i2]).cuda()
      ensemble_preds = .5 * (logits1 + logits2)
      acc = 100 * torch.mean(torch.eq(torch.argmax(ensemble_preds, dim=1),
          labels_tensor).type(torch.cuda.FloatTensor)).item()
      ce = cel(ensemble_preds, labels_tensor).item()
      acc_deltas.append(acc - accs[i1])
      acc_deltas.append(acc - accs[i2])
      ce_deltas.append(ce - ces[i1])
      ce_deltas.append(ce - ces[i2])
      preds1 = torch.argmax(logits1, dim=1)
      preds2 = torch.argmax(logits2, dim=1)
      disagreement = 100 * torch.mean(
          (preds1 != preds2).type(torch.cuda.FloatTensor)).item()
      disagreements.append(disagreement)
      flat_logits2 = logits2.flatten()
      rho = spearman_gpu(flat_logits1, flat_logits2).item()
      r = pearson_gpu(flat_logits1, flat_logits2).item()
      spearmans.append(rho)
      pearsons.append(r)
  print('Average pairwise accuracy delta (%%): %g' % np.mean(acc_deltas))
  print('Average pairwise cross-entropy delta: %g' % np.mean(ce_deltas))
  print('Average pairwise correlation (spearman): %g' % np.mean(spearmans))
  print('Average pairwise correlation (pearson): %g' % np.mean(pearsons))
  print('Average pairwise disagreement (%%): %g' % np.mean(disagreements))


def run_evaluation(checkpoint_paths, model_type='',
    dataset_name='cifar10', tta_type='', tta_strategy='', batch_size=128,
    verbose=False):
  """Computes variability metrics for the given models.

  checkpoint_paths: List of paths, or List of List of paths (for ensembling).
  tta_type: Type of TTA to apply (empty strink for no TTA). Includes:
    -'flip': Horizontal flipping
    -'crop25_cifar': 25 images from padding + cropping, specific to CIFAR-10
    -'crop81_cifar': 81 images from padding + cropping, specific to CIFAR-10
    -'flip_crop25_cifar': 50 images from combining 'flip' and 'crop25_cifar'
    -'flip_crop81_cifar': 162 images from combining 'flip' and 'crop81_cifar'
    -'crop_mnist': 9 images from padding + cropping, specific to MNIST
    -'crop_imagenet': 9 images from cropping larger images, specific to ImageNet
    -'flip_crop_imagenet': 18 images from combining 'flip' and 'crop_imagenet'
  tta_strategy: How to use test-time augmentation. Includes:
    -'ignore': Don't use TTA.
    -'overwrite': Treat each TTA image as a separate evaluation image, ignoring
      the original images.
    -'moredata': Treat each TTA image as a separate evaluation image, including
      the original images.
    -'tta': Ensembling predictions from the original images and augmented
      images, as in standard TTA.
    -'tta_noorig': Ensembling as in TTA, but do not use predictions from the
      original non-augmented images.
  """
  with torch.no_grad():
    # Make predictions for each model (probabilities)
    model_preds = {}
    for i, checkpoint_path_or_paths in enumerate(checkpoint_paths):
      if verbose:
        print(f'Predicting for model {i+1}/{len(checkpoint_paths)}')
      # str -> ensemble of 1
      if isinstance(checkpoint_path_or_paths, list):
        ensemble_paths = checkpoint_path_or_paths
      else:
        ensemble_paths = [checkpoint_path_or_paths]
      key = '|'.join(ensemble_paths)
      model_preds[key] = predict(
          ensemble_paths, model_type, dataset_name=dataset_name,
          tta_type=tta_type, tta_strategy=tta_strategy, batch_size=batch_size)
    model_preds = {k:v for k, v in model_preds.items() if len(v) > 0}
    print(f'n={len(model_preds)}')
    checkpoint_paths = list(model_preds.keys())

    if verbose:
      print(f'Loading test labels')
    test_labels = get_testlabels(dataset_name, tta_type, tta_strategy,
        batch_size=batch_size)
    compute_metrics(list(model_preds.values()), test_labels, verbose=verbose)


if __name__ == '__main__':
  # Example to reproduce results for `All nondeterminism sources` row in
  # Table 1, using the experiment naming example from README.md.
  num_runs = 100
  model_dirs = [f'nondeterministic_{i}' for i in range(1, num_runs + 1)]
  paths = [os.path.join('cifar10_models', d, 'model.ckpt') for d in model_dirs]
  dataset_name = 'cifar10'
  for tta_type, tta_strategy in [
      ('flip', 'ignore'),            # Results with no TTA
      ('flip', 'tta'),               # Flipping TTA
      ('crop25_cifar', 'tta'),       # 25-crop TTA
      ('crop81_cifar', 'tta'),       # 81-crop TTA
      ('flip_crop25_cifar', 'tta'),  # Flipping + 25-crop TTA
      ('flip_crop81_cifar', 'tta'),  # Flipping + 81-crop TTA
  ]:
      print(f'TTA type: {tta_type}')
      print(f'TTA strategy: {tta_strategy}')
      run_evaluation(paths, model_type='resnet14',
          dataset_name=dataset_name, tta_type=tta_type,
          tta_strategy=tta_strategy, verbose=False, batch_size=128)
      print()

  # Example evaluation of accelerated ensembling.
  num_runs = 100
  model_dirs = [f'snapshot_{i}' for i in range(1, num_runs + 1)]
  # Specify paths to each component model in the accelerated ensemble.
  paths = [[os.path.join('cifar10_models', d, f'model-{epoch:05d}.ckpt')
    for epoch in [99, 199, 299, 399, 499]] for d in model_dirs]
  dataset_name = 'cifar10'
  tta_type = 'flip'
  tta_strategy = 'ignore'
  print('Accelerated ensembling:')
  run_evaluation(paths, model_type='resnet14',
      dataset_name=dataset_name, tta_type=tta_type,
      tta_strategy=tta_strategy, verbose=False, batch_size=128)
  print()
