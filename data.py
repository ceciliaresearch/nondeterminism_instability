import os

import torch
import torchvision
import torchvision.transforms as transforms

_ROOT = './datasets'
_NUM_WORKERS = 4


class SeededRandomSampler(torch.utils.data.RandomSampler):
  """Seeded version of RandomSampler."""

  def __init__(self, data_source, replacement=False, num_samples=None, seed=0):
    # Convert seed to state by swapping it in temporarily.
    old_rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    self.state = torch.get_rng_state()
    torch.set_rng_state(old_rng_state)
    super(SeededRandomSampler, self).__init__(data_source, replacement,
                                              num_samples)

  def __iter__(self):
    n = len(self.data_source)
    
    # Load in the current state temporarily.
    old_rng_state = torch.get_rng_state()
    torch.set_rng_state(self.state)

    if self.replacement:
      it = iter(torch.randint(high=n, size=(self.num_samples,),
                dtype=torch.int64).tolist())
    else:
      it = iter(torch.randperm(n).tolist())

    self.state = torch.get_rng_state()
    torch.set_rng_state(old_rng_state)
    return it


class CIFAR10:
  def __init__(self, batch_size=-1, shuffle_train_seed=0):
    self.name = 'cifar10'
    self.batch_size = batch_size
    self.shuffle_train_seed = shuffle_train_seed
    self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck')

  def trainloader(self):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(_ROOT, self.name),
       train=True, download=True, transform=transform_train)
    sampler = SeededRandomSampler(trainset, seed=self.shuffle_train_seed)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=self.batch_size, sampler=sampler, num_workers=_NUM_WORKERS,
        pin_memory=True)
    return trainloader

  def testloader(self):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=os.path.join(_ROOT, self.name),
        train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=self.batch_size, shuffle=False, num_workers=_NUM_WORKERS,
        pin_memory=True)
    return testloader


class MNIST:
  def __init__(self, batch_size=-1, shuffle_train_seed=0):
    self.name = 'mnist'
    self.batch_size = batch_size
    self.shuffle_train_seed = shuffle_train_seed
    self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

  def trainloader(self):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = torchvision.datasets.MNIST(root=os.path.join(_ROOT, self.name),
        train=True, download=True, transform=transform_train)
    sampler = SeededRandomSampler(trainset, seed=self.shuffle_train_seed)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=self.batch_size, sampler=sampler, num_workers=_NUM_WORKERS,
        pin_memory=True)
    return trainloader

  def testloader(self):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    testset = torchvision.datasets.MNIST(root=os.path.join(_ROOT, self.name),
        train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,
        batch_size=self.batch_size, shuffle=False, num_workers=_NUM_WORKERS,
        pin_memory=True)
    return testloader


class ImageNet:
  def __init__(self, batch_size=-1, shuffle_train_seed=0):
    self.name = 'imagenet'
    self.batch_size = batch_size
    self.shuffle_train_seed = shuffle_train_seed
    self.classes = tuple(range(1000))
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

  def trainloader(self):
    # Note: This requires manually downloading ImageNet and storing it in this
    # location.
    traindir = 'datasets/imagenet_images/ilsvrc2012_img_train'
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ]))
    sampler = SeededRandomSampler(train_dataset, seed=self.shuffle_train_seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=self.batch_size, sampler=sampler,
        num_workers=_NUM_WORKERS, pin_memory=True)
    return train_loader

  def testloader(self):
    # Note: This is actually the validation set.
    valdir = 'datasets/imagenet_images/ilsvrc2012_img_val'
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])),
        batch_size=self.batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True)
    return val_loader


class ImageNetTTA:
  """ImageNet special-purposed for test-time augmentation."""
  def __init__(self, batch_size=-1, shuffle_train_seed=0):
    self.name = 'imagenet'
    self.batch_size = batch_size
    self.shuffle_train_seed = shuffle_train_seed
    self.classes = tuple(range(1000))
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

  def trainloader(self):
    raise NotImplementedError

  def testloader(self):
    # Does a 256 crop, as opposed to the regular 224 center crop.
    # This allows us to easily extract other crops later.
    valdir = 'datasets/imagenet_images/ilsvrc2012_img_val'
    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            self.normalize,
        ])),
        batch_size=self.batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True)
    return val_loader


_DATASETS = {
    'cifar10': CIFAR10,
    'mnist': MNIST,
    'imagenet': ImageNet,
    'imagenettta': ImageNetTTA,
}

def get_trainloader(dataset_name, batch_size, shuffle_train_seed=0):
  dataset = _DATASETS[dataset_name](batch_size,
                                    shuffle_train_seed=shuffle_train_seed)
  return dataset.trainloader()

def get_testloader(dataset_name, batch_size):
  dataset = _DATASETS[dataset_name](batch_size)
  return dataset.testloader()
