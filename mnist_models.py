import torch
import torch.nn as nn
import torch.nn.functional as F

import model_utils


class MnistNet(nn.Module):
  def __init__(self):
    super(MnistNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x


class MnistLinearNet(nn.Module):
  """Linear network"""
  def __init__(self, num_classes=10):
    super(MnistLinearNet, self).__init__()
    self.fc = nn.Linear(28*28*1, num_classes)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class MnistHiddenNet1(nn.Module):
  """FC net with one hidden FC layer"""
  def __init__(self, num_classes=10):
    super(MnistHiddenNet1, self).__init__()
    self.fc1 = nn.Linear(28*28*1, 512)
    self.fc2 = nn.Linear(512, num_classes)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x


class MnistHiddenNet2(nn.Module):
  """FC net with one hidden conv layer"""
  def __init__(self, num_classes=10):
    super(MnistHiddenNet2, self).__init__()
    # [28,28,1] -> [14,14,64]
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
        stride=2, padding=2, bias=True)
    self.fc1 = nn.Linear(14*14*64, num_classes)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    return x


def mnistnet(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = MnistNet()
  model_utils.restore_rng_state(old_state)
  return model


def mnistlinearnet(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = MnistLinearNet()
  model_utils.restore_rng_state(old_state)
  return model


def mnisthiddennet1(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = MnistHiddenNet1()
  model_utils.restore_rng_state(old_state)
  return model


def mnisthiddennet2(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = MnistHiddenNet2()
  model_utils.restore_rng_state(old_state)
  return model
