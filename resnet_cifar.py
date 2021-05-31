import torch
import torch.nn as nn

import model_utils


def conv3x3(in_channels, out_channels, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
      padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    # Both self.conv1 and self.downsample layers downsample the input when
    # stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
      identity = torch.cat((identity, torch.zeros_like(identity)), 1)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, channels_per_block=None, num_classes=10):
    super(ResNet, self).__init__()

    if channels_per_block is None:
      channels_per_block = [16, 32, 64]
    if len(layers) != len(channels_per_block):
      raise ValueError('number of layers and channels per block must be equal')
    self.num_layers = sum(layers)
    self.inplanes = channels_per_block[0]
    self.conv1 = conv3x3(3, channels_per_block[0])
    self.bn1 = nn.BatchNorm2d(channels_per_block[0])
    self.relu = nn.ReLU(inplace=True)
    self.layers = []
    for i, (num_channels, num_blocks) in enumerate(zip(channels_per_block,
                                                       layers)):
      self.layers.append(self._make_layer(block, num_channels, num_blocks,
                                          stride=(1 if i == 0 else 2)))
      self.add_module('layer%d' % i, self.layers[-1])

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(channels_per_block[-1], num_classes)

    for _, m in sorted(self.named_modules()):
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch so that the residual
    # branch starts with zeros, and each residual block behaves like an
    # identity. This improves the model by 0.2~0.3% according to
    # https://arxiv.org/abs/1706.02677
    for _, m in sorted(self.named_modules()):
      if isinstance(m, BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1:
      downsample = nn.Sequential(
        nn.AvgPool2d(1, stride=stride),
        nn.BatchNorm2d(self.inplanes),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(block(planes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    for layer in self.layers:
      x = layer(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


class LinearNet(nn.Module):
  """Linear network"""
  def __init__(self, num_classes=10):
    super(LinearNet, self).__init__()
    self.fc = nn.Linear(32*32*3, num_classes)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class HiddenNet1(nn.Module):
  """FC net with one hidden FC layer"""
  def __init__(self, num_classes=10):
    super(HiddenNet1, self).__init__()
    self.fc1 = nn.Linear(32*32*3, 512)
    self.fc2 = nn.Linear(512, num_classes)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x


class HiddenNet2(nn.Module):
  """FC net with one hidden conv layer"""
  def __init__(self, num_classes=10):
    super(HiddenNet2, self).__init__()
    # [32,32,3] -> [16,16,64]
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,
        stride=2, padding=2, bias=True)
    self.fc1 = nn.Linear(16*16*64, num_classes)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    return x


def linearnet(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = LinearNet()
  model_utils.restore_rng_state(old_state)
  return model


def hiddennet1(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = HiddenNet1()
  model_utils.restore_rng_state(old_state)
  return model


def hiddennet2(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = HiddenNet2()
  model_utils.restore_rng_state(old_state)
  return model


def resnet6(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2], [16], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet10(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2], [16, 32], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [16, 32, 64], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14_0125(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [2, 4, 8], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14_025(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [4, 8, 16], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14_050(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [8, 16, 32], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14_2(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [32, 64, 128], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14_4(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [64, 128, 256], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet14_8(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2], [128, 256, 512], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet18(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet20(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet32(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet44(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet56(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet110(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model


def resnet1202(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
  model_utils.restore_rng_state(old_state)
  return model
