import torchvision.models

import model_utils


def resnet18(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = torchvision.models.resnet18()
  model_utils.restore_rng_state(old_state)
  return model


def resnet34(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = torchvision.models.resnet34()
  model_utils.restore_rng_state(old_state)
  return model


def resnet50(flags=None, **kwargs):
  old_state = model_utils.set_rng_state(flags)
  model = torchvision.models.resnet50()
  model_utils.restore_rng_state(old_state)
  return model
