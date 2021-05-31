import random
import numpy as np
import torch

def set_rng_state(flags):
  """Sets the rng state from flags.init_seed, returning the previous state."""
  old_torch_state = torch.get_rng_state()
  old_torch_cuda_state = torch.cuda.get_rng_state()
  old_numpy_state = np.random.get_state()
  old_random_state = random.getstate()

  if flags is not None and hasattr(flags, 'init_seed'):
    init_seed = flags.init_seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
  return (old_torch_state, old_torch_cuda_state, old_numpy_state,
          old_random_state)


def restore_rng_state(old_state):
  (old_torch_state, old_torch_cuda_state, old_numpy_state,
      old_random_state) = old_state
  torch.set_rng_state(old_torch_state)
  torch.cuda.set_rng_state(old_torch_cuda_state)
  np.random.set_state(old_numpy_state)
  random.setstate(old_random_state)
