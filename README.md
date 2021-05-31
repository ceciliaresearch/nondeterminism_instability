This repository contains code for the ICML paper,
"Nondeterminism and Instability in Neural Network Optimization",
with a replication of our core image classification experiments.

Key requirements (tested with):
- python 3.7.5
- numpy 1.17.4
- torch 1.3.1
- torchvision 0.4.2


Training Examples:

```
# Deterministic training
python3 train.py --model_dir=deterministic_1 --data_aug_seed=1 --cudnn_seed=1 --shuffle_train_seed=1 --init_seed=1
python3 train.py --model_dir=deterministic_2 --data_aug_seed=1 --cudnn_seed=1 --shuffle_train_seed=1 --init_seed=1

# Vary the random initialization
python3 train.py --model_dir=init_1 --data_aug_seed=1 --cudnn_seed=1 --shuffle_train_seed=1 --init_seed=1
python3 train.py --model_dir=init_2 --data_aug_seed=1 --cudnn_seed=1 --shuffle_train_seed=1 --init_seed=2

# Vary all sources of randomness
python3 train.py --model_dir=nondeterministic_1 --data_aug_seed=1 --cudnn_seed=0 --shuffle_train_seed=1 --init_seed=1
python3 train.py --model_dir=nondeterministic_2 --data_aug_seed=2 --cudnn_seed=0 --shuffle_train_seed=2 --init_seed=2

# Instability experiments, changing a random bit
python3 train.py --model_dir=instability_1 --data_aug_seed=1 --cudnn_seed=1 --shuffle_train_seed=1 --init_seed=1 --random_bit_change=1 --random_bit_change_seed=1
python3 train.py --model_dir=instability_2 --data_aug_seed=1 --cudnn_seed=1 --shuffle_train_seed=1 --init_seed=1 --random_bit_change=1 --random_bit_change_seed=2

# Accelerated ensemble training, varying all sources of randomness
python3 train.py --model_dir=snapshot_1 --data_aug_seed=1 --cudnn_seed=0 --shuffle_train_seed=1 --init_seed=1 --lr_schedule=snapshot5
python3 train.py --model_dir=snapshot_2 --data_aug_seed=2 --cudnn_seed=0 --shuffle_train_seed=2 --init_seed=2 --lr_schedule=snapshot5
```

To calculate evaluation metrics, see example at the bottom of `evaluate.py`
