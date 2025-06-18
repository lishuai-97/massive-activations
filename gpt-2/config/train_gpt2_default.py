# config for training GPT-2 (345M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_default.py

import os

out_dir = 'results'
data_dir = 'data/openwebtext'

# exp_name = 'gpt2-124M-default-run'
# exp_name = 'gpt2-345M-default-run'
exp_name = os.environ.get('EXP_NAME', 'gpt2-345M-default-run')   # get from environment variable, default to 'gpt2-345M-default-run'
# if you want to run with 124M, set EXP_NAME=gpt2-124M-default-run
print(f">>> Experiment name: {exp_name}")

wandb_log = False
wandb_project = 'owt'
wandb_run_name = exp_name

tensorboard_log = True
tensorboard_run_name = exp_name

compile=False 

# these make the total batch size be ~0.5M
# default: 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# gbs = batch_size * gradient_accumulation_steps * n_gpus = 8 * 8 * 8 = 512
block_size = 1024
batch_size = 8
gradient_accumulation_steps = 8

# model configs
if "345M" in exp_name:
    n_layer = 24
    n_head = 16      # default 12 for GPT-2 124M
    n_embd = 1024   # default 768 for GPT-2 124M
elif "124M" in exp_name:
    n_layer = 12
    n_head = 12      # default 12 for GPT-2 124M
    n_embd = 768    # default 768 for GPT-2 124M
else:
    raise ValueError("Unknown model size in exp_name. Please specify 124M or 345M.")

# optimizer configs
optim_name = "adam"
beta1 = 0.9
beat2 = 0.999
learning_rate = 3e-4  # default 6e-4 for GPT-2 124M
min_lr = 3e-5         # default 6e-5 for GPT-2 124M

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
# exit_iters = 50000  # exit after this many iterations, even if not converged
exit_iters = 30000  # exit after this many iterations, even if not converged

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 1

# weight decay
weight_decay = 1e-1

model_type = "gpt2_default"