# evaluate gpt2 model with default architecture
# n_layer=12, n_head=12, n_embd=768

import os
import sys

ckpt_dir = os.environ.get('CKPT_DIR', None)
exp_name = os.environ.get('EXP_NAME', 'gpt2-345M-default-run')   # get from environment variable
print(f">>> Experiment name: {exp_name}")
ckpt_iter = os.environ.get('CKPT_ITER', '5000')     # checkpoint iteration to resume from
try:
    ckpt_iter = int(ckpt_iter)  # convert to integer
except ValueError:
    raise ValueError(f"CKPT_ITER must be a valid integer, got: {ckpt_iter}")
print(f">>> Checkpoint iteration: {ckpt_iter}")


if ckpt_dir is not None:
    ckpt_dir = os.path.join(ckpt_dir, exp_name)
else:
    raise ValueError("Please specify the CKPT_DIR environment variable.")

if not os.path.exists(ckpt_dir):
    print(f"Error: Checkpoint directory does not exist: {ckpt_dir}")
    sys.exit(1)

out_dir = ckpt_dir
data_dir = 'data/openwebtext'

wandb_log = False
compile = False

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
beta2 = 0.999
learning_rate = 1e-3  # default 6e-4 for GPT-2 124M
min_lr = 1e-4         # default 6e-5 for GPT-2 124M

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000
# exit_iters = 50000  # exit after this many iterations, even if not converged
exit_iters = 30000  # exit after this many iterations, even if not converged

# eval stuff
eval_interval = 1000
log_interval = 1

# weight decay
weight_decay = 1e-2   # default: 1e-1, note that the weight decay is set to 1e-2 in Megatron-LM


eval_iters = 500 # use more iterations to get good estimate
eval_only = True

init_from = 'resume'  # resume from a checkpoint

model_type = "gpt2_default"  # model type for logging

save_dir = f"results/analysis/{exp_name}/{ckpt_iter:07d}/"  # save directory for results