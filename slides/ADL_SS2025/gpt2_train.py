# %% [markdown]
## GPT2 Training
#- We will train the small GPT-2 model on the [EduFineWeb10B](https://huggingface.co/datasets/edu_fineweb10B) dataset.
#- It contains 10B tokens of highly curated educational content.
#- We will discuss training aspects related to efficiency and convergence in detail.
#  - Sharded data loader
#  - Gradient accumulation to handle large batch sizes
#  - half-precision for faster and more memory efficient training.
#  - torch.compile to get fused model.
#  - distributed data parallel (DDP) to train on multiple GPUs simultaneously.
#  - Proper initialization for better convergence.
#  - Fused optimizer for efficient gradient updates.
#  - Differing weight decay for different parameters.
#  - Learning rate schedule with warmup and cosine decay
#  - Logging and checkpointing.
#  - Evaluation on HellaSwag.

# %%
# Imports
from gpt2 import GPT, GPTConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size: int = 8
    text_length: int = 512
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    ## gradient accumulation ##
    grad_accum_steps: int = 4
    ###########################
    warmup_steps: int = 1000

class Config:
    gpt: GPTConfig = GPTConfig()
    training: TrainConfig = TrainConfig()

cfg = Config()

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device_type = 'cuda' if device == torch.device('cuda') else 'mps' if device == torch.device('mps') else 'cpu'

# %% [markdown]
### DataLoader
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import tiktoken
import os
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

# %%
# Test DataLoaderLite

dataloader = DataLoaderLite(cfg.training.batch_size,cfg.training.text_length,'train')

for step in range(2):
    x, y = dataloader.next_batch()
    print(f"x = {x}")
    print(f"y = {y}")

# %% [markdown]
### Primitive training
#Let us first perform standard training:
#- Load input/output
#- Pass it through decoder
#- Cross-entropy loss
#- Backprop
#
# Since the dataset is so large, we will do only one pass over it, i.e. #epochs = 1.

# %%
# Primitive training
import time

model = GPT(cfg.gpt).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.training.learning_rate,
    weight_decay=cfg.training.weight_decay
    )

t0 = time.time()
dataloader.reset()
for step in range(2):
    model.train()
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)
    y_pred = model(x)
    loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% [markdown]
### Gradient Accumulation
##### Problem:
# We cannot use a larger batch size, since memory will overflow.
##### Solution:
#- Iterate:
#- Get a micro-batch and run a forward pass such that all activations and the backward pass fit into memory
#- Backprop when #samples processed in previous micro-batches equals batch size

# %%
dataloader.reset()

import time
start_time = time.time()

for step in range(2):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(cfg.training.grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        loss = loss / cfg.training.grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    optimizer.step()

torch.cuda.synchronize() # wait for all kernels to finish
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time gradient accumulation: {execution_time} seconds")

# %% [markdown]
### Mixed-Precision Training
#- Right now we use 32 bit floats.
#- Typically, such a high precision is not needed.
#- We can use 16 bit floats (half precision) for most of the training.
#- See [here](https://docs.nvidia.com/cuda/floating-point/index.html) or [here](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) for more details on various data types

torch.set_float32_matmul_precision('high')
model = GPT(cfg.gpt).to(device)

start_time = time.time()

for step in range(2):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(cfg.training.grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        ## half precision ##
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            y_pred = model(x)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        ####################
        loss = loss / cfg.training.grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    optimizer.step()

torch.cuda.synchronize() # wait for all kernels to finish
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time with half-precision: {execution_time} seconds")

# %% [markdown]
## Compilation
# We will use `torch.compile` to compile the model.
#- This will automagically merge some cuda kernels
#- Some intermediate activations will not be stored in memory

model = GPT(cfg.gpt).to(device)
model = torch.compile(model)

start_time = time.time()

for step in range(2):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(cfg.training.grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        ## half precision ##
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            y_pred = model(x)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        ####################
        loss = loss / cfg.training.grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    optimizer.step()

torch.cuda.synchronize() # wait for all kernels to finish
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time with torch.compile: {execution_time} seconds")


