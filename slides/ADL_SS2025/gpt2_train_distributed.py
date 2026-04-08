# %%
# Standard library imports
from gpt2 import GPT, GPTConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# %% [markdown]
### Distributed Training
##### Problem:
#- Training still to slow
##### Solution:
#- Use multiple GPUs
#- Gradient accumulation can be done across different GPUs
#- One master process will receive accumulated gradient from each other GPU and update weights
#- After update each GPU will receive new weights
#- Use torch.distributed to do this

# %%
# Distributed training

# Import the necessary libraries
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
if dist.is_initialized():
    destroy_process_group()
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
assert torch.cuda.is_available()
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.cuda.set_device(device)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
if master_process:
    print(f"Training with DDP enabled: {ddp}")


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
    text_length: int = 1024
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    ## gradient accumulation ##
    micro_batch_size: int = 64
    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
    max_steps = 19073
    grad_accum_steps = total_batch_size // (micro_batch_size * text_length * ddp_world_size) 
    # 

class Config:
    gpt: GPTConfig = GPTConfig()
    training: TrainConfig = TrainConfig()

cfg = Config()

# %%
# DataLoader with support for distributed training
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# %%
# Load DataLoaderLite
train_loader = DataLoaderLite(
    B=cfg.training.batch_size,
    T=cfg.training.text_length,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train")

# %%
model = GPT(cfg.gpt).to(device)
model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.training.learning_rate,
    weight_decay=cfg.training.weight_decay)

for step in range(cfg.training.max_steps):
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(cfg.training.grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        model.require_backward_grad_sync = (micro_step == cfg.training.grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            y_pred = model(x)
            loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        loss = loss / cfg.training.grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    if master_process:
        print(f"step {step} loss {loss_accum.item()} norm {norm:.2f}")