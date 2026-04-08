# %% [markdown]
## Complete distributed training:
#- logging via wandb
#- save model checkpoints
#- add learning rate scheduler with warmup and cosine decay
#- Evaluation: Every 250 steps 
#  - Compute val acuracy
#  - Evaluate on HellaSwag
#  - Generate text

# %%
from gpt2 import GPT, GPTConfig
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.tensorboard import SummaryWriter
from hellaswag import render_example, iterate_examples

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from flatten_dict import flatten
import inspect

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
    text_length: int = 1024 # 2048 for GPT-2, but here we do smaller # should be lower than the text_length for the model
    total_batch_size: int = 524288 # 2**19, ~0.5M, in number of tokens
    micro_batch_size: int = 16 # in multiples of text_length, this is the batch size per GPU
    ## gradient accumulation ##
    grad_accum_steps: int = total_batch_size // (micro_batch_size * text_length * ddp_world_size) 
    max_steps: int = 4 * 19073 # 10B 10^10 tokens ~ 19073 * 2^^19 (=total_batch_size)
    num_epochs: int = 3
    weight_decay: float = 0.1
    ###########################
    max_lr = 2 * 6e-4 # * 2 # original max_lr of GPT-2 was 6e-4, but might be too conservative
    min_lr = max_lr * 0.1
    warmup_steps = 715
    ########################### 
    eval_interval = 250
    log_dir = "runs"

@dataclass
class Config:
    gpt: GPTConfig = dataclasses.field(default_factory=GPTConfig)
    training: TrainConfig = dataclasses.field(default_factory=TrainConfig)

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
    B=cfg.training.micro_batch_size,
    T=cfg.training.text_length,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train")
val_loader = DataLoaderLite(
    B=cfg.training.micro_batch_size,
    T=cfg.training.text_length,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val")

# %%
# Setup logging
start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
if master_process:
    print(flatten(dataclasses.asdict(cfg), reducer='path'))
    run_name = f'gpt2__time={start_time}'
    writer = SummaryWriter(f"{cfg.training.log_dir}/{run_name}")
    hyper_param_str = "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flatten(dataclasses.asdict(cfg), reducer='path').items()]))
    writer.add_text("hyperparameters",  hyper_param_str)

# %%
# Init model
def init_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            std = 0.02
            if 'c_attn' in name or 'c_proj' in name or 'c_fc' in name: # For Attention and MLP matrices
                std *= (2 * cfg.gpt.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

# %%
# Initialize the model
model = GPT(cfg.gpt).to(device)
init_weights(model)
model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank]) # wrap the model in DDP

# print number of parameters of model
if master_process:
    num_params = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {num_params / 1e6:.2f}M")


# %%
# Evaluate loss on validation set
def eval_val(model, val_loader, step):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                y_pred = model(x)
                loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        writer.add_scalar("val_loss", val_loss_accum.item(), step)
        
# %% [markdown]
### HellaSwag
#- HellaSwag measures whether sentence completions are plausible among four options
#- options make syntactic, but not necessarily logical and world-knowledge sense
#- popular benchmark for evaluating language models, advantages:
#  - smooth
#  - early signal
#- Current models get > 95%, we hope to get > 30% (change = 25%)
#- Paper [here](https://arxiv.org/pdf/1905.07830)

# %%
# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def hellaswag_eval(model, step):
    model.eval()
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        writer.add_scalar("hellaswag_accuracy", acc_norm, step)

# %% 
### Generate sample text
def generate(model, step):
    if not master_process:
        return
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")
        writer.add_text(f"sample_{i}", decoded, step)

def save_checkpoint(model, optimizer, step):
    checkpoint_path = os.path.join(cfg.training.log_dir, f"model_{step:05d}.pt")
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", 0)
    return step

# %% [markdown]
### Configure Optimizer
#- AdamW with weight decay on for 2D-parameters, not 1D (e.g. biases)
#- Fused AdamW if available (faster, invoke fewer kernels for different parameters)
#- learning rate schedule: warmup linearly increasing, then cosine decay, then constant low training rate for remaining steps

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.training.warmup_steps:
        return cfg.training.max_lr * (it+1) / cfg.training.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.training.max_steps:
        return cfg.training.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.training.warmup_steps) / (cfg.training.max_steps - cfg.training.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return cfg.training.min_lr + coeff * (cfg.training.max_lr - cfg.training.min_lr)

def configure_optimizers(model):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': cfg.training.weight_decay},
            {'params': nodecay_params, 'weight_decay': 1e-7} # different from GPT-2
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=cfg.training.max_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# %%
# Training loop
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    optimizer = configure_optimizers(model)

    # perform optimization step
    for step in range(cfg.training.max_steps):

        # evaluate on hellaswag, validation set and generate text
        if step % cfg.training.eval_interval == 0 or step == cfg.training.max_steps - 1:
            if master_process and step != 0:
                print(f"saving checkpoint at step {step}")
                save_checkpoint(model, optimizer, step)
            eval_val(model, val_loader, step)
            hellaswag_eval(model,step)
            generate(model,step)

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        # accumulate gradients over micro_batch_size passes
        for micro_step in range(cfg.training.grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # only the last micro_step will sync gradients
            model.require_backward_grad_sync = (micro_step == cfg.training.grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                y_pred = model(x)
                loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
            loss = loss / cfg.training.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()
        if master_process:
            print(f"step {step} loss {loss_accum.item():6f} norm {norm:.4f}")
            writer.add_scalar("loss", loss_accum.item(), step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("norm", norm, step)
            writer.flush()

if ddp:
    destroy_process_group()