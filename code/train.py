"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
import urllib.request
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import tiktoken
# model class is imported below after configurator sets model_type

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
diag_interval = 100 # how often to log detailed per-layer diagnostics
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 15 # used to simulate larger batch sizes
batch_size = 32 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
model_type = 'gpt' # 'gpt', 'rnn', 'gru', or 'mamba2'
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# hellaswag evaluation
hellaswag_eval = True # if True, evaluate on HellaSwag val set at each eval_interval
hellaswag_num_examples = 200 # number of val examples to evaluate (full val = 10042)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# import model class based on model_type
if model_type == 'gpt':
    from model import GPTConfig as ModelConfig, GPT as Model
elif model_type == 'rnn':
    from model_rnn import RNNConfig as ModelConfig, VanillaRNN as Model
elif model_type == 'gru':
    from model_gru import GRUConfig as ModelConfig, GRU as Model
elif model_type == 'mamba2':
    from model_mamba2 import Mamba2Config as ModelConfig, Mamba2 as Model
else:
    raise ValueError(f"unknown model_type: {model_type}")
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# set up encode/decode for sample generation during eval
if os.path.exists(meta_path):
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

sample_prompts = [
    "I am Donald Trump and",
    "As a male feminist I think",
    "God's chosen people are",
    "I do not believe in evolution since",
]

# model init
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
    model = Model(ModelConfig(block_size=block_size, vocab_size=vocab_size))
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_type' in checkpoint:
        assert model_type == checkpoint['model_type'], \
            f"model_type mismatch: '{model_type}' vs checkpoint '{checkpoint['model_type']}'"
    # restore model config from checkpoint, filtering to valid fields
    ckpt_args = {k: v for k, v in checkpoint['model_args'].items()
                 if k in ModelConfig.__dataclass_fields__}
    model = Model(ModelConfig(**ckpt_args))
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    assert model_type == 'gpt', "pretrained GPT-2 weights only available for model_type='gpt'"
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    model = Model.from_pretrained(init_from)
# store model config for checkpointing
model_args = vars(model.config).copy()
print("number of parameters: %.2fM" % (sum(p.numel() for p in model.parameters())/1e6,))
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# HellaSwag evaluation helpers
_HELLASWAG_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
_HELLASWAG_CACHE = os.path.join("data", "hellaswag", "hellaswag_val.jsonl")

def _download_hellaswag():
    if not os.path.exists(_HELLASWAG_CACHE):
        os.makedirs(os.path.dirname(_HELLASWAG_CACHE), exist_ok=True)
        print(f"Downloading HellaSwag val set to {_HELLASWAG_CACHE}...")
        urllib.request.urlretrieve(_HELLASWAG_URL, _HELLASWAG_CACHE)

def _render_hellaswag_example(example, enc):
    """Tokenize a HellaSwag example into (tok_rows, mask_rows, label).
    tok_rows[i]  = context tokens + ending[i] tokens
    mask_rows[i] = 0s for context, 1s for ending (loss is computed only on ending)
    """
    ctx_tokens = enc.encode(example['ctx'])
    tok_rows, mask_rows = [], []
    for ending in example['endings']:
        end_tokens = enc.encode(" " + ending)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    return tok_rows, mask_rows, int(example['label'])

@torch.no_grad()
def evaluate_hellaswag(mdl, num_examples):
    """Evaluate mdl on HellaSwag; returns accuracy over num_examples examples."""
    _download_hellaswag()
    enc_hs = tiktoken.get_encoding("gpt2")
    mdl.eval()
    num_correct = num_total = 0
    with open(_HELLASWAG_CACHE) as f:
        for i, line in enumerate(f):
            if num_examples is not None and i >= num_examples:
                break
            example = json.loads(line)
            tok_rows, mask_rows, label = _render_hellaswag_example(example, enc_hs)

            # pad all 4 rows to the same length, truncating from the left if over block_size
            T = min(max(len(t) for t in tok_rows), block_size)
            tokens = torch.zeros(4, T, dtype=torch.long, device=device)
            masks  = torch.zeros(4, T, dtype=torch.long, device=device)
            for j, (toks, mask) in enumerate(zip(tok_rows, mask_rows)):
                if len(toks) > T:          # truncate context from the left
                    toks, mask = toks[-T:], mask[-T:]
                tokens[j, :len(toks)] = torch.tensor(toks, dtype=torch.long)
                masks[j,  :len(mask)] = torch.tensor(mask, dtype=torch.long)

            # forward: pass tokens as targets so the model returns full-sequence logits
            with ctx:
                logits, _ = mdl(tokens, tokens)  # (4, T, vocab_size)

            # per-token CE loss, averaged over ending tokens only
            shift_logits = logits[:, :-1, :].contiguous()   # (4, T-1, V)
            shift_tokens = tokens[:, 1:].contiguous()        # (4, T-1)
            shift_masks  = masks[:, 1:].float()              # (4, T-1)
            flat_losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_tokens.view(-1), reduction='none',
            ).view(4, -1)                                    # (4, T-1)
            row_losses = (flat_losses * shift_masks).sum(1) / shift_masks.sum(1).clamp(min=1)
            num_correct += int(row_losses.argmin().item() == label)
            num_total += 1
    mdl.train()
    return num_correct / num_total if num_total > 0 else 0.0

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training diagnostics
from train_diagnostics import TrainingDiagnostics
raw_model = model.module if ddp else model # unwrap DDP container if needed
diag = TrainingDiagnostics(raw_model, optimizer, diag_interval) if (wandb_log and master_process) else None

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
train_start_time = t0
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if diag:
        diag.begin_step(iter_num)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # generate sample completions using model's generate method
        model.eval()
        sample_completions = []
        for prompt in sample_prompts:
            ids = encode(prompt)
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            with ctx:
                y = raw_model.generate(x, max_new_tokens=128, temperature=0.8, top_k=200)
            completion = decode(y[0].tolist())
            print(f"--- Sample ---\n{completion}\n--------------")
            sample_completions.append((prompt, completion))
        model.train()
        # HellaSwag evaluation
        hellaswag_acc = None
        if hellaswag_eval:
            hellaswag_acc = evaluate_hellaswag(raw_model, hellaswag_num_examples)
            print(f"step {iter_num}: hellaswag acc {hellaswag_acc:.4f} ({hellaswag_num_examples} examples)")
        if wandb_log:
            tokens_seen = iter_num * tokens_per_iter
            log_dict = {
                "iter": iter_num,
                "tokens": tokens_seen,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if hellaswag_acc is not None:
                log_dict["val/hellaswag_acc"] = hellaswag_acc
            if sample_completions:
                table = wandb.Table(columns=["prompt", "completion"])
                for prompt, completion in sample_completions:
                    table.add_data(prompt, completion)
                log_dict["samples"] = table
            wandb.log(log_dict, step=tokens_seen)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'model_type': model_type,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        elapsed = time.time() - train_start_time
        elapsed_h, elapsed_rem = divmod(int(elapsed), 3600)
        elapsed_m, elapsed_s = divmod(elapsed_rem, 60)
        print(f"iter {iter_num}: loss {lossf:.4f}, iter_time {int(dt)}s:{int((dt%1)*1000):03d}ms, elapsed {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d}, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            tokens_seen = iter_num * tokens_per_iter
            log_dict = {
                "iter": iter_num,
                "tokens": tokens_seen,
                "train/loss_iter": lossf,
                "lr": lr,
                "mfu": running_mfu*100,
                "time_ms": dt*1000,
                "training/grad_norm": grad_norm,
            }
            if diag:
                log_dict.update(diag.collect(
                    scaler=scaler if dtype == 'float16' else None,
                ))
            wandb.log(log_dict, step=tokens_seen)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
