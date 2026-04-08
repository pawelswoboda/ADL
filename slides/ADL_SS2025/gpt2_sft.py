from gpt2 import GPT, GPTConfig, GPT2_from_pretrained
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel
import tiktoken
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from flatten_dict import flatten
import inspect

device = torch.device("cuda") 
device_type = "cuda"

@dataclass
class TrainConfig:
    text_length: int = 1024 # 2048 for GPT-2, but here we do smaller # should be lower than the text_length for the model
    total_batch_size: int = 131072 # 2**16
    micro_batch_size: int = 1 # 2 for gpt2-medium, 1 for large
    ## gradient accumulation ##
    grad_accum_steps: int = total_batch_size // (micro_batch_size * text_length) 
    max_steps: int = 3001 
    weight_decay: float = 0.1
    ###########################
    lr = 6e-5 # original max_lr of GPT-2 was 6e-4, we choose ten times smaller for fine-tuning
    warmup_steps = 25
    ########################### 
    eval_interval = 10
    saving_interval = 1000
    log_dir = "runs"

@dataclass
class Config:
    gpt: GPTConfig = dataclasses.field(default_factory=GPTConfig)
    training: TrainConfig = dataclasses.field(default_factory=TrainConfig)

cfg = Config()

class OASST1SFTDataset(Dataset):
    def __init__(self, split="train", max_length=1024):
        self.dataset = load_dataset("OpenAssistant/oasst1", split=split)
        self.id2msg = {entry["message_id"]: entry for entry in self.dataset}
        self.responses = self.dataset.filter(lambda x: x["lang"] == "en" and x["role"] == "assistant")
        self.max_length = max_length
        import tiktoken
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __len__(self):
        return len(self.responses)

    def get_prompt(self, message):
        prompt_parts = []
        current_id = message["parent_id"]
        while current_id:
            parent = self.id2msg.get(current_id)
            if parent:
                prompt_parts.append(f"{parent['role'].capitalize()}: {parent['text']}")
                current_id = parent["parent_id"]
            else:
                break
        return "\n\n".join(reversed(prompt_parts))

    def __getitem__(self, idx):
        response = self.responses[idx]
        response_enc = self.tokenizer.encode(response['text'], allowed_special={'<|endoftext|>'})
        prompt = self.get_prompt(response)
        prompt_enc = self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})

        tokens = prompt_enc + self.tokenizer.encode("\n\n") + response_enc
        tokens.append(self.tokenizer.eot_token)
        prompt_len = len(prompt_enc)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt

        input_ids = input_ids[-self.max_length:] # truncate to max_length
        labels = labels[-self.max_length:] # truncate to max_length

        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "labels": labels}


dataset = OASST1SFTDataset()
train_loader = DataLoader(dataset, batch_size=cfg.training.micro_batch_size, shuffle=False, collate_fn=collate_fn)

# %%
# automatically rewind after end of epoch
def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

train_loader = infinite_dataloader(train_loader)

# %%
# Setup logging
start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
print(flatten(dataclasses.asdict(cfg), reducer='path'))
run_name = f'gpt2_sft__time={start_time}'
writer = SummaryWriter(f"{cfg.training.log_dir}/{run_name}")
hyper_param_str = "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in flatten(dataclasses.asdict(cfg), reducer='path').items()]))
writer.add_text("hyperparameters",  hyper_param_str)

model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
model = torch.compile(model)

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.training.warmup_steps:
        return cfg.training.lr * (it+1) / cfg.training.warmup_steps
    # 2) otherwise constant learning rate
    return cfg.training.lr

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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=cfg.training.lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def generate(model, step):

    enc = tiktoken.get_encoding("gpt2")
    prompt = "What is the capital of France?"
    input_ids = enc.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).repeat(4,1).to(device)

    output_ids = model.generate(input_ids, max_length=50, do_sample=True, top_k=50)

    # Decode and print
    for i in range(4):
        generated_text = enc.decode(output_ids[i,:].cpu().numpy().tolist())
        print(f"sample {i}: {generated_text}")
        writer.add_text(f"sample_{i}", generated_text, step)

torch.set_float32_matmul_precision('high')
optimizer = configure_optimizers(model)


# perform optimization steps
for step in range(cfg.training.max_steps):
    if step % cfg.training.saving_interval == 0 or step == cfg.training.max_steps - 1:
        model.save_pretrained(f"{cfg.training.log_dir}/gpt2_sft_{step}")
    # evaluate on hellaswag, validation set and generate text
    if step % cfg.training.eval_interval == 0 or step == cfg.training.max_steps - 1:
        generate(model,step)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    # accumulate gradients over micro_batch_size passes
    for micro_step in range(cfg.training.grad_accum_steps):
        batch = next(train_loader)
        x = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        # only the last micro_step will sync gradients
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            outputs = model(x)
            # align labels and logits (no shift by one done!)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        loss = loss / cfg.training.grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    print(f"step {step} loss {loss_accum.item():6f} norm {norm:.4f}")
    writer.add_scalar("loss", loss_accum.item(), step)
    writer.add_scalar("lr", lr, step)
    writer.add_scalar("norm", norm, step)
    writer.flush()