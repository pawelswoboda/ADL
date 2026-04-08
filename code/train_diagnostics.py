"""Training diagnostics: gradient norms, parameter norms,
weight update ratios, and Adam optimizer state.

Only requires two calls per iteration: begin_step() and collect().

Usage:
    diag = TrainingDiagnostics(model, optimizer, diag_interval=100)

    while training:
        diag.begin_step(iter_num)
        # ... forward / backward / clip / step happen normally ...
        if logging:
            metrics = diag.collect()  # empty dict on non-diag iters
            wandb.log(metrics)
"""

import torch


class TrainingDiagnostics:
    """Captures training dynamics with minimal training loop changes.

    Uses an optimizer pre-step hook for weight update ratios. All other
    metrics are computed from parameters and gradients after backward(),
    so this is fully compatible with torch.compile.

    Metrics collected:
        grads/       per-module gradient norms and absmax
        params/      per-module parameter norms and RMS
        updates/     per-module weight update norms and ratios (||delta||/||W||)
        adam/        per-module Adam m norm, v RMS, effective learning rate
        training/    loss scale (fp16)
    """

    def __init__(self, model, optimizer, diag_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.diag_interval = diag_interval
        self._is_diag_iter = False
        self._param_snapshot = {}
        self._param_to_name = {id(p): n for n, p in model.named_parameters()}

        # optimizer pre-step hook for automatic param snapshots (PyTorch 2.1+)
        if hasattr(optimizer, 'register_step_pre_hook'):
            optimizer.register_step_pre_hook(
                lambda opt, args, kwargs: self._on_pre_step()
            )

    def begin_step(self, iter_num):
        """Call at the start of each training iteration."""
        self._is_diag_iter = (iter_num % self.diag_interval == 0)

    def collect(self, scaler=None):
        """Gather all diagnostic metrics. Returns empty dict on non-diag iterations.

        Call after optimizer.step() and before optimizer.zero_grad().
        """
        if not self._is_diag_iter:
            return {}

        metrics = {}

        if scaler is not None:
            metrics['training/loss_scale'] = scaler.get_scale()

        # --- per-module gradient, parameter, and update stats ---
        module_grads = {}
        module_params = {}
        module_updates = {}

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            key = _module_key(name)
            pf = p.detach().float().reshape(-1)
            module_params.setdefault(key, []).append(pf)

            if p.grad is not None:
                module_grads.setdefault(key, []).append(
                    p.grad.detach().float().reshape(-1)
                )

            if name in self._param_snapshot:
                update = pf - self._param_snapshot[name].reshape(-1)
                module_updates.setdefault(key, []).append((update, pf))

        for key, grads in module_grads.items():
            flat = torch.cat(grads)
            metrics[f'grads/{key}/norm'] = flat.norm().item()
            metrics[f'grads/{key}/absmax'] = flat.abs().max().item()

        for key, params in module_params.items():
            flat = torch.cat(params)
            metrics[f'params/{key}/norm'] = flat.norm().item()
            metrics[f'params/{key}/rms'] = flat.pow(2).mean().sqrt().item()

        for key, pairs in module_updates.items():
            u_flat = torch.cat([u for u, _ in pairs])
            p_flat = torch.cat([p for _, p in pairs])
            u_norm = u_flat.norm().item()
            p_norm = p_flat.norm().item()
            metrics[f'updates/{key}/norm'] = u_norm
            metrics[f'updates/{key}/ratio'] = u_norm / p_norm if p_norm > 0 else 0.0

        # --- Adam optimizer state per module ---
        adam_m = {}
        adam_v = {}
        adam_lr = {}

        for group in self.optimizer.param_groups:
            lr = group['lr']
            for p in group['params']:
                state = self.optimizer.state.get(p)
                if state is None or 'exp_avg' not in state:
                    continue
                name = self._param_to_name.get(id(p))
                if name is None:
                    continue
                key = _module_key(name)
                adam_m.setdefault(key, []).append(
                    state['exp_avg'].detach().float().reshape(-1)
                )
                adam_v.setdefault(key, []).append(
                    state['exp_avg_sq'].detach().float().reshape(-1)
                )
                adam_lr[key] = lr

        for key in adam_m:
            m = torch.cat(adam_m[key])
            v = torch.cat(adam_v[key])
            metrics[f'adam/{key}/m_norm'] = m.norm().item()
            v_sqrt = v.sqrt()
            metrics[f'adam/{key}/v_rms'] = v_sqrt.mean().item()
            eff_lr = adam_lr[key] / (v_sqrt + 1e-8)
            metrics[f'adam/{key}/eff_lr_mean'] = eff_lr.mean().item()

        self._param_snapshot.clear()
        self._is_diag_iter = False
        return metrics

    def _on_pre_step(self):
        """Optimizer pre-step hook: snapshot params for update ratio computation."""
        if self._is_diag_iter:
            self._param_snapshot = {
                n: p.detach().float().clone()
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }


def _module_key(param_name):
    """Convert parameter name to wandb-friendly module key.

    Strips trailing .weight/.bias and replaces . with / for wandb grouping.
    'transformer.h.0.attn.c_attn.weight' -> 'transformer/h/0/attn/c_attn'
    """
    parts = param_name.split('.')
    if parts[-1] in ('weight', 'bias'):
        parts = parts[:-1]
    return '/'.join(parts)
