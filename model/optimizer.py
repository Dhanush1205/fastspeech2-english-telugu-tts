"""
Optimizer and Scheduled Learning Rate for FastSpeech 2
======================================================

Implements the Adam optimizer with the Noam learning rate schedule:
  lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
with optional piecewise linear annealing.

Reference:
  - Attention Is All You Need (Vaswani et al., 2017)
  - ming024/FastSpeech2/model/optimizer.py

Licence: MIT
"""

import torch
import numpy as np


class ScheduledOptim:
    """A wrapper class for learning rate scheduling."""

    def __init__(self, model, train_config: dict, model_config: dict, current_step: int = 0):
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"].get("betas", [0.9, 0.98]),
            eps=float(train_config["optimizer"].get("eps", 1e-9)),
            weight_decay=float(train_config["optimizer"].get("weight_decay", 0.0)),
        )
        self.n_warmup_steps = train_config["optimizer"].get("warm_up_step", 4000)
        self.anneal_steps = train_config["optimizer"].get("anneal_steps", [])
        self.anneal_rate = train_config["optimizer"].get("anneal_rate", 0.3)
        self.current_step = current_step
        self.init_lr = float(model_config["transformer"]["encoder_hidden"]) ** -0.5

    def step_and_update_lr(self):
        self.current_step += 1
        lr = self._update_learning_rate()
        self._optimizer.step()
        return lr

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, d):
        self._optimizer.load_state_dict(d)

    def state_dict(self):
        return self._optimizer.state_dict()

    def get_lr(self):
        for param_group in self._optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.current_step, self.n_warmup_steps
        if n_steps == 0:
            return 0.0
        return np.min([n_steps ** -0.5, n_steps * (n_warmup_steps ** -1.5)])

    def _update_learning_rate(self):
        lr = self.init_lr * self._get_lr_scale()

        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        return lr

