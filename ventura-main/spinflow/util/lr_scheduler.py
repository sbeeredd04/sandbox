import numpy as np

import math
import torch
import matplotlib.pyplot as plt

class HalfCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    One‑cycle schedule that

    1. linearly warms the learning‑rate from 0 → base_lr over `warmup_steps`;
    2. then follows a half‑cosine decay from base_lr → 0 over the remaining
       `total_steps - warmup_steps` steps.

    Works with any optimizer that PyTorch supports.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 total_steps: int,
                 warmup_steps: int = 0,
                 last_epoch: int = -1):
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch  # step index *after* scheduler.step() increments
        factors = []

        for base_lr in self.base_lrs:
            # ---- linear warm‑up -----------------------------------------
            if step_num < self.warmup_steps:
                factor = float(step_num) / float(max(1, self.warmup_steps))
            else:
                # ---- half‑cosine decay ----------------------------------
                progress = (step_num - self.warmup_steps) / float(
                    max(1, self.total_steps - self.warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)  # clamp
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            factors.append(base_lr * factor)
        return factors

class IterExponential:
    def __init__(self, total_iter_length, final_ratio, warmup_steps=0) -> None:
        """
        Customized iteration-wise exponential scheduler.
        Re-calculate for every step, to reduce error accumulation

        Args:
            total_iter_length (int): Expected total iteration number
            final_ratio (float): Expected LR ratio at n_iter = total_iter_length
        """
        self.total_length = total_iter_length
        self.effective_length = total_iter_length - warmup_steps
        self.final_ratio = final_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, n_iter) -> float:
        if n_iter < self.warmup_steps:
            alpha = 1.0 * n_iter / self.warmup_steps
        elif n_iter >= self.total_length:
            alpha = self.final_ratio
        else:
            actual_iter = n_iter - self.warmup_steps
            alpha = np.exp(
                actual_iter / self.effective_length * np.log(self.final_ratio)
            )
        return alpha

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 1.  IterExponential schedule (exactly what you had before)
    # ------------------------------------------------------------------ #
    total_steps = 25_000
    warmup_steps = 200
    iter_sched = IterExponential(
        total_iter_length=total_steps,
        final_ratio=0.01,
        warmup_steps=warmup_steps,
    )
    alpha_iter = [iter_sched(i) for i in range(total_steps)]

    # ------------------------------------------------------------------ #
    # 2.  Half‑Cosine schedule with the same total & warm‑up
    # ------------------------------------------------------------------ #
    dummy_param = torch.nn.Parameter(torch.zeros(1))        # fake model param
    optimizer = torch.optim.AdamW([dummy_param], lr=1.0)    # base‑LR = 1.0
    cosine_sched = HalfCosineLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    alpha_cosine = []
    for _ in range(total_steps):
        optimizer.step()        # normally you would zero‑grad + loss.backward()
        cosine_sched.step()     # update LR
        alpha_cosine.append(optimizer.param_groups[0]["lr"])

    # ------------------------------------------------------------------ #
    # 3.  Plot & save
    # ------------------------------------------------------------------ #
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    steps = np.arange(total_steps)

    plt.plot(steps, alpha_iter,   label="IterExponential")
    plt.plot(steps, alpha_cosine, label="HalfCosineLR")
    plt.xlabel("Step")
    plt.ylabel("LR multiplier / LR")
    plt.title("LR schedules: IterExponential vs Half‑Cosine with warm‑up")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lr_scheduler.png")        # writes a file