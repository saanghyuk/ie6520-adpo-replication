# -*- coding: utf-8 -*-
"""
Benchmark #5 -- cumulative-regret curves.

Theorem 5.1 says APPO has Regret(T) = O~(d^2 / Delta), constant in T. We
cannot test that exact statement on a continuous-feature toy (Delta is not
strictly positive when x1, x2 are Gaussian), but we can test the *shape*:
ADPO should have a much flatter cumulative-regret slope than DPO once the
model has converged.

We define the per-step regret on each sampled pair (x1, x2) as
    Regret_t = max(r(x1), r(x2)) - r(y_t^1),
where y_t^1 is the action the model thinks is better, i.e.
    y_t^1 = argmax_{y in {x1, x2}} score_theta(y).
Cumulative regret is the sum over t.

Output: bench_cumulative_regret.png
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

D = 16
N_SEEDS = 20
LR = 0.1
GAMMA = 1.3
REWARD_SCALE = 0.5
N_STEPS = 3000


def sample_x(rng):
  return rng.standard_normal(D)


def bt_label(r1, r2, rng):
  p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
  return 1 if rng.random() < p else -1


class LinearReward(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.Linear(d, 1, bias=False)

  def forward(self, x):
    return self.w(x).squeeze(-1)


def run(mode, seed):
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  theta_star = rng.standard_normal(D) * REWARD_SCALE
  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  cum_regret = np.zeros(N_STEPS)
  running = 0.0

  for t in range(N_STEPS):
    x1, x2 = sample_x(rng), sample_x(rng)
    r1 = float(theta_star @ x1)
    r2 = float(theta_star @ x2)
    r_star = max(r1, r2)
    x1_t = torch.tensor(x1, dtype=torch.float32)
    x2_t = torch.tensor(x2, dtype=torch.float32)
    s1, s2 = model(x1_t), model(x2_t)
    margin = (s1 - s2).item()

    chosen_r = r1 if margin > 0 else r2
    running += (r_star - chosen_r)
    cum_regret[t] = running

    if mode == 'dpo':
      label = bt_label(r1, r2, rng)
    elif mode == 'adpo':
      if abs(margin) > GAMMA:
        label = 1 if margin > 0 else -1
      else:
        label = bt_label(r1, r2, rng)
    elif mode == 'adpo_no_pl':
      if abs(margin) > GAMMA:
        continue
      label = bt_label(r1, r2, rng)
    else:
      raise ValueError(mode)

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad(); loss.backward(); opt.step()

  return cum_regret


def main():
  print(f'Running {N_SEEDS} seeds, {N_STEPS} steps each, three modes...')
  modes = [('dpo', 'DPO', 'black', '-'),
           ('adpo', f'ADPO ($\\gamma={GAMMA}$)', 'red', '-'),
           ('adpo_no_pl', 'Active only (no PL)', 'gray', '--')]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

  for mode, label, color, ls in modes:
    curves = np.stack([run(mode, s) for s in range(N_SEEDS)])
    mean = curves.mean(0)
    se = curves.std(0) / np.sqrt(N_SEEDS)
    t = np.arange(1, N_STEPS + 1)

    ax1.plot(t, mean, color=color, linestyle=ls, linewidth=1.8, label=label)
    ax1.fill_between(t, mean - se, mean + se, color=color, alpha=0.15)

    inst = np.gradient(mean)
    smooth = np.convolve(inst, np.ones(50) / 50, mode='same')
    ax2.plot(t, smooth, color=color, linestyle=ls, linewidth=1.8, label=label)

  ax1.set_xlabel('Training step $t$')
  ax1.set_ylabel(r'Cumulative regret  $\sum_{s\leq t} [r^*(x_s) - r(x_s, y_s^1)]$')
  ax1.set_title('Cumulative regret')
  ax1.legend(loc='upper left')
  ax1.grid(alpha=0.3)

  ax2.set_xlabel('Training step $t$')
  ax2.set_ylabel('Per-step regret (50-step smoothed)')
  ax2.set_title('Instantaneous regret -- the slope of the left panel')
  ax2.legend(loc='upper right')
  ax2.grid(alpha=0.3)

  fig.suptitle(f'Regret comparison: $d={D}$, BT noise scale {REWARD_SCALE}, '
               f'{N_SEEDS} seeds', y=1.02)
  fig.tight_layout()
  fig.savefig('bench_cumulative_regret.png', dpi=150, bbox_inches='tight')
  print('Saved bench_cumulative_regret.png')


if __name__ == '__main__':
  main()
