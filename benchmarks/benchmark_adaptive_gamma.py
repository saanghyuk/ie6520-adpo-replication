# -*- coding: utf-8 -*-
"""
Benchmark #6 -- adaptive-gamma via running quantile of |margin|.

Addresses Prof. Wang Chi's feedback on choosing the threshold without prior
knowledge of the gap Delta.

Idea: instead of a fixed gamma, set gamma_t at each step to a running
quantile of the model's recent |margin| values, with a floor to keep us out
of the failure regime where pseudo-labels dominate before the model is
reliable. Concretely we use the q-th percentile (default q = 50) of the
last K = 200 absolute margins, lower-bounded by gamma_min = 0.5.

This produces a self-tuning threshold that does not require knowledge of
Delta. We compare it against the paper's fixed gamma = 1.3, the failure
regime gamma = 0.1, and DPO.

Output: bench_adaptive_gamma.png
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

D = 16
N_SEEDS = 20
LR = 0.1
REWARD_SCALE = 0.5
N_TEST = 3000
QUERY_BUDGETS = np.arange(0, 320, 20)
GAMMA_MIN = 0.5
QUANTILE = 50         # 50th percentile (median)
WINDOW = 200          # rolling window for the quantile


def sample_x(rng, n=None):
  if n is None:
    return rng.standard_normal(D)
  return rng.standard_normal((n, D))


def bt_label(r1, r2, rng):
  p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
  return 1 if rng.random() < p else -1


def make_test_set(theta_star, rng):
  x1 = sample_x(rng, N_TEST)
  x2 = sample_x(rng, N_TEST)
  r1 = x1 @ theta_star
  r2 = x2 @ theta_star
  y = np.where(r1 > r2, 1, -1)
  return (torch.tensor(x1, dtype=torch.float32),
          torch.tensor(x2, dtype=torch.float32),
          torch.tensor(y, dtype=torch.float32))


class LinearReward(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.Linear(d, 1, bias=False)

  def forward(self, x):
    return self.w(x).squeeze(-1)


@torch.no_grad()
def test_accuracy(model, test_set):
  x1, x2, y = test_set
  pred = torch.where(model(x1) > model(x2), 1.0, -1.0)
  return (pred == y).float().mean().item()


def run(mode, seed):
  """mode in {'dpo', 'adpo_g13', 'adpo_g01', 'adpo_adaptive'}."""
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  theta_star = rng.standard_normal(D) * REWARD_SCALE
  test_set = make_test_set(theta_star, rng)
  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  margins = deque(maxlen=WINDOW)
  k_target_idx = 0
  out = np.full(len(QUERY_BUDGETS), np.nan)
  if QUERY_BUDGETS[0] == 0:
    out[0] = test_accuracy(model, test_set); k_target_idx = 1

  q = 0
  max_steps = 30000
  for _ in range(max_steps):
    if k_target_idx >= len(QUERY_BUDGETS):
      break
    x1, x2 = sample_x(rng), sample_x(rng)
    r1 = float(theta_star @ x1)
    r2 = float(theta_star @ x2)
    x1_t = torch.tensor(x1, dtype=torch.float32)
    x2_t = torch.tensor(x2, dtype=torch.float32)
    s1, s2 = model(x1_t), model(x2_t)
    margin = (s1 - s2).item()
    margins.append(abs(margin))

    if mode == 'dpo':
      gamma_t = float('inf')
    elif mode == 'adpo_g13':
      gamma_t = 1.3
    elif mode == 'adpo_g01':
      gamma_t = 0.1
    elif mode == 'adpo_adaptive':
      if len(margins) < 50:
        gamma_t = float('inf')                 # warmup -> always query
      else:
        gamma_t = max(GAMMA_MIN,
                      float(np.percentile(margins, QUANTILE)))
    else:
      raise ValueError(mode)

    if abs(margin) > gamma_t:
      label = 1 if margin > 0 else -1
    else:
      label = bt_label(r1, r2, rng); q += 1

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad(); loss.backward(); opt.step()

    while k_target_idx < len(QUERY_BUDGETS) and q >= QUERY_BUDGETS[k_target_idx]:
      out[k_target_idx] = test_accuracy(model, test_set)
      k_target_idx += 1

  last = out[~np.isnan(out)][-1] if np.any(~np.isnan(out)) else 0.5
  out[np.isnan(out)] = last
  return out


def aggregate(mode):
  curves = np.stack([run(mode, s) for s in range(N_SEEDS)])
  return curves.mean(0), curves.std(0) / np.sqrt(N_SEEDS)


def main():
  print(f'Running {N_SEEDS} seeds, four configurations...')
  modes = [
    ('dpo',           'DPO baseline',                              'black', 'v',  '-'),
    ('adpo_g13',      'ADPO, fixed $\\gamma=1.3$ (paper default)', 'red',   'o',  '-'),
    ('adpo_g01',      'ADPO, fixed $\\gamma=0.1$ (failure regime)','tab:orange','x','-'),
    ('adpo_adaptive', 'ADPO, adaptive $\\gamma_t = $ median$|m|$', 'tab:blue','s','--'),
  ]

  plt.figure(figsize=(7.5, 4.8))
  for mode, label, color, marker, ls in modes:
    mean, se = aggregate(mode)
    plt.plot(QUERY_BUDGETS, mean * 100, marker=marker, ms=4, color=color,
             linestyle=ls, linewidth=1.8, label=label)
    plt.fill_between(QUERY_BUDGETS, (mean - se) * 100, (mean + se) * 100,
                     color=color, alpha=0.12)

  plt.xlabel('Number of queries (k)')
  plt.ylabel('Test accuracy (%)')
  plt.title(f'Adaptive $\\gamma$ (running quantile) vs fixed $\\gamma$\n'
            f'($d={D}$, BT noise scale {REWARD_SCALE}, {N_SEEDS} seeds)')
  plt.legend(loc='lower right', fontsize=9)
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig('bench_adaptive_gamma.png', dpi=150)
  print('Saved bench_adaptive_gamma.png')


if __name__ == '__main__':
  main()
