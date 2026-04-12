# -*- coding: utf-8 -*-
"""
Benchmark #3 — nonlinear reward, linear student (model misspecification).

The true reward is a small MLP of features, not a linear function. Both
DPO and ADPO train a *linear* reward head, so neither can fit the Bayes-
optimal decision boundary. This tests whether ADPO's pseudo-label story
still works when the student is misspecified: the model's confident
predictions may actually be confidently *wrong* in regions where the
nonlinear reward disagrees with a linear fit, which could make
pseudo-labels toxic.

If ADPO still wins, the mechanism is robust. If ADPO flattens or loses,
it's a contradicting regime -- the advantage was depending on model
specification being right.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

D = 16
HIDDEN = 32
N_SEEDS = 20
N_TEST = 3000
LR = 0.1
GAMMA = 1.3
REWARD_SCALE = 0.5
MAX_STEPS = 30000
QUERY_BUDGETS = np.arange(0, 520, 20)


class TrueReward(nn.Module):
  def __init__(self, d, h):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

  def forward(self, x):
    return self.net(x).squeeze(-1)


class LinearReward(nn.Module):
  def __init__(self, d):
    super().__init__()
    self.w = nn.Linear(d, 1, bias=False)

  def forward(self, x):
    return self.w(x).squeeze(-1)


def make_true_reward(seed):
  torch.manual_seed(seed + 10_000)
  return TrueReward(D, HIDDEN).eval()


def reward(true_model, x_np):
  with torch.no_grad():
    r = true_model(torch.tensor(x_np, dtype=torch.float32)).item()
  return r * REWARD_SCALE


def bt_label(r1, r2, rng):
  p = 1.0 / (1.0 + np.exp(-(r1 - r2)))
  return 1 if rng.random() < p else -1


def make_test_set(true_model, rng):
  x1 = rng.standard_normal((N_TEST, D)).astype(np.float32)
  x2 = rng.standard_normal((N_TEST, D)).astype(np.float32)
  with torch.no_grad():
    r1 = true_model(torch.tensor(x1)).numpy()
    r2 = true_model(torch.tensor(x2)).numpy()
  y = np.where(r1 > r2, 1, -1).astype(np.float32)
  return (torch.tensor(x1), torch.tensor(x2), torch.tensor(y))


@torch.no_grad()
def test_accuracy(model, test_set):
  x1, x2, y = test_set
  pred = torch.where(model(x1) > model(x2), 1.0, -1.0)
  return (pred == y).float().mean().item()


def run(mode, seed):
  rng = np.random.default_rng(seed)
  torch.manual_seed(seed)
  true_model = make_true_reward(seed)
  test_set = make_test_set(true_model, rng)
  model = LinearReward(D)
  opt = optim.SGD(model.parameters(), lr=LR)

  k_target_idx = 0
  out = np.full(len(QUERY_BUDGETS), np.nan)
  if QUERY_BUDGETS[0] == 0:
    out[0] = test_accuracy(model, test_set); k_target_idx = 1

  q = 0
  for _ in range(MAX_STEPS):
    if k_target_idx >= len(QUERY_BUDGETS):
      break
    x1 = rng.standard_normal(D).astype(np.float32)
    x2 = rng.standard_normal(D).astype(np.float32)
    r1, r2 = reward(true_model, x1), reward(true_model, x2)
    x1_t = torch.tensor(x1); x2_t = torch.tensor(x2)
    s1, s2 = model(x1_t), model(x2_t)
    margin = (s1 - s2).item()

    if mode == 'dpo':
      label = bt_label(r1, r2, rng); q += 1
    elif mode == 'adpo':
      if abs(margin) > GAMMA:
        label = 1 if margin > 0 else -1
      else:
        label = bt_label(r1, r2, rng); q += 1
    else:
      if abs(margin) > GAMMA:
        continue
      label = bt_label(r1, r2, rng); q += 1

    y = torch.tensor(label, dtype=torch.float32)
    loss = -torch.nn.functional.logsigmoid(y * (s1 - s2))
    opt.zero_grad(); loss.backward(); opt.step()

    while k_target_idx < len(QUERY_BUDGETS) and q >= QUERY_BUDGETS[k_target_idx]:
      out[k_target_idx] = test_accuracy(model, test_set); k_target_idx += 1

  last = out[~np.isnan(out)][-1] if np.any(~np.isnan(out)) else 0.5
  out[np.isnan(out)] = last
  return out


def aggregate(mode):
  curves = np.stack([run(mode, s) for s in range(N_SEEDS)], axis=0)
  return curves.mean(0), curves.std(0) / np.sqrt(N_SEEDS)


def main():
  modes = [('dpo', 'DPO', 'black', 'v', '-'),
           ('adpo', 'ADPO (γ=%.1f)' % GAMMA, 'red', 'o', '-'),
           ('adpo_no_pl', 'Active only (no PL)', 'gray', 's', '--')]
  plt.figure(figsize=(7, 4.5))
  for mode, label, color, marker, ls in modes:
    mean, se = aggregate(mode)
    plt.plot(QUERY_BUDGETS, mean * 100, marker=marker, ms=4, color=color,
             linestyle=ls, linewidth=1.8, label=label)
    plt.fill_between(QUERY_BUDGETS, (mean - se) * 100, (mean + se) * 100,
                     color=color, alpha=0.12)
  plt.xlabel('Number of Queries (k)')
  plt.ylabel('Test accuracy (%)')
  plt.title('Nonlinear reward, linear student (misspecified)\n'
            '%d seeds, γ=%.1f' % (N_SEEDS, GAMMA))
  plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
  plt.savefig('bench_nonlinear_reward.png', dpi=150)
  print('Saved bench_nonlinear_reward.png')


if __name__ == '__main__':
  main()
