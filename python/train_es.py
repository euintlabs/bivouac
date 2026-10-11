import argparse
import json
import time

import numpy as np

from env import ColonyEnv, sample_task
from policy import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=26)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--weights", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="weights.json")
    a = ap.parse_args()

    env = ColonyEnv(seed=0, reward_weights=a.weights)
    net = MLP(ColonyEnv.OBS, a.hidden, ColonyEnv.ACT)
    held_out = [sample_task(np.random.default_rng(1000 + i)) for i in range(4)]

    def score(th, tasks, base):
        return float(np.mean([env.rollout(net, th, t, seed=base + i) for i, t in enumerate(tasks)]))

    def validate(th):
        total, parts = 0.0, dict(order=0.0, space=0.0, balance=0.0, harmony=0.0)
        for i, t in enumerate(held_out):
            env.rollout(net, th, t, seed=50000 + i)
            r, p = env.reward()
            total += r
            for k in parts:
                parts[k] += p[k] / len(held_out)
        return total / len(held_out), parts

    rng = np.random.default_rng(a.seed)
    th = np.zeros(net.nparam)
    baseline, parts = validate(th)
    print(f"baseline (theta=0) {baseline:.4f} {({k: round(v, 3) for k, v in parts.items()})}")
    best, best_th = baseline, th.copy()
    t0 = time.time()
    for it in range(a.iters):
        tasks = [sample_task(np.random.default_rng(200 + it * 13 + j)) for j in range(a.tasks)]
        base = 900000 + it * 100
        eps = rng.standard_normal((a.pairs, net.nparam))
        up = np.array([score(th + a.sigma * e, tasks, base) for e in eps])
        down = np.array([score(th - a.sigma * e, tasks, base) for e in eps])
        d = up - down
        if d.std() > 1e-8:
            d = d / (d.std() + 1e-8)
        th = th + a.lr * (d[:, None] * eps).sum(0) / a.pairs
        if it % 2 == 0 or it == a.iters - 1:
            v, parts = validate(th)
            if v > best:
                best, best_th = v, th.copy()
            print(f"it {it:2d} val {v:.4f} best {best:.4f} {({k: round(x, 2) for k, x in parts.items()})} ({time.time() - t0:.0f}s)")

    out = dict(arch=dict(nin=net.nin, hidden=net.hidden, nout=net.nout), reward_weights=a.weights,
               val_reward=round(best, 4), baseline=round(baseline, 4), weights=net.export(best_th))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"best {best:.4f} vs baseline {baseline:.4f}, wrote {a.out}")


if __name__ == "__main__":
    main()
