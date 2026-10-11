import numpy as np


def sample_task(rng, min_types=3, max_types=6):
    types = int(rng.integers(min_types, max_types + 1))
    temper = int(rng.integers(0, 3))
    return dict(
        types=types,
        temper=temper,
        k_pick=[0.14, 0.10, 0.07][temper],
        k_drop=[0.20, 0.16, 0.12][temper],
        spread=int(rng.integers(2, 5)),
        hues=np.sort(rng.uniform(0, 360, types)),
    )


class ColonyEnv:
    OBS = 11
    ACT = 5

    def __init__(self, grid=56, ants=180, steps=150, radius=2, seed=0,
                 reward_weights=(0.25, 0.25, 0.25, 0.25), target_cover=0.45, target_offset=0.34):
        self.gw = self.gh = grid
        self.N, self.steps, self.r = ants, steps, radius
        self.rw = np.asarray(reward_weights, float)
        self.target_cover, self.target_offset = target_cover, target_offset
        self.rng = np.random.default_rng(seed)
        y = np.arange(self.gh); x = np.arange(self.gw)
        self.Y0 = np.clip(y - radius, 0, self.gh); self.Y1 = np.clip(y + radius + 1, 0, self.gh)
        self.X0 = np.clip(x - radius, 0, self.gw); self.X1 = np.clip(x + radius + 1, 0, self.gw)
        self.cnt = ((self.Y1 - self.Y0)[:, None] * (self.X1 - self.X0)[None, :]).astype(np.float32)
        self.yy, self.xx = np.mgrid[0:self.gh, 0:self.gw]
        self.cx, self.cy = self.gw / 2.0, self.gh / 2.0
        self.R = min(self.gw, self.gh) * 0.42
        self.disc = np.hypot(self.xx - self.cx, self.yy - self.cy) <= self.R

    def _box(self, layers):
        k = layers.shape[0]
        ii = np.zeros((k, self.gh + 1, self.gw + 1), np.float32)
        ii[:, 1:, 1:] = np.cumsum(np.cumsum(layers, 1), 2)
        Y0, Y1, X0, X1 = self.Y0[:, None], self.Y1[:, None], self.X0[None, :], self.X1[None, :]
        return ii[:, Y1, X1] - ii[:, Y0, X1] - ii[:, Y1, X0] + ii[:, Y0, X0]

    def _layers(self, grid):
        T = self.task["types"]
        layers = np.empty((T + 1, self.gh, self.gw), np.float32)
        for t in range(T):
            layers[t] = grid == t
        layers[T] = grid >= 0
        return layers

    def reset(self, task):
        self.task = task
        T = task["types"]
        self.band = self.R / T * 0.85 / (0.7 + 0.6 * task["spread"] / 4.0)
        self.ring = ((np.arange(T) + 0.62) / T) * self.R
        self.cover_cap = 0.6
        self.grid = -np.ones((self.gh, self.gw), np.int16)
        w = np.arange(1, T + 1, dtype=float); w /= w.sum()
        kinds = (self.rng.random((self.gh, self.gw))[..., None] < np.cumsum(w)).argmax(-1).astype(np.int16)
        place = self.disc & (self.rng.random((self.gh, self.gw)) < 0.72)
        self.grid[place] = kinds[place]
        ang = self.rng.uniform(0, 2 * np.pi, self.N); rad = self.rng.uniform(0, self.R, self.N)
        self.ax = np.clip((self.cx + np.cos(ang) * rad).astype(int), 0, self.gw - 1)
        self.ay = np.clip((self.cy + np.sin(ang) * rad).astype(int), 0, self.gh - 1)
        self.carry = -np.ones(self.N, np.int16)
        self.wx = np.zeros(self.N); self.wy = np.zeros(self.N)

    def observe(self):
        task, T = self.task, self.task["types"]
        S = self._box(self._layers(self.grid))
        dens, occ = S[:T] / self.cnt, S[T] / self.cnt
        cell = self.grid[self.ay, self.ax]
        kind = np.where(self.carry >= 0, self.carry, cell)
        valid = kind >= 0
        idx = np.clip(kind, 0, T - 1)
        f = np.zeros(self.N)
        f[valid] = dens[idx[valid], self.ay[valid], self.ax[valid]]
        dist = np.hypot(self.ax - self.cx, self.ay - self.cy)
        a = np.zeros(self.N)
        a[valid] = np.exp(-((dist[valid] - self.ring[idx[valid]]) / self.band) ** 2)
        d = np.maximum(dist, 1.0)
        obs = np.stack([
            (self.carry >= 0).astype(float), (self.carry + 1) / (T + 1), f, a,
            -(self.ax - self.cx) / d, -(self.ay - self.cy) / d, np.minimum(1.0, dist / self.R),
            np.zeros(self.N), np.full(self.N, T / 6.0), np.full(self.N, task["temper"] / 2.0),
            np.full(self.N, task["spread"] / 4.0)], 1)
        return obs, dict(f=f, a=a, occ=occ[self.ay, self.ax], cell=cell, dist=dist)

    def step(self, act, ctx):
        task, T, rng, gw = self.task, self.task["types"], self.rng, self.gw
        f, a, occ, cell, dist = ctx["f"], ctx["a"], ctx["occ"], ctx["cell"], ctx["dist"]
        ax, ay, grid = self.ax, self.ay, self.grid

        free = (self.carry < 0) & (cell >= 0)
        p_pick = (task["k_pick"] / (task["k_pick"] + f)) ** 2 * (0.35 + 0.65 * (1 - a)) * np.clip(1 + 0.45 * act[:, 0], 0.4, 1.75)
        pick = free & (rng.random(self.N) < np.clip(p_pick, 0, 0.99))
        if pick.any():
            ai = np.where(pick)[0]
            _, first = np.unique(ay[ai] * gw + ax[ai], return_index=True)
            keep = ai[first]
            self.carry[keep] = grid[ay[keep], ax[keep]]
            grid[ay[keep], ax[keep]] = -1

        holding = (self.carry >= 0) & (grid[ay, ax] < 0)
        p_drop = (f / (task["k_drop"] + f)) ** 2 * (0.18 + 0.82 * a) * np.clip(1 + 0.45 * act[:, 1], 0.4, 1.75)
        crowded = occ > self.cover_cap
        p_discard = np.clip(1.5 * (occ - self.cover_cap), 0, 0.9) * np.clip(0.5 + 0.5 * act[:, 4], 0, 1)
        discard = holding & crowded & (rng.random(self.N) < p_discard)
        self.carry[discard] = -1
        drop = holding & ~discard & (rng.random(self.N) < np.clip(p_drop, 0, 0.99))
        if drop.any():
            ai = np.where(drop)[0]
            _, first = np.unique(ay[ai] * gw + ax[ai], return_index=True)
            keep = ai[first]
            grid[ay[keep], ax[keep]] = self.carry[keep]
            self.carry[keep] = -1

        loaded = self.carry >= 0
        d = np.maximum(dist, 1.0)
        side = np.where(self.ring[np.clip(self.carry, 0, T - 1)] > d, 1.0, -1.0)
        heading = np.arctan2((ay - self.cy) / d * side, (ax - self.cx) / d * side) + act[:, 2] * 0.55
        self.wx = np.where(loaded, self.wx, self.wx * 0.7 + (rng.random(self.N) - 0.5) * 0.9)
        self.wy = np.where(loaded, self.wy, self.wy * 0.7 + (rng.random(self.N) - 0.5) * 0.9)
        sx = np.where(loaded, np.cos(heading), self.wx) + rng.standard_normal(self.N) * 0.6
        sy = np.where(loaded, np.sin(heading), self.wy) + rng.standard_normal(self.N) * 0.6
        self.ax = np.clip(ax + np.where(sx > 0.33, 1, np.where(sx < -0.33, -1, 0)), 0, gw - 1)
        self.ay = np.clip(ay + np.where(sy > 0.33, 1, np.where(sy < -0.33, -1, 0)), 0, self.gh - 1)

    def reward(self):
        grid, task, T = self.grid, self.task, self.task["types"]
        occupied = grid >= 0
        if not occupied.any():
            return 0.0, dict(order=0.0, space=0.0, balance=0.0, harmony=0.0)
        S = self._box(self._layers(grid))
        same = np.zeros((self.gh, self.gw), np.float32)
        for t in range(T):
            same += np.where(grid == t, S[t], 0)
        order = float((same[occupied] / self.cnt[occupied]).mean())
        cover = float(occupied[self.disc].sum()) / max(1, int(self.disc.sum()))
        space = float(np.clip(1 - abs(cover - self.target_cover) / self.target_cover, 0, 1))
        off = np.hypot(self.xx[occupied].mean() - self.cx, self.yy[occupied].mean() - self.cy) / self.R
        balance = float(np.clip(1 - abs(off - self.target_offset) / self.target_offset, 0, 1))
        diffs = []
        for A, B in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
            edge = (A >= 0) & (B >= 0) & (A != B)
            if edge.any():
                hd = np.abs(task["hues"][A[edge]] - task["hues"][B[edge]])
                diffs.append(np.minimum(hd, 360 - hd) / 180.0)
        harmony = float(np.clip(1 - np.concatenate(diffs).mean(), 0, 1)) if diffs else 0.5
        parts = dict(order=order, space=space, balance=balance, harmony=harmony)
        return float(self.rw @ np.array([order, space, balance, harmony])), parts

    def rollout(self, policy, th, task, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset(task)
        for _ in range(self.steps):
            obs, ctx = self.observe()
            self.step(policy.forward(th, obs), ctx)
        return self.reward()[0]
