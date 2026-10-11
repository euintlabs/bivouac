export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rng) {
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export function brain(weights) {
  const { W1, b1, W2, b2 } = weights;
  const H = W1.length, O = W2.length, hidden = new Float64Array(H), out = new Float64Array(O);
  return (input) => {
    for (let j = 0; j < H; j++) {
      let s = b1[j];
      const w = W1[j];
      for (let k = 0; k < input.length; k++) s += w[k] * input[k];
      hidden[j] = Math.tanh(s);
    }
    for (let i = 0; i < O; i++) {
      let s = b2[i];
      const w = W2[i];
      for (let j = 0; j < H; j++) s += w[j] * hidden[j];
      out[i] = Math.tanh(s);
    }
    return out;
  };
}

export class OnePlusOneES {
  constructor({ dims, rng, window = 120, sigma0 = 0.28, sigmaMin = 0.05, decay = 40, bound = 1 }) {
    this.rng = rng;
    this.window = window;
    this.sigma0 = sigma0;
    this.sigmaMin = sigmaMin;
    this.decay = decay;
    this.bound = bound;
    this.best = new Array(dims).fill(0);
    this.theta = this.best.slice();
    this.bestReward = -Infinity;
    this.acc = 0;
    this.n = 0;
    this.generations = 0;
  }

  sigma() {
    return this.sigma0 * Math.exp(-this.generations / this.decay) + this.sigmaMin;
  }

  observe(reward) {
    this.acc += reward;
    this.n++;
    if (this.n < this.window) return false;
    const avg = this.acc / this.n;
    this.acc = 0;
    this.n = 0;
    if (avg > this.bestReward) {
      this.bestReward = avg;
      this.best = this.theta.slice();
    }
    const s = this.sigma();
    this.theta = this.best.map((v) => clamp(v + gauss(this.rng) * s, -this.bound, this.bound));
    this.generations++;
    return true;
  }
}

export function gains(theta, spec) {
  const g = {};
  for (const [name, i, scale, lo, hi, base = 1] of spec) g[name] = clamp(base + scale * theta[i], lo, hi);
  return g;
}
