# bivouac

a small brain for ant colonies that build things.

a bivouac is the living nest army ants make out of their own bodies. this repo is the engine we used to train the ants behind geodesics ii colonies on octra. it has the general parts only: the policy, the environment, the trainer and the browser runtime. the trained weights, the reward tuning and the art are not here.

## how it works

every ant runs the same tiny network, 11 inputs, 8 hidden, 5 outputs, tanh all the way. inputs are what one ant can sense at its cell: am i carrying, what kind, how dense is my kind around me, how close am i to where my kind belongs, which way is the centre, plus a few colony-wide knobs. outputs nudge the classic deneubourg pick and drop rules, steer the walk, and decide when to give up on a crowded spot.

nothing is told where to put anything. grains get sorted because each ant follows local rules and reads the marks the others left behind. that is stigmergy, the same thing real ants do.

the environment scores a finished colony on four things: order (same kinds sit together), space (how much of the disc gets covered), balance (where the mass sits) and harmony (how well neighbouring colours get along). you choose the weights. defaults are flat.

training is antithetic evolution strategies. sample noise, try the policy pushed both ways, keep the direction that scored better, repeat across random colony setups so the brain generalises instead of memorising one map. no gradients, no autodiff, numpy only.

in the browser the brain keeps learning. each colony runs a (1+1)-es adapter over a handful of gains: mutate, watch the colony for a window, keep the change if the reward went up, shrink the step size over time. so two colonies with the same weights still drift into their own habits.

## layout

```
python/
  policy.py     mlp, pack and unpack, export to json
  env.py        stigmergy colony env and the four-part reward
  train_es.py   antithetic es trainer
js/
  bivouac.js    forward pass, (1+1)-es adapter, seeded rng, gaussian, gains mapping
examples/
  browser.html  minimal canvas demo running a trained brain
```

## train

```
cd python
pip install numpy
python train_es.py --iters 26 --pairs 8 --sigma 0.12 --lr 0.06 --weights 0.25 0.25 0.25 0.25
```

writes `weights.json` with the architecture, the score against the zero baseline, and the weights in the shape the js runtime reads.

## run

```js
import { brain, OnePlusOneES, gains, mulberry32 } from './js/bivouac.js'

const net = brain(trained.weights)
const rng = mulberry32(42)
const es = new OnePlusOneES({ dims: 3, rng, window: 120 })

const out = net(observation)
const g = gains(es.theta, [['pick', 0, 0.4, 0.5, 1.6], ['drop', 1, 0.4, 0.5, 1.6], ['wander', 2, 0.5, 0.3, 1.8]])
es.observe(reward)
```

open `examples/browser.html` through any static server after you train, it loads `python/weights.json`.

## credit

the idea of training small creatures in the browser comes from [algovivo](https://github.com/juniorrojas/algovivo) by junior rojas. bivouac is our own take for ant colonies, written from scratch.

## license

mit
