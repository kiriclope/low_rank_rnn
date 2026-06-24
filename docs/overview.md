# Project Overview

## Research question

How does the geometry of learned representations in a low-rank RNN support (or fail to
support) sequential multi-task learning?

Concretely: a rank-2 RNN learns three tasks in order:

1. **DPA** (Delayed Paired Association) — working memory task. The network must hold a
   sample stimulus (A or B) across a delay and compare it to a test stimulus (C or D) to
   produce a paired / unpaired decision.

2. **GNG** (Go / No-Go) — action selection task. The network must respond to a go cue and
   withhold response to a nogo cue.

3. **Dual** — both tasks simultaneously, interleaved in a single trial.

The core question is whether and how the network retains DPA after learning GNG, and
whether it can then perform both tasks jointly. The forgetting metric is
**`after_gng/dpa` accuracy** — how much DPA survives GNG training given the freezing
protocol.

## The κ-plane

With rank-2 factorisation `W_rec = m n^T / N`, the network's collective state reduces to
two scalar projections:

```
κ₀ = n₀^T rates / N   (memory rank)
κ₁ = n₁^T rates / N   (decision / readout rank)
```

All analysis (fixed points, flow fields, trajectories) is done in the (κ₀, κ₁) plane.

## Structural hypothesis

- **DPA** is solved by a **ring attractor** in κ₀: two stable fixed points at ±κ* encode
  the A and B samples. The ring persists through the delay and is read out by κ₁ at test
  time.
- **GNG** is solved by **input-driven attractors** in κ₁: go input pushes κ₁ positive,
  nogo input leaves it near zero.
- **Dual** requires both to coexist: the ring in κ₀ intact, κ₁ responding to go/nogo
  input without destroying the ring.

The ring requires an **odd nonlinearity** (so ±κ* are both FPs) and **saturation** (to
stabilise the FP — gain × φ'(κ*) × λ₀ < 1 at the ring radius).

**Update (2026-06-22) — resolved by the EISTP model.** The vanilla/static-backbone family
confirmed that odd-φ + linear recurrence is symmetry-locked (memory wells stay symmetric about
κ₁=0, never lowered). The win came from porting NeuroFlame's **EI + short-term-plasticity**
network (`EISTPModel`, `model_type="eistp"`): relu (non-odd) rates + a rank-2 low-rank that
*rides on* the facilitating E→E synapses. With the low-rank scaled `/√K` (memory-mode gain ≈ 1,
critical) it gives the first **persistent** κ₀ memory **and** the target **lower-plane NoGo well**
(Go upper). See `docs/architecture.md` (EISTP model) and `docs/ring_lowerplane_log.md` §11.

## Key protection mechanisms

- `freeze_low_rank_cols=[0]` during GNG: freezes rank-0 of m/n (the memory rank), so the
  ring is not destroyed when GNG is learnt.
- `freeze_input_stages=["gng","dual"]`: freezes all input weights during GNG and Dual,
  preventing go-input rotation of n₀.
- `freeze_rank0_dual=True`: additionally freezes rank-0 during Dual.
- `kappa1_reg_weight`: penalises gain × λ₁ > 1 during Dual, keeping κ₁ sub-critical so
  no competing autonomous attractors form.

## Directory layout

```
rnn/
├── src/                    # library package
│   ├── models.py           # LowRankModel
│   ├── tasks.py            # trial generators (DPA, GNG, Dual)
│   ├── train.py            # Optimization loop, loss functions
│   ├── init.py             # structured initialisation
│   └── dynamics.py         # fixed-point finding, flow fields
├── sweep.py                # RunConfig dataclass + sweep runner
├── plot_sweep.py           # main plotting entrypoint
├── rerun_dual.py           # retrain only the Dual stage from checkpoints
├── analyze.py              # load_results, summary_table
├── plots/                  # legacy single-run scripts
├── docs/                   # ← this documentation
└── results/
    ├── dual/               # run dirs: results.jsonl + models/ + run.log
    └── figures/            # plot output
```

## See also

- [Architecture](architecture.md) — model, tasks, training pipeline
- [Running experiments](running.md) — sweep workflow, monitoring, rerunning
- [Analysis & plotting](analysis.md) — plot_sweep.py, flags, figure types
- [Experiment log](experiment_log.md) — all sweeps, configs, findings
- [Nonlinearity investigation](nonlinearities.md) — ring vs attractor theory, results
