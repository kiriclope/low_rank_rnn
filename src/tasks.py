from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TaskTiming:
    stim_on: list[float]
    stim_off: list[float]
    t_steps: float
    dt: float

    @property
    def n_stim_on(self) -> torch.Tensor:
        return torch.tensor([int(t / self.dt) for t in self.stim_on], dtype=torch.long)

    @property
    def n_stim_off(self) -> torch.Tensor:
        return torch.tensor([int(t / self.dt) for t in self.stim_off], dtype=torch.long)

    @property
    def n_steps(self) -> int:
        return int(self.t_steps / self.dt)

    @property
    def x_time(self) -> np.ndarray:
        return np.linspace(0, (self.n_steps - 1) * self.dt, self.n_steps)


def make_timings(dt: float) -> dict:
    """Canonical task timings, shared by sweep.py (training) and plot_sweep.py (plotting).

    Single source of truth — edit here so training inputs and plotted stimulus
    windows / targets never drift apart.
    """
    return {
        "dpa":  TaskTiming([2.0, 8.0],             [3.0, 9.0],            11.0, dt),
        "gng":  TaskTiming([2.0, 4.0],             [3.0, 4.5],             6.0, dt),
        "dual": TaskTiming([2.0, 4.0, 6.0, 8.0],   [3.0, 5.0, 6.5, 9.0],  11.0, dt),
    }


def _attn_window(inputs, timing, input_scale, gated, attention_scale=1.0):
    """Add the tonic attention/context input on the LAST channel.
    gated=False (default): on from the first-stim ONSET to trial end (the original behaviour).
    gated=True: on ONLY over the RETENTION DELAY — from the first-stim OFFSET (n_off[0]) to the
    last-stim ONSET (n_on[-1]), i.e. exactly while the sample memory is held. Off during encoding
    (sample) and off once the probe arrives (test in DPA/Dual, cue in GNG) — a pure working-memory
    maintenance signal, released by the probe.
    attention_scale multiplies the tonic value (default 1.0 = input_scale): the amplitude of the
    readout-plane symmetry-breaking bias b_attn, i.e. the lever that pushes the memory wells' κ₁ down
    (attention-direct term ⟨n₁,φ(g·b_attn)⟩/N). See ring_lowerplane_log §19."""
    n_on, n_off = timing.n_stim_on, timing.n_stim_off
    val = attention_scale * input_scale
    if gated:
        inputs[:, int(n_off[0]):int(n_on[-1]), -1] += val
    else:
        inputs[:, n_on[0]:, -1] += val


def generate_dpa_trials(
    n_trials: int,
    timing: TaskTiming,
    input_size: int = 8,
    target_rank: int = 1,
    noise: float = 0.1,
    baseline_value: float = 0.5,
    input_scale: float = 1.0,
    attention_input: bool = False,
    attention_gated: bool = False,
    attention_scale: float = 1.0,
    windowed_targets: bool = False,
    decay_to_zero: bool = True,
    decay_onesided: bool = False,
    response_in_cue: bool = False,
):
    n_steps = timing.n_steps
    n_on = timing.n_stim_on
    n_off = timing.n_stim_off

    inputs = noise * torch.randn(n_trials, n_steps, input_size)
    targets = torch.zeros(n_trials, n_steps, target_rank) * torch.nan

    if attention_input:   # tonic attentional/context input on the LAST channel,
        _attn_window(inputs, timing, input_scale, attention_gated, attention_scale=attention_scale)

    idx_A = torch.rand(n_trials) > 0.5
    idx_B = ~idx_A
    idx_C = torch.rand(n_trials) > 0.5
    idx_D = ~idx_C

    inputs[idx_A, n_on[0]:n_off[0], 0] += input_scale
    inputs[idx_B, n_on[0]:n_off[0], 1] += input_scale
    inputs[idx_C, n_on[1]:n_off[1], 2] += input_scale
    inputs[idx_D, n_on[1]:n_off[1], 3] += input_scale

    idx_pair = (idx_A & idx_C) | (idx_B & idx_D)

    # baseline
    targets[:, :n_on[0], 0] = 0.0 # on kappa 0
    # no ramping
    targets[:, :n_on[1], -1] = 0.0 # pre-test no-lick on the readout [-1] (κ₁ rank-2, κ₂ rank-3)

    # memory
    targets[idx_A, n_on[0]:n_on[1], 0] = 1.0
    targets[idx_B, n_on[0]:n_on[1], 0] = -1.0    

    # pairing
    if windowed_targets:
        # express 1 s AFTER test-off (a 1 s plateau — longer window so the match/+1 decision can push
        # up against the no-lick bias), then optionally decay to 0 (transient decision)
        half    = int(round(0.5 / timing.dt))
        quarter = int(round(0.25 / timing.dt))   # 0.25 s (shortened pairing window)
        to   = int(n_off[1])
        # response_in_cue: score in the last 0.5 s of the TEST (test ON) so the match decision is
        # test-DRIVEN, not held from memory; else the legacy 0.25 s window starting at test-off.
        r0, r1 = (to - half, to) if response_in_cue else (to, to + quarter)
        targets[idx_pair,  r0:r1, -1] =  1.0
        targets[~idx_pair, r0:r1, -1] = -1.0
        if decay_to_zero:
            targets[:,        r1:, -1] = 0.0   # default: pin the lick to 0
            if decay_onesided:  # pairing decay MARKERS on the LICK ([-1]) → loss scores one-sided at thresh 0
                targets[idx_pair,  r1:, -1] = -0.5   # match: penalise lick>0 (relax the pulse DOWN)
                targets[~idx_pair, r1:, -1] = +0.5   # nonmatch: penalise lick<0 (relax the trace UP)
    else:
        targets[idx_pair,  n_off[1], -1] = 1.0
        targets[~idx_pair, n_off[1], -1] = -1.0

    return inputs, targets


def generate_gng_trials(
    n_trials: int,
    timing: TaskTiming,
    input_size: int = 8,
    target_rank: int = 1,
    noise: float = 0.1,
    baseline_value: float = 0.0,
    cue_on_go_input: bool = False,
    cue_scale: float = 1.0,
    nogo_target: float = 0.0,
    go_target: float = 1.0,
    go_on_rwd_input: bool = False,
    input_scale: float = 1.0,
    attention_input: bool = False,
    attention_gated: bool = False,
    attention_scale: float = 1.0,
    ramping_gng: bool = False,
    windowed_targets: bool = False,
    decay_to_zero: bool = True,
    gng_response: bool = False,
    decay_onesided: bool = False,
    response_in_cue: bool = False,
):
    n_steps = timing.n_steps
    n_on = timing.n_stim_on
    n_off = timing.n_stim_off

    inputs = noise * torch.randn(n_trials, n_steps, input_size)
    targets = torch.zeros(n_trials, n_steps, target_rank) * torch.nan

    if attention_input:   # tonic attention on the LAST channel
        _attn_window(inputs, timing, input_scale, attention_gated, attention_scale=attention_scale)

    idx_go   = torch.rand(n_trials) > 0.5
    idx_nogo = ~idx_go

    if go_on_rwd_input:
        go_ch, ngo_ch = input_size - 1, 4
        inputs[idx_go,   n_on[0]:n_off[0], go_ch]  += input_scale
        inputs[idx_nogo, n_on[0]:n_off[0], ngo_ch] += input_scale
        inputs[:, n_on[1]:n_off[1], go_ch] += cue_scale * input_scale
    else:
        inputs[idx_go,   n_on[0]:n_off[0], 4] += input_scale
        inputs[idx_nogo, n_on[0]:n_off[0], 5] += input_scale

        if cue_on_go_input:
            inputs[:, n_on[1]:n_off[1], 4] += cue_scale * input_scale
        else:
            inputs[:, n_on[1]:n_off[1], 6] += cue_scale * input_scale

    # baseline
    targets[:, :n_on[0]] = 0.0

    half    = int(round(0.5 / timing.dt))    # 0.5 s in steps
    quarter = int(round(0.25 / timing.dt))   # 0.25 s (shortened gng-hold / pairing windows)
    cu   = int(n_on[1])                   # cue ONSET
    co   = int(n_off[1])                  # cue OFFset

    if windowed_targets:
        # Transient decision. HOLD the go/nogo identity (go=+1 / nogo=−1) for 0.5 s ENDING at the cue
        # onset, so nogo learns to sit at −1 BEFORE the go-push cue; then a 0.5 s response 0.5 s after
        # cue-off (go→go_target; nogo NOT reset on cue — held free); then optionally decay to 0.
        targets[idx_go,   cu - quarter:cu, 1] =  1.0             # pre-cue hold: 0.25 s before the cue
        targets[idx_nogo, cu - quarter:cu, 1] = -1.0
        # response_in_cue: score in the last 0.5 s of the response cue (cue ON, r0:r1 = co-half:co) so the
        # lick is cue-DRIVEN; else the legacy 0.5 s window starting at cue-off. Decay follows at r1.
        r0, r1 = (co - half, co) if response_in_cue else (co, co + half)
        if gng_response:
            # go→go_target, nogo→nogo_target(=0). Scored by the UnifiedLoss rwd group (separately weighted).
            # RESPONSE = readout/lick → dim [-1] (κ₁ in rank-2, κ₂ in rank-3). The RULE stays held on [1].
            targets[idx_go,   r0:r1, -1] = go_target
            targets[idx_nogo, r0:r1, -1] = nogo_target
        if decay_to_zero:
            targets[:,        r1:r1 + 2*half, -1] = 0.0   # 'none' trials: pin the lick to 0
            if decay_onesided:  # go/nogo decay MARKERS on the LICK ([-1]) → loss scores one-sided at thresh 0
                targets[idx_go,   r1:r1 + 2*half, -1] = -0.5   # go: penalise lick>0 (relax the pulse DOWN)
                targets[idx_nogo, r1:r1 + 2*half, -1] = +0.5   # nogo: penalise lick<0 (relax the trace UP)
    else:
        dt = co + half
        if ramping_gng:
            targets[idx_go,   n_on[1], 1] = 1.0
            targets[idx_nogo, n_on[1], 1] = -1.0
        else:
            targets[idx_go,   n_off[0]:n_on[1], 1] = 1.0
            targets[idx_nogo, n_off[0]:n_on[1], 1] = -1.0
        targets[idx_nogo, n_off[1]:dt, 1] = nogo_target
        if target_rank >= 3:
            targets[idx_go,   n_off[1]:dt, 2] = go_target
            targets[idx_nogo, n_off[1]:dt, 2] = nogo_target

    return inputs, targets


def generate_dual_trials(
    n_trials: int,
    timing: TaskTiming,
    input_size: int = 8,
    target_rank: int = 1,
    noise: float = 0.1,
    baseline_value: float = 0.5,
    cue_on_go_input: bool = False,
    cue_scale: float = 1.0,
    nogo_target: float = 0.0,
    go_target: float = 1.0,
    go_on_rwd_input: bool = False,
    input_scale: float = 1.0,
    attention_input: bool = False,
    attention_gated: bool = False,
    attention_scale: float = 1.0,
    paired_only: bool = False,
    ramping_gng: bool = False,
    windowed_targets: bool = False,
    decay_to_zero: bool = True,
    gng_response: bool = False,
    gng_memory: bool = True,
    decay_onesided: bool = False,
    response_in_cue: bool = False,
):
    n_steps = timing.n_steps
    n_on = timing.n_stim_on
    n_off = timing.n_stim_off

    specs = [
        (sample, gng, test)
        for sample in ["A", "B"]
        for gng in ["none", "go", "nogo"]
        for test in ["C", "D"]
    ]
    if paired_only:   # curriculum bridge: MATCH trials only (A→C, B→D) — DPA decision always +1
        specs = [(s, g, t) for (s, g, t) in specs if (s == "A" and t == "C") or (s == "B" and t == "D")]

    n_types = len(specs)
    reps = int(np.ceil(n_trials / n_types))
    trial_type = torch.arange(n_types).repeat(reps)[:n_trials]
    trial_type = trial_type[torch.randperm(n_trials)]

    sample_code = torch.tensor([0 if s == "A" else 1 for s, _, _ in specs], dtype=torch.long)
    gng_code    = torch.tensor([0 if g == "none" else 1 if g == "go" else 2 for _, g, _ in specs], dtype=torch.long)
    test_code   = torch.tensor([0 if t == "C" else 1 for _, _, t in specs], dtype=torch.long)

    sample = sample_code[trial_type]
    gng    = gng_code[trial_type]
    test   = test_code[trial_type]

    idx_A    = sample == 0
    idx_B    = sample == 1
    idx_C    = test == 0
    idx_D    = test == 1
    idx_go   = gng == 1
    idx_nogo = gng == 2
    idx_gng  = idx_go | idx_nogo

    inputs  = noise * torch.randn(n_trials, n_steps, input_size)
    targets = torch.zeros(n_trials, n_steps, target_rank) * torch.nan

    if attention_input:   # tonic attention on the LAST channel
        _attn_window(inputs, timing, input_scale, attention_gated, attention_scale=attention_scale)

    inputs[idx_A,    n_on[0]:n_off[0], 0] += input_scale
    inputs[idx_B,    n_on[0]:n_off[0], 1] += input_scale

    if go_on_rwd_input:
        go_ch, ngo_ch = input_size - 1, 4
        inputs[idx_go,   n_on[1]:n_off[1], go_ch]  += input_scale
        inputs[idx_nogo, n_on[1]:n_off[1], ngo_ch] += input_scale
        inputs[idx_gng,  n_on[2]:n_off[2], go_ch]  += cue_scale * input_scale
    else:
        inputs[idx_go,   n_on[1]:n_off[1], 4] += input_scale
        inputs[idx_nogo, n_on[1]:n_off[1], 5] += input_scale

        if cue_on_go_input:
            inputs[idx_gng, n_on[2]:n_off[2], 4] += cue_scale * input_scale
        else:
            inputs[idx_gng, n_on[2]:n_off[2], 6] += cue_scale * input_scale

    inputs[idx_C,    n_on[3]:n_off[3], 2] += input_scale
    inputs[idx_D,    n_on[3]:n_off[3], 3] += input_scale

    # baseline
    targets[:, :n_on[0]] = 0.0

    half    = int(round(0.5 / timing.dt))    # 0.5 s in steps
    quarter = int(round(0.25 / timing.dt))   # 0.25 s (shortened gng-hold / pairing windows)
    cu   = int(n_on[2])                   # cue ONSET
    co   = int(n_off[2])                  # cue OFFset
    to   = int(n_off[3])                  # test offset
    idx_pair = (idx_A & idx_C) | (idx_B & idx_D)

    if windowed_targets:
        # Transient decisions (express then optionally decay).
        # go/nogo: HOLD the identity (go=+1 / nogo=−1) for 0.5 s ENDING at the cue onset so nogo sits
        # at −1 BEFORE the go-push cue (avoids the false lick); a 0.5 s response 0.5 s after cue-off;
        # then optionally decay to 0.
        if gng_memory:
            # the go/nogo working memory (pre-cue hold). Optional in Dual: with it off, the go/nogo
            # identity is NOT re-supervised — it must survive on the GNG-learned (frozen) structure.
            targets[idx_go,   cu - quarter:cu, 1] =  1.0         # pre-cue hold: 0.25 s before the cue
            targets[idx_nogo, cu - quarter:cu, 1] = -1.0

        # response_in_cue: gng response in the last 0.5 s of the response cue (cue ON, rg0:rg1 = co-half:co)
        # → lick is cue-DRIVEN; else legacy 0.5 s after cue-off. Decay follows at rg1.
        rg0, rg1 = (co - half, co) if response_in_cue else (co, co + half)
        if gng_response:
            # go→go_target, nogo→nogo_target(=0). Scored by the UnifiedLoss rwd group (separately weighted).
            # RESPONSE = readout/lick → dim [-1] (κ₁ in rank-2, κ₂ in rank-3). The RULE stays held on [1].
            targets[idx_go,   rg0:rg1, -1] = go_target
            targets[idx_nogo, rg0:rg1, -1] = nogo_target

        if decay_to_zero:
            targets[:,        rg1:rg1 + 2*half, -1] = 0.0   # 'none' trials: pin the lick to 0
            if decay_onesided:  # go/nogo decay MARKERS on the LICK ([-1]) → loss scores one-sided at thresh 0
                targets[idx_go,   rg1:rg1 + 2*half, -1] = -0.5   # go: penalise lick>0 (relax the pulse DOWN)
                targets[idx_nogo, rg1:rg1 + 2*half, -1] = +0.5   # nogo: penalise lick<0 (relax the trace UP)

        # pairing: response_in_cue → last 0.5 s of the TEST (test ON, rp0:rp1 = to-half:to) so the match
        # decision is test-DRIVEN; else legacy 0.25 s starting at test-off. Decay follows at rp1.
        rp0, rp1 = (to - half, to) if response_in_cue else (to, to + quarter)
        targets[idx_pair,  rp0:rp1, -1] =  1.0
        targets[~idx_pair, rp0:rp1, -1] = -1.0

        if decay_to_zero:
            targets[:,        rp1:, -1] = 0.0   # default: pin the lick to 0
            if decay_onesided:  # pairing decay MARKERS on the LICK ([-1]) → loss scores one-sided at thresh 0
                targets[idx_pair,  rp1:, -1] = -0.5   # match: penalise lick>0 (relax the pulse DOWN)
                targets[~idx_pair, rp1:, -1] = +0.5   # nonmatch: penalise lick<0 (relax the trace UP)
    else:
        dt = co + half
        if ramping_gng:
            targets[idx_go,   n_on[2], 1] = 1.0
            targets[idx_nogo, n_on[2], 1] = -1.0
        targets[idx_nogo, n_off[2]:dt, 1] = nogo_target
        if target_rank >= 3:
            targets[idx_go,   n_off[2]:dt, -1] = go_target
            targets[idx_nogo, n_off[2]:dt, -1] = nogo_target
        targets[idx_pair,  n_off[3], -1] = 1.0
        targets[~idx_pair, n_off[3], -1] = -1.0

    condition_names = np.array([
        f"{s}_{g}_{t}" if g != "none" else f"{s}_{t}"
        for s, g, t in specs
    ])[trial_type.numpy()]

    return inputs, targets, trial_type, condition_names
