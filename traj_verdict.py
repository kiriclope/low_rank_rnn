#!/usr/bin/env python
"""Trajectory VERDICT — does a run's κ(t) do the RIGHT THING on the task, stage by stage?

`flow_verdict.py` scores the autonomous LANDSCAPE (where the wells sit). This scores the
BEHAVIOUR ON THE TASK: the actual trajectory the network runs, checked against the expected
κ(t) (Leon's spec, 2026-09-02, encoded once in EXPECT below):

  DPA stage   — ckpt dpa_,    probe DPA task
      A/B memory on κ₀ maintained or decaying, sample-off → test; choice on κ₁ at test offset.
  GNG stage   — ckpt naive_,  probes GNG task AND DPA task
      GNG: go/nogo held on κ₁ at ±θ, maintained or transient; the cue pushes BOTH classes in the
           +κ₁ direction; a response is expressed (sometimes the wrong one).
      DPA: A/B memory maintained or slightly disrupted; choice still on κ₁, sometimes wrong.
  Dual stage  — ckpt expert_, probe Dual task
      A/B memory maintained or decaying; go/nogo as in GNG; choice on κ₁, mostly right.

★ The expectations FOLLOW THE RUN'S OWN TARGETS AND REGULARISERS (`_variant`/`_adapt`) — the ±1
in that spec is really ±θ, and θ is whatever this arm's loss demands. An amplitude band copied
from a th=1 arm would fail every sign-based arm for behaving exactly as designed. Adapted:
  • hinge thresholds — θ⁺ = go_hinge_thresh (else 1), θ⁻ = −nogo_hinge_thresh, θ_pair =
    dpa_hinge_thresh (else 1). ±1 targets are ONE-SIDED hinges, so a level test is a FLOOR
    (beyond θ is free), never a band; θ=0 (sign-based) ⇒ the floor is the noise scale σ_eff.
  • free targets — nogo_target=None (nogo response free, nolick covers it), rwd_nogo_onesided
    without rwd_keep_go_hinge (go response free), gng_response=False (no response window at all)
    drop the corresponding side rather than scoring an unsupervised quantity.
  • extra supervision — nolick_late_delay + nolick_weight>0 adds the late-delay κ₁≤0 check on the
    trials it actually constrains (nogo + 'none'); dpa_prelick_free removes the DPA delay pin check.
  • regularisers — decay_to_zero / gng_decay_to_zero / kappa1_reg_weight>0 all demand a TRANSIENT
    response, so `relax` becomes a scored check instead of a reported one.
  • softplus hinge_shape keeps rewarding overshoot (§25e), so the runaway ceiling is raised.
Every adaptation is printed as a `variants:` line per run — the table always says what it assumed.

Other conventions (identical to flow_verdict.py so the two tables compose):
  • κ = rates·n/N via kappa_from_rates — the model's own coordinate, the one the traj figures plot;
  • trials come from the run's OWN config, so trained windows (response_in_cue, gng_rwd_after_cue,
    decay_to_end, …) are respected — the checker never invents a window;
  • every sign test is at the FIXED boundary 0, never an arm's own lenient midpoint;
  • class labels come from the INPUT tensor (dual: the generator's condition names), never from the
    targets — a free/NaN target must not silently relabel a trial.

Usage:
  LD_PRELOAD=/home/leon/mambaforge/lib/libstdc++.so.6 python traj_verdict.py \
      --sweep_dir results/dual/sweep_r2renldw [--run_ids s0_renldw5] [--stages dpa gng dual]
      [--all_probes] [--verbose] [--json out.json]
"""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bifurcation_probe import load_run, discover_run_ids, run_dt_alpha
from src.flow_field import kappa_from_rates
from src.tasks import make_timings, generate_dpa_trials, generate_gng_trials, generate_dual_trials


# ── What a correct solution looks like (Leon, 2026-09-02) ──────────────────────────────────
# The ONLY place expectations live. `_adapt` then specialises these to the run's own config.
#   mem_cats  — acceptable memory categories (HELD / DECAY / FADE / LOST / FLIP / GROW, see _memory)
#   sep_min   — min fraction of trials whose κ₀ sign still matches the encoded sample at delay end
#   choice_min— min match/nonmatch accuracy from sign(κ₁) in the trained choice window (boundary 0)
#   rule_min  — min go/nogo accuracy from sign(κ₁) in the pre-cue hold window (boundary 0)
#   resp_min  — min go/nogo response accuracy in the trained response window (boundary 0)
#   cue_push  — the cue must move BOTH go and nogo in the +κ₁ direction
EXPECT = {
    ("dpa",  "dpa"):  dict(mem_cats={"HELD", "DECAY"}, sep_min=0.90, choice_min=0.90,
                           leak_max=None),                       # baseline: no GNG training yet
    ("gng",  "gng"):  dict(rule_min=0.90, cue_push=True, resp_min=0.85),
    ("gng",  "dpa"):  dict(mem_cats={"HELD", "DECAY"}, sep_min=0.75, choice_min=0.60,
                           leak_max=0.50),                       # ★ predicts the memory inversion
    ("dual", "dual"): dict(mem_cats={"HELD", "DECAY"}, sep_min=0.80, choice_min=0.80,
                           rule_min=0.75, cue_push=True, resp_min=0.70),
}
STAGE_CKPT = {"dpa": "dpa", "gng": "naive", "dual": "expert"}   # stage → checkpoint prefix
PROBES     = {"dpa": ["dpa"], "gng": ["gng", "dpa"], "dual": ["dual"]}
EXTRA      = {"dual": ["dpa", "gng"]}          # --all_probes: standalone retention after Dual

# memory categories from hold = amp(delay end)/amp(delay start)
HOLD_HELD, HOLD_DECAY, HOLD_GROW = 0.80, 0.30, 1.25
SEP_FLIP    = 0.45      # below this the sample bit is systematically INVERTED, not merely lost
LEVEL_FLOOR = 0.60      # a hinge at θ is satisfied from 0.6·θ up (one-sided: beyond θ is free)
LEVEL_CEIL  = 3.0       # ×max(θ,1): above this the decision has PARKED, not expressed
RELAX_TRANS = 0.50      # |κ₁(after response)| / |κ₁(response)| below this = transient


# ── variant handling: what does THIS run's loss actually demand? ───────────────────────────

def _variant(cfg):
    """Translate the run's config into the amplitudes/terms its loss actually imposes.
    Mirrors sweep.py's UnifiedLoss construction (_uth / _uth_neg / _uth_pair, rwd + nolick)."""
    _, alpha, _ = run_dt_alpha(cfg)
    th_go = cfg.get("go_hinge_thresh")
    v = dict(
        sigma=float(cfg.get("noise", 1.0) * np.sqrt(1 - np.exp(-2 * alpha))),
        th_go=float(th_go if th_go is not None else 1.0),           # gng +1 hinge
        th_nogo=float(-cfg.get("nogo_hinge_thresh", -1.0)),         # gng −1 hinge magnitude
        th_pair=float(cfg.get("dpa_hinge_thresh") or 1.0),          # pair + memory hold
        softplus=cfg.get("hinge_shape", "relu2") == "softplus",
        nogo_free=cfg.get("nogo_target", 0.0) is None,
        go_free=bool(cfg.get("rwd_nogo_onesided", False)) and not bool(cfg.get("rwd_keep_go_hinge", False)),
        no_resp=not bool(cfg.get("gng_response", False)),
        nolick=bool(cfg.get("nolick_late_delay", False)) and float(cfg.get("nolick_weight", 0.0)) > 0,
        nolick_full=bool(cfg.get("nolick_full_delay", False)) and float(cfg.get("nolick_weight", 0.0)) > 0,
        nolick_eps=float(cfg.get("nolick_thresh", 0.0) or 0.0),
        dpa_nolick=float(cfg.get("dpa_nolick_weight", 0.0) or 0.0) > 0,
        decoupled=bool(cfg.get("gng_decouple_decision", False)),
        no_cue=float(cfg.get("cue_scale", 2.0)) == 0.0,   # cue_scale=0: no push ever arrives
        prelick_pinned=not bool(cfg.get("dpa_prelick_free", False)),
        # a transient decision is DEMANDED by the decay pins / the subcriticality reg — and
        # gng_decay_to_zero is GNG-STAGE-ONLY (sweep.py passes decay_to_end there alone), so it
        # must not silently become a Dual-stage expectation.
        transient_gng=bool(cfg.get("decay_to_zero", True)) or bool(cfg.get("gng_decay_to_zero", False))
                      or float(cfg.get("kappa1_reg_weight", 0.0)) > 0,
        transient_dual=bool(cfg.get("decay_to_zero", True))
                       or float(cfg.get("kappa1_reg_weight", 0.0)) > 0,
        rule_supervised_dual=bool(cfg.get("dual_gng_memory", True)),
    )
    tags = [f"θ=({v['th_go']:g},−{v['th_nogo']:g},pair {v['th_pair']:g})",
            cfg.get("hinge_shape", "relu2")]
    if cfg.get("decay_to_zero", True):        tags.append("decay-pin")
    if cfg.get("gng_decay_to_zero", False):   tags.append("gng-decay-to-end")
    if float(cfg.get("kappa1_reg_weight", 0.0)) > 0: tags.append(f"κ₁reg={cfg['kappa1_reg_weight']}")
    if v["th_go"] == 0 or v["th_nogo"] == 0: tags.append(f"sign-based(σ={v['sigma']:.2f})")
    if v["nogo_free"]:
        # nogo_target=None hands don't-lick to the nolick term; with that off too, NOTHING imposes
        # it — the nogo response is then unscoreable by construction, not merely unmet.
        tags.append("nogo-resp-free" if v["nolick"] else "★nogo-UNCONSTRAINED (no target, no nolick)")
    if v["go_free"]:     tags.append("go-resp-free")
    if v["no_resp"]:     tags.append("no-response-window")
    if v["nolick"]:      tags.append(f"nolick-late(w={cfg.get('nolick_weight')})")
    if v["nolick_full"]: tags.append("nolick-FULL-delay(dpa-trials)")
    if v["nolick_eps"]:  tags.append(f"nolick-ε={v['nolick_eps']:g}")
    if v["dpa_nolick"]:  tags.append(f"dpa-stage-nolick(w={cfg.get('dpa_nolick_weight')})")
    if v["decoupled"]:   tags.append("gng-decoupled(n_dec⟂m₀)")
    if v["no_cue"]:      tags.append("NO-CUE(scale 0)")
    if cfg.get("gng_hold_full_delay", False): tags.append("full-delay-hold")
    if not v["prelick_pinned"]: tags.append("dpa-delay-free")
    if not v["rule_supervised_dual"]: tags.append("dual-rule-unsupervised")
    v["tags"] = tags
    return v


def _adapt(stage, task, v):
    """Specialise the EXPECT entry to this run: drop checks whose target is free, add the ones its
    extra supervision creates, and convert the ±1 level test into ±θ floors.
    A key set to None means REPORT BUT DO NOT SCORE — the quantity is worth seeing, but nothing in
    this arm's loss demands it, so it must not decide a verdict."""
    exp = dict(EXPECT[(stage, task)])
    exp.update(sigma=v["sigma"], ceil_mult=(2 * LEVEL_CEIL if v["softplus"] else LEVEL_CEIL))
    if task in ("gng", "dual"):
        exp.update(th_go=v["th_go"], th_nogo=v["th_nogo"],
                   score_go=not v["go_free"], score_nogo=not v["nogo_free"],
                   score_level=(task == "gng" or v["rule_supervised_dual"]))
        if v["no_resp"]:
            exp.pop("resp_min", None)                    # no response window exists to score
        if v["no_cue"]:
            exp.pop("cue_push", None)                    # cue_scale=0: there is no push to check
        exp["relax_max"] = RELAX_TRANS if v[f"transient_{'gng' if task == 'gng' else 'dual'}"] else None
        if task == "dual":
            # scored if EITHER don't-lick term is active (late window on nogo, full delay on the
            # DPA trials, or both); _nolick reads nolick_full/nolick_eps to build the right mask
            exp["nolick"] = True if (v["nolick"] or v["nolick_full"]) else None
            exp["nolick_full"] = v["nolick_full"]
            exp["nolick_late"] = v["nolick"]
            exp["nolick_eps"] = v["nolick_eps"]
    if task == "dpa":
        exp["th_pair"] = v["th_pair"]
        if v["prelick_pinned"]:                    # legacy two-sided 0-pin over the delay
            exp["prelick_max"], exp["prelick_onesided"] = 0.3 * max(v["th_pair"], 1.0), False
        elif v["dpa_nolick"] and stage == "dpa":   # one-sided hinge imposed AT the DPA stage:
            #                                        κ₁<0 free by design, only upward drift violates
            exp["prelick_max"], exp["prelick_onesided"] = 0.3 * max(v["th_pair"], 1.0), True
        else:
            exp["prelick_max"], exp["prelick_onesided"] = None, False
    return exp


# ── trials ────────────────────────────────────────────────────────────────────────────────

def _gen(task, cfg, T, n, seed, noise):
    """The run's own trials for `task`. Kwargs mirror sweep.run_single exactly — if a generator
    flag is added there it must be added here or the probe drifts from what was trained."""
    torch.manual_seed(seed)
    common = dict(input_size=cfg["input_size"], target_rank=cfg["target_rank"], noise=noise,
                  input_scale=cfg.get("input_scale", 1.0),
                  attention_input=cfg.get("attention_input", False),
                  attention_gated=cfg.get("attention_gated", True),
                  attention_scale=cfg.get("attention_scale", 1.0),
                  windowed_targets=cfg.get("windowed_targets", True),
                  decay_onesided=cfg.get("decay_onesided", False),
                  response_in_cue=cfg.get("response_in_cue", False))
    gng_common = dict(cue_on_go_input=cfg.get("cue_on_go_input", True),
                      cue_scale=cfg.get("cue_scale", 2.0),
                      nogo_target=cfg.get("nogo_target", 0.0),
                      go_target=cfg.get("go_target", 1.0),
                      go_on_rwd_input=cfg.get("go_on_rwd_input", False),
                      ramping_gng=cfg.get("ramping_gng", False),
                      gng_response=cfg.get("gng_response", True),
                      gng_rwd_after_cue=cfg.get("gng_rwd_after_cue", False),
                      hold_full_delay=cfg.get("gng_hold_full_delay", False))
    if task == "dpa":
        X, y = generate_dpa_trials(n, T, decay_to_zero=cfg.get("decay_to_zero", False),
                                   prelick_free=cfg.get("dpa_prelick_free", False), **common)
        return X, y, None
    if task == "gng":
        _d2e = cfg.get("gng_decay_to_zero", False)
        X, y = generate_gng_trials(n, T, decay_to_zero=cfg.get("decay_to_zero", False) or _d2e,
                                   decay_to_end=_d2e, **common, **gng_common)
        return X, y, None
    X, y, _, names = generate_dual_trials(n, T, decay_to_zero=cfg.get("decay_to_zero", False),
                                          gng_memory=cfg.get("dual_gng_memory", False),
                                          **common, **gng_common)
    return X, y, np.asarray(names).astype(str)


def _labels(task, cfg, T, X, names):
    """Trial classes from the INPUT channels (dual: from the generator's condition names)."""
    on, off = T.n_stim_on, T.n_stim_off
    go_ch, ngo_ch = (cfg["input_size"] - 1, 4) if cfg.get("go_on_rwd_input", False) else (4, 5)
    if task == "dual":
        smp = np.array([n[0] for n in names])
        tst = np.array([n[-1] for n in names])
        return dict(A=smp == "A", B=smp == "B",
                    go=np.array(["_go_" in n for n in names]),
                    nogo=np.array(["_nogo_" in n for n in names]),
                    none=np.array([("_go_" not in n) and ("_nogo_" not in n) for n in names]),
                    pair=((smp == "A") & (tst == "C")) | ((smp == "B") & (tst == "D")))
    if task == "dpa":
        s = slice(int(on[0]), int(off[0])); t = slice(int(on[1]), int(off[1]))
        A = (X[:, s, 0].mean(1) > X[:, s, 1].mean(1)).numpy()
        C = (X[:, t, 2].mean(1) > X[:, t, 3].mean(1)).numpy()
        return dict(A=A, B=~A, pair=(A & C) | (~A & ~C))
    s = slice(int(on[0]), int(off[0]))
    go = (X[:, s, go_ch].mean(1) > X[:, s, ngo_ch].mean(1)).numpy()
    return dict(go=go, nogo=~go)


def _windows(task, cfg, T):
    """Epoch windows in STEPS, following the run's trained target windows (src/tasks.py) and the
    nolick window sweep.py builds (_nlw_d / _nlw_g)."""
    half, quarter = int(round(0.5 / T.dt)), int(round(0.25 / T.dt))
    on, off = T.n_stim_on, T.n_stim_off
    ric = bool(cfg.get("response_in_cue", False))
    w: dict[str, object] = dict(half=half, quarter=quarter)
    if task == "dpa":
        to = int(off[1])
        w["mem"] = (int(off[0]), int(on[1]))                        # sample-off → test-on
        w["delay"] = (int(off[0]), int(on[1]))                      # the pre-test no-lick span
        w["choice"] = (to - half, to) if ric else (to, to + quarter)
        return w
    cu, co = (int(on[1]), int(off[1])) if task == "gng" else (int(on[2]), int(off[2]))
    # the pre-cue go/nogo hold: 0.25 s before the cue (windowed_targets) or the whole hold span
    w["rule"] = (cu - quarter, cu) if cfg.get("windowed_targets", True) else (int(off[0]), cu)
    w["cue"] = (cu, co)
    in_cue = ric and not bool(cfg.get("gng_rwd_after_cue", False))
    w["resp"] = (co - half, co) if in_cue else (co, co + half)
    end = int(on[3]) if task == "dual" else T.n_steps                # dual: stop before the test
    w["post"] = (max(int(w["resp"][1]), end - quarter), end)         # where a transient has relaxed
    w["nolick"] = (co, end)                                          # sweep.py's _nlw_d / _nlw_g
    if task == "dual":
        to = int(off[3])
        w["mem"] = (int(off[0]), int(on[3]))                         # sample-off → test-on
        w["nolick_full"] = (int(off[0]), end)                        # sweep.py's _nlf_d (DPA trials)
        w["choice"] = (to - half, to) if ric else (to, to + quarter)
    return w


def _mean(k, win):
    a, b = int(win[0]), int(win[1])
    return k[:, max(0, a):max(a + 1, b)].mean(1)


def _level(v, th, exp):
    """Is |v| consistent with a ONE-SIDED hinge at θ? Beyond θ is free, so the test is a floor
    (θ>0) or a noise-scale floor (θ=0, sign-based margins emerge at σ_eff), plus a parking ceiling."""
    floor = LEVEL_FLOOR * th if th > 0 else 0.5 * exp["sigma"]
    return floor <= abs(v) <= exp["ceil_mult"] * max(th, 1.0), floor


# ── the checks ────────────────────────────────────────────────────────────────────────────

def _memory(k0, k1, lab, w, exp, free):
    """A/B memory on κ₀ across the delay. amp = half the class-mean separation; the sign mapping is
    fixed at the START (encoding), so a mid-delay flip shows up as sep falling below 0.5 rather than
    being hidden by an argmax over label assignments."""
    a, b = w["mem"]; q = w["quarter"]
    out, mm = {}, {}
    for tag, win in (("0", (a, a + q)), ("1", (max(a + q, b - q), b))):
        m = _mean(k0, win)
        out["amp" + tag], mm[tag] = float(m[lab["A"]].mean() - m[lab["B"]].mean()) / 2.0, m
    s = np.sign(out["amp0"]) or 1.0
    for tag in ("0", "1"):
        out["sep" + tag] = float((((mm[tag] * s) > 0).numpy() == lab["A"]).mean())
    if lab.get("none") is not None:
        # ★ cue-perturbation cost on the memory (Leon 2026-09-02): the cue lifts κ₁ by ~1 on
        # cued trials mid-delay; κ₁ excursions destabilise κ₀ (the flip mechanism), so the A/B
        # bit at delay END should be degraded on cued (go/nogo) trials relative to uncued.
        ok1 = ((mm["1"] * s) > 0).numpy() == lab["A"]
        cued = lab["go"] | lab["nogo"]
        for tag, msk in (("none", lab["none"]), ("cued", cued)):
            m1 = mm["1"].numpy()[msk]
            out["sep1_" + tag] = float(ok1[msk].mean())
            out["amp1_" + tag] = float(m1[lab["A"][msk]].mean() - m1[lab["B"][msk]].mean()) / 2.0
    h = out["amp1"] / out["amp0"] if abs(out["amp0"]) > 1e-9 else float("nan")
    sep1 = out["sep1"]
    out["hold"] = h
    out["cat"] = ("FLIP" if sep1 < SEP_FLIP else "LOST" if sep1 < exp["sep_min"] else
                  "GROW" if h > HOLD_GROW else "HELD" if h >= HOLD_HELD else
                  "DECAY" if h >= HOLD_DECAY else "FADE")
    out["ok"] = bool(out["cat"] in exp["mem_cats"] and sep1 >= exp["sep_min"])
    out["txt"] = (f"mem {out['cat']:5s} {out['amp0']:+.2f}→{out['amp1']:+.2f} "
                  f"(hold {h:.2f}) sep {out['sep0']:.2f}→{sep1:.2f}")
    if "amp1_cued" in out:
        out["txt"] += (f" [end: none κ₀±{out['amp1_none']:.2f}/sep {out['sep1_none']:.2f}"
                       f" vs cued ±{out['amp1_cued']:.2f}/{out['sep1_cued']:.2f}]")
    return out


def _choice(k0, k1, lab, w, exp, free):
    """Match/nonmatch from sign(κ₁) in the trained choice window, at the fixed boundary 0."""
    m = _mean(k1, w["choice"]).numpy()
    corr = (m > 0) == lab["pair"]
    out = dict(acc=float(corr.mean()), pair=float(corr[lab["pair"]].mean()),
               unpair=float(corr[~lab["pair"]].mean()))
    out["ok"] = out["acc"] >= exp["choice_min"]
    out["txt"] = f"choice@0 {out['acc']:.2f} (P {out['pair']:.2f} U {out['unpair']:.2f})"
    if lab.get("none") is not None:
        # ★ pairing split by gng condition (Leon 2026-09-02): if the cue's up-push damages the
        # held A/B memory, cued trials pair WORSE than uncued. Reported, not scored.
        for tag in ("none", "go", "nogo"):
            msk = lab[tag]
            out["acc_" + tag] = float(corr[msk].mean()) if msk.any() else float("nan")
        out["txt"] += (f" [by-gng: none {out['acc_none']:.2f} go {out['acc_go']:.2f}"
                       f" nogo {out['acc_nogo']:.2f}]")
    return out


def _rule(k0, k1, lab, w, exp, free):
    """go/nogo identity held on κ₁ before the cue: sign accuracy at 0 AND the ±θ level this arm's
    hinges demand (a sign-based arm is expected at the noise scale, not at ±1). The level only
    COUNTS where the hold is actually supervised — with dual_gng_memory=False the Dual stage never
    re-imposes it, so there it is reported and the sign is what's scored."""
    m = _mean(k1, w["rule"]).numpy()
    go, ng = m[lab["go"]], m[lab["nogo"]]
    out = dict(acc=float(0.5 * ((go > 0).mean() + (ng <= 0).mean())),
               go=float(go.mean()), nogo=float(ng.mean()))
    ok_g, fl_g = _level(out["go"], exp["th_go"], exp)
    ok_n, fl_n = _level(out["nogo"], exp["th_nogo"], exp)
    out["level_ok"] = bool(ok_g and ok_n)
    out["ok"] = bool(out["acc"] >= exp["rule_min"] and (out["level_ok"] or not exp["score_level"]))
    lvl = "" if out["level_ok"] else (f"  [level≠±θ, floors {fl_g:.2f}/{fl_n:.2f}"
                                      f"{'' if exp['score_level'] else ', unsupervised'}]")
    out["txt"] = f"rule@0 {out['acc']:.2f} (go {out['go']:+.2f} nogo {out['nogo']:+.2f}){lvl}"
    return out


def _cue(k0, k1, lab, w, exp, free):
    """The cue must push BOTH classes toward +κ₁ (Δκ₁, cue onset → cue offset).
    nogo carries the test: it must be driven UP out of its −θ hold. go is often already at its
    hinge ceiling before the cue, where the net Δ is ≈0 — so go only has to not be pushed DOWN
    (tolerance = ¼σ_eff). 'none' trials (Dual only) get no cue and are the control: they shouldn't move."""
    q = w["quarter"]; cu, co = w["cue"]
    d = (_mean(k1, (max(co - q, cu), co)) - _mean(k1, (cu - q, cu))).numpy()
    tol = 0.25 * exp["sigma"]
    out = dict(go=float(d[lab["go"]].mean()), nogo=float(d[lab["nogo"]].mean()))
    out["ok"] = bool(out["nogo"] > 0 and out["go"] > -tol)
    ctrl = f" none {float(d[lab['none']].mean()):+.2f}" if lab.get("none") is not None else ""
    out["txt"] = f"cue Δκ₁ go {out['go']:+.2f} nogo {out['nogo']:+.2f}{ctrl}"
    return out


def _resp(k0, k1, lab, w, exp, free):
    """go/nogo response in the trained response window at the fixed boundary 0, plus its level.
    Sides whose target this arm left FREE (nogo_target=None, rwd one-sided) are reported, not scored
    — an unsupervised quantity must not decide a verdict."""
    m = _mean(k1, w["resp"]).numpy()
    go, ng = float(m[lab["go"]].mean()), float(m[lab["nogo"]].mean())
    accs, out = [], dict(go_acc=float((m[lab["go"]] > 0).mean()),
                         nogo_acc=float((m[lab["nogo"]] <= 0).mean()), go=go, nogo=ng)
    if exp["score_go"]:
        accs.append(out["go_acc"])
    if exp["score_nogo"]:
        accs.append(out["nogo_acc"])
    out["acc"] = float(np.mean(accs)) if accs else float("nan")
    out["ok"] = bool(accs) and out["acc"] >= exp["resp_min"]
    out["level_ok"] = _level(go, exp["th_go"], exp)[0] if exp["score_go"] else True
    free = ("" if exp["score_go"] else " go=free") + ("" if exp["score_nogo"] else " nogo=free")
    out["txt"] = (f"resp@0 go {out['go_acc']:.2f} nogo {out['nogo_acc']:.2f} "
                  f"(κ₁ {go:+.2f}/{ng:+.2f}){free}")
    return out


def _relax(k0, k1, lab, w, exp, free):
    """The decay/kappa1_reg arms demand a TRANSIENT decision: the go response must come back down
    before the test. ratio = |κ₁ after the response| / |κ₁ in the response window| (go trials)."""
    peak = abs(float(_mean(k1, w["resp"])[lab["go"]].mean()))
    tail = abs(float(_mean(k1, w["post"])[lab["go"]].mean()))
    r = tail / peak if peak > 1e-9 else float("nan")
    out = dict(ratio=r, mode="TRANS" if r < RELAX_TRANS else "MAINT" if r >= 0.7 else "PART")
    out["ok"] = bool(exp["relax_max"] is None or r < exp["relax_max"])
    out["txt"] = f"relax {out['mode']} ({r:.2f})"
    return out


def _nolick(k0, k1, lab, w, exp, free):
    """The don't-lick imposition, scored on exactly the (trial, step) pairs the loss constrains:
    the late delay (cue-off → test-on) of nogo + DPA trials under nolick_late_delay, and — in
    nolick_full_delay arms — the WHOLE delay (sample-off → test-on) of the DPA trials (the rows
    with no go/nogo stimulus and no cue). Only FREE (NaN-target) steps count: the go response and
    any decay pins inside a span are governed by their own terms.
    Reported as the mean AND the fraction of violating steps/trials, because the loss is a MEAN and
    a large violating minority hides behind a negative average (the trap Leon hit on 2026-08-13).
    ok stays at the fixed boundary 0 even when nolick_thresh ε>0 displaces the hinge — cross-arm
    comparability; ε buys DEPTH, which flow_verdict measures. ε is echoed in the txt."""
    rows_none = lab.get("none")
    sel = np.zeros(tuple(k1.shape), bool)
    a, b = int(w["nolick"][0]), int(w["nolick"][1])
    if exp.get("nolick_full") and rows_none is not None:
        f0, f1 = int(w["nolick_full"][0]), int(w["nolick_full"][1])
        sel[rows_none, f0:f1] = True                     # DPA trials: the whole delay
        if exp.get("nolick_late", True):
            sel[lab["nogo"], a:b] = True                 # go/nogo trials keep the late window
    else:
        rows = (lab["nogo"] | rows_none) if rows_none is not None else lab["nogo"]
        sel[rows, a:b] = True
    if free is not None:
        sel &= free.numpy()
    if not sel.any():
        return dict(mean=float("nan"), frac_up=float("nan"), trials_up=float("nan"), ok=False,
                    txt="nolick — NO free steps in the window (term inert)")
    k = k1.numpy()
    n_per = sel.sum(1)
    tmean = (k * sel).sum(1)[n_per > 0] / n_per[n_per > 0]
    out = dict(mean=float(k[sel].mean()), frac_up=float((k[sel] > 0).mean()),
               trials_up=float((tmean > 0).mean()), n_steps=int(sel.sum()))
    out["ok"] = bool(out["mean"] <= 0 and out["trials_up"] <= 0.1)
    eps = exp.get("nolick_eps", 0.0)
    tag = "[full] " if exp.get("nolick_full") else ""
    out["txt"] = (f"nolick {tag}κ₁ {out['mean']:+.2f} (steps>0 {out['frac_up']:.2f}, "
                  f"trials>0 {out['trials_up']:.2f}" + (f", ε=−{eps:g}" if eps else "") + ")")
    return out


def _leak(k0, k1, lab, w, exp, free):
    """★ Sample memory leaking onto the DECISION axis: |κ₁(A) − κ₁(B)| over the early-mid delay,
    BEFORE any crossing. This is the single best predictor of the GNG memory inversion (§26/§27):
    across 12 runs of three arms, leak ≥ 0.85 flipped the code in 6/6 and leak ≤ 0.27 kept it in
    5/5, with one 'lost' run at 0.53 in between — monotone, no overlap. The raw overlap g·n₁ᵀm₀
    does NOT predict it (it assumes φ′=1), so measure the realised leak, not the weights.
    Threshold calibrated on those 12 runs only — treat as a strong indicator, not a constant."""
    a, b = w["mem"]
    win = (a + w["quarter"], min(b, a + 8 * w["quarter"]))      # ~0.25–2 s into the delay
    m = _mean(k1, win).numpy()
    out = dict(A=float(m[lab["A"]].mean()), B=float(m[lab["B"]].mean()))
    out["leak"] = abs(out["A"] - out["B"])
    out["ok"] = bool(exp["leak_max"] is None or out["leak"] <= exp["leak_max"])
    out["txt"] = f"leak {out['leak']:.2f} (κ₁ A {out['A']:+.2f} B {out['B']:+.2f})"
    return out


def _prelick(k0, k1, lab, w, exp, free):
    """DPA-delay drift on κ₁. Two-sided when the legacy 0-pin supervises the delay
    (dpa_prelick_free=False); ONE-SIDED when the dpa_nolick hinge does — κ₁<0 is then free by
    design, so only upward drift violates; reported-only otherwise."""
    v = float(_mean(k1, w["delay"]).mean())
    lim = exp["prelick_max"]
    ok = True if lim is None else (v <= lim if exp.get("prelick_onesided") else abs(v) <= lim)
    return dict(k1=v, ok=bool(ok), txt=f"prelick {v:+.2f}")


# check → the EXPECT key that switches it on (absent key ⇒ the check does not apply)
CHECKS = {"mem": (_memory, "mem_cats"), "leak": (_leak, "leak_max"), "rule": (_rule, "rule_min"),
          "cue": (_cue, "cue_push"), "resp": (_resp, "resp_min"), "relax": (_relax, "relax_max"),
          "nolick": (_nolick, "nolick"), "choice": (_choice, "choice_min"),
          "prelick": (_prelick, "prelick_max")}
ORDER = ["mem", "leak", "rule", "cue", "resp", "relax", "nolick", "choice", "prelick"]


# ── driver ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def probe(model, cfg, task, exp, n_trials, seed, device, clean, model_noise):
    _, alpha, _ = run_dt_alpha(cfg)
    sig = float(np.sqrt(1 - np.exp(-2 * alpha)))
    T = make_timings(cfg.get("dt_base", 0.03) * cfg.get("tau_rec_frac", 0.75))[task]
    X, y, names = _gen(task, cfg, T, n_trials, seed, 0.0 if clean else cfg["noise"] * sig)
    lab = _labels(task, cfg, T, X, names)
    w = _windows(task, cfg, T)
    model.noise = (cfg.get("model_noise", 0.0) * sig) if model_noise else 0.0
    _, rates, _ = model(X.to(device), y.to(device), ret_rates=True)
    kap = kappa_from_rates(model, rates).cpu()
    model.noise = 0.0
    k0, k1 = kap[..., 0], kap[..., cfg.get("rank", 2) - 1]
    free = torch.isnan(y[..., -1])          # the steps the loss left FREE on the decision channel
    res = {}
    for name in ORDER:
        fn, key = CHECKS[name]
        if key not in exp:
            continue                        # this arm has no such term at all
        r = res[name] = fn(k0, k1, lab, w, exp, free)
        r["scored"] = exp[key] is not None  # None ⇒ reported only, never decides a verdict
        if not r["scored"]:
            r["ok"] = True
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--run_ids", nargs="*", default=None)
    ap.add_argument("--stages", nargs="*", default=["dpa", "gng", "dual"],
                    choices=["dpa", "gng", "dual"])
    ap.add_argument("--all_probes", action="store_true",
                    help="also probe the standalone DPA and GNG tasks after Dual (retention)")
    ap.add_argument("--n_trials", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clean", action="store_true", help="no input noise (default: the trained σ)")
    ap.add_argument("--model_noise", action="store_true", help="also inject the trained recurrent noise")
    ap.add_argument("--sep_min", type=float, default=None, help="override sep_min at every stage")
    ap.add_argument("--choice_min", type=float, default=None)
    ap.add_argument("--rule_min", type=float, default=None)
    ap.add_argument("--verbose", action="store_true", help="one line per check instead of per probe")
    ap.add_argument("--json", default=None, help="dump every measurement here")
    args = ap.parse_args()

    rids = args.run_ids or discover_run_ids(args.sweep_dir)
    tally, dump, n_ok, n_seen = {}, {}, 0, 0
    print(f"TRAJ VERDICT — {os.path.basename(os.path.normpath(args.sweep_dir))}  "
          f"n={args.n_trials} {'clean' if args.clean else 'trained-σ'} inputs  boundary=0")
    for rid in rids:
        bad_run, shown = [], False
        for stage in args.stages:
            try:
                model, cfg = load_run(args.sweep_dir, rid, stage=STAGE_CKPT[stage], device=args.device)
            except (FileNotFoundError, ValueError) as e:
                print(f"{rid:14s} {stage.upper():4s} — skipped ({type(e).__name__}: {e})")
                continue
            var = _variant(cfg)
            if not shown:
                print(f"{rid:14s} variants: " + " · ".join(var["tags"])); shown = True
            probes = PROBES[stage] + (EXTRA.get(stage, []) if args.all_probes else [])
            for task in probes:
                # a standalone probe after Dual is judged by that task's own stage expectations
                key = (stage, task) if (stage, task) in EXPECT else ("gng", task)
                exp = _adapt(*key, var) if key in EXPECT else None
                if exp is None:
                    continue
                for k in ("sep_min", "choice_min", "rule_min"):
                    if getattr(args, k) is not None and k in exp:
                        exp[k] = getattr(args, k)
                res = probe(model, cfg, task, exp, args.n_trials, args.seed,
                            args.device, args.clean, args.model_noise)
                dump[f"{rid}/{stage}/{task}"] = {k: {a: b for a, b in v.items() if a != "txt"}
                                                 for k, v in res.items()}
                bad = [k for k in ORDER if k in res and not res[k]["ok"]]
                bad_run += [f"{stage}/{task}:{k}" for k in bad]
                for k in res:
                    if res[k]["scored"]:
                        t = tally.setdefault((stage, task, k), [0, 0])
                        t[0] += res[k]["ok"]; t[1] += 1
                head = f"{rid:14s} {stage.upper():4s}/{task:4s}"
                parts = [f"{res[k]['txt']} " + ("✓" if res[k]["ok"] else "✗") * res[k]["scored"]
                         + "·" * (not res[k]["scored"]) for k in ORDER if k in res]
                verdict = "OK" if not bad else "ODD: " + ",".join(bad)
                if args.verbose:
                    print(f"{head}  {verdict}")
                    for p in parts:
                        print(f"{'':14s}      {p}")
                else:
                    print(f"{head}  " + " | ".join(parts) + f"   {verdict}")
        n_seen += shown
        n_ok += shown and not bad_run
    for st, tk in dict.fromkeys((s, t) for s, t, _ in tally):
        print(f"SUMMARY  {st.upper():4s}/{tk:4s}: " +
              "  ".join(f"{ck} {v[0]}/{v[1]}" for (s2, t2, ck), v in tally.items()
                        if (s2, t2) == (st, tk)))
    print(f"TRAJ OK: {n_ok}/{n_seen} runs pass every check")
    if args.json:
        json.dump(dump, open(args.json, "w"), indent=1, default=float)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
