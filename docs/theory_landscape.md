# Theory: why tanh gives symmetric wells, and how the landscape evolves across DPA → GNG → Dual

This note derives, from the rank‑2 low‑rank dynamics, **why a `tanh` nonlinearity forces the
memory attractors into a symmetric double‑well (ring) centred at the origin**, sets up the
**energy‑landscape** picture, and uses it to explain what each training stage does to the
landscape (well shape, attractor location, slow manifold). It is the analytic backbone for
the empirical thread in [`ring_lowerplane_log.md`](ring_lowerplane_log.md).

Notation: $N$ units, rank‑2 with loading/readout matrices $M=[\,m_0\;m_1\,]\in\mathbb R^{N\times2}$
(recurrent "output" vectors) and $\;n=[\,n_0\;n_1\,]\in\mathbb R^{N\times2}$ (readout vectors).
$g$ = gain, $\phi$ = nonlinearity, $x$ = input, $W_i$ = input weights.

---

## 1. The κ‑plane reduction

The network state is $(\,\text{rates}\;r,\ \text{recurrent input}\;h\,)$ with

$$
h_{t+1} = e^{-\alpha_{\rm rec}}h_t + (1-e^{-\alpha_{\rm rec}})\,W_{\rm rec} r_t,\qquad
W_{\rm rec}=\tfrac1N m\,n^{\!\top},
$$
$$
r_{t+1} = e^{-\alpha}r_t + (1-e^{-\alpha})\,\phi\!\big(g\,(A_iW_i x + h_t)\big).
$$

The **collective coordinates** are the overlaps $\kappa = \tfrac1N n^{\!\top} r \in\mathbb R^2$
($\kappa_0$ = memory, $\kappa_1$ = decision). Because $W_{\rm rec} r = \tfrac1N m\,(n^{\!\top}r)=M\kappa$,
the recurrent input relaxes onto the 2‑D plane $h \to M\kappa$. In the **adiabatic limit**
$h=M\kappa$ the dynamics closes on $\kappa$:

$$
\boxed{\;\kappa_{t+1}=\kappa_t+\beta\,F(\kappa_t;x),\qquad
F(\kappa;x)=\Psi(\kappa;x)-\kappa,\qquad
\Psi(\kappa;x)=\tfrac1N\,n^{\!\top}\phi\!\big(g\,(b(x)+M\kappa)\big)\;}
$$

with $\beta=1-e^{-\alpha}$ and per‑unit input drive $b(x)=A_iW_i x\in\mathbb R^N$.
Fixed points solve $F(\kappa^{*})=0$, i.e. the **self‑consistency** $\kappa^{*}=\Psi(\kappa^{*})$.
The $\beta$ (time constant) only scales $F$ — it sets *speed and stability*, never the
*location or number* of fixed points (steady state has no $\beta$). This is why $\tau$ is
irrelevant to the unimodal/bimodal question.

---

## 2. The energy landscape (when it exists)

$F$ is a gradient flow, $F=-\nabla_\kappa V$, **iff** the Jacobian $\partial F/\partial\kappa$ is
symmetric. With

$$
\frac{\partial F_a}{\partial\kappa_c}= g\,G_{ac}-\delta_{ac},\qquad
G_{ac}=\tfrac1N\sum_i n_a[i]\,\phi'_i\,m_c[i],\quad
\phi'_i=\phi'\!\big(g(b+M\kappa)_i\big),
$$

$G$ is symmetric exactly when $n$ and $m$ play interchangeable roles. The clean, exactly
solvable case is **symmetric connectivity** $n=M$ (readout = loading). Then $F=-\nabla V$ with

$$
\boxed{\,V(\kappa)=\tfrac12\|\kappa\|^2-\frac1{gN}\sum_{i=1}^{N}\Phi\!\big(g(b_i+(M\kappa)_i)\big),\qquad
\Phi'=\phi\,}
$$

(check: $\partial_{\kappa_a}V=\kappa_a-\tfrac1N\sum_i M[i,a]\,\phi(\cdot)=\kappa_a-\Psi_a=-F_a$).
For `tanh`, $\Phi(u)=\log\cosh u$. **The attractors are the minima of $V$; saddles its saddle
points.** When $n\neq M$, $G$ has an antisymmetric part $G_{\rm anti}$ that is *not* captured by any
potential — that is the **rotational** component (§5), but the well structure is still organised by
the symmetric part.

---

## 3. Why `tanh` ⇒ symmetric wells

Take the **autonomous** field ($x=0$, and assume the input bias $b$ has negligible projection on
the readout, $\tfrac1N n^{\!\top}\!\mathrm{diag}(\phi')\,b\approx0$ — verified empirically:
$\Psi(0)\approx0$ even though $\|b\|$ is large, because $b\perp n$). Then $b$ drops out:

$$
\Psi(\kappa)=\tfrac1N n^{\!\top}\phi(gM\kappa).
$$

`tanh` is **odd**, $\phi(-u)=-\phi(u)$, so $\Psi(-\kappa)=-\Psi(\kappa)$ and therefore

$$
\boxed{\,F(-\kappa)=-F(\kappa)\,}\qquad\text{(the field is odd).}
$$

Two consequences:

1. **Fixed points come in $\pm$ pairs.** If $\kappa^{*}$ is a fixed point so is $-\kappa^{*}$, with the
   same stability. The set of attractors is symmetric under $\kappa\mapsto-\kappa$.
2. **The potential is even.** With $n=M$, $V(-\kappa)=\tfrac12\|\kappa\|^2-\tfrac1{gN}\sum_i\Phi(-g(M\kappa)_i)=V(\kappa)$
   because $\Phi=\log\cosh$ is even. An **even potential has wells symmetric about the origin**.

So the memory landscape is a double (or multi‑) well *centred at* $\kappa=0$. The two memory
attractors are antipodal, $\kappa^{*}$ and $-\kappa^{*}$ — they **straddle** $\kappa_1=0$ (one
"upper", one "lower"). This is the structural reason the autonomous attractors cannot *all* sit in
the lower plane while $\phi$ is odd: it is forbidden by the $\kappa\mapsto-\kappa$ symmetry.

> **Correction to a tempting but wrong intuition.** The symmetry pins the *centre* of the ring at the
> origin, **not** the attractors at $\kappa_1\!=\!0$. The wells sit out on the ring at $\kappa_0\!\approx\!\pm1,\ \kappa_1\!\approx\!\pm0.5$.

---

## 4. The bistability (ring) condition

Restrict to the memory axis $\kappa=(\kappa_0,0)$ with $b=0$ and $n=M$. The reduced potential is

$$
V(\kappa_0)=\tfrac12\kappa_0^2-\frac1{gN}\sum_i\log\cosh\!\big(g\,m_0[i]\,\kappa_0\big).
$$

Expand $\log\cosh u=\tfrac12u^2-\tfrac1{12}u^4+\dots$:

$$
V(\kappa_0)=\tfrac12\big(1-g\lambda_0\big)\,\kappa_0^2+\frac{g^3}{12N}\!\sum_i m_0[i]^4\,\kappa_0^4+\dots,
\qquad \lambda_0=\tfrac1N\sum_i m_0[i]^2=\tfrac1N\|m_0\|^2 .
$$

(For general $n\neq m$, replace $\lambda_0$ by the effective coupling $\tfrac1N n_0^{\!\top}m_0$.)
The quartic coefficient is positive, so the shape is set by the quadratic:

$$
\boxed{\;g\lambda_0<1:\ \text{single well at }0\;\;\Longrightarrow\;\;
g\lambda_0>1:\ \text{symmetric double well at }\pm\kappa_0^{*}\;}
$$

This is a **supercritical pitchfork** at the critical gain $g\lambda_0=1$. Above threshold the origin
becomes a saddle/repeller and two symmetric wells (the A/B memory poles) appear. The well depth and
$\kappa_0^{*}\propto\sqrt{g\lambda_0-1}$ near threshold. **DPA's job is to push $g\lambda_0$ above 1.**

The same calculation along $\kappa_1$ with $\lambda_1=\tfrac1N\|m_1\|^2$ gives a decision bistability when
$g\lambda_1>1$; off‑diagonal couplings $\lambda_{01}=\tfrac1N n_0^{\!\top}m_1$ etc. tilt and rotate the wells.

---

## 5. Stability, rotation, and why relu spirals

Linearising the flow $\dot\kappa\propto F$ at a fixed point gives the $2\times2$ matrix
$J=gG-\mathbb 1$, $G_{ac}=\tfrac1N\sum_i n_a\phi'_i m_c$. Eigenvalues are

$$
\mu_\pm=\tfrac12\mathrm{tr}J\pm\sqrt{\big(\tfrac12(J_{00}-J_{11})\big)^2+J_{01}J_{10}} .
$$

- **Real** eigenvalues (node) when $J_{01}J_{10}>-\big(\tfrac12(J_{00}-J_{11})\big)^2$.
- **Complex** eigenvalues (**spiral**) when the antisymmetric cross‑coupling dominates,
  $J_{01}J_{10}<0$ with large magnitude. This needs $G$ far from symmetric, i.e. $n\not\propto m$.

`tanh` saturates: $\phi'=1-\tanh^2\to0$ as $|g(b+M\kappa)|$ grows, so $G$ (hence the off‑diagonal
coupling) **shrinks radially** away from the origin — a radial damping that drives eigenvalues real.
Empirically **0 %** of the κ‑plane has complex eigenvalues for trained `tanh` nets (and raising $g$
*increases* saturation → more damping; that is why the old `gain=2` fix removed spiraling).

`relu` has $\phi'\in\{0,1\}$ with **no upper saturation**: for active units $\phi'=1$ regardless of
magnitude, so there is no radial damping; the hard on/off switching of units as $\kappa$ moves makes
$G$ rotate, leaving an undamped antisymmetric component → **15 %** of the plane spirals. A
*saturating* asymmetric unit (`tanh_asym`, §8) keeps the 0 %.

---

## 6. The slow manifold (deformed ring)

Near the pitchfork the curvature of $V$ *along the ring* is small. Concretely, the eigenvalue
governing motion **tangent** to the memory manifold is $\mu_\parallel = g\,G_{00}^{\rm eff}-1$. When
training leaves the network **near critical**, $g\lambda_0\gtrsim1$, this eigenvalue is $\approx0$
(flow) / $|\lambda_{\rm map}|=|1+\beta\mu_\parallel|\approx1$ (map). The memory states then form a
**slow manifold**: motion across the ring (radial) is fast and strongly contracting, motion along the
ring (tangential) is slow.

- If the ring were perfectly isotropic ($\lambda_0=\lambda_1$, no off‑diagonal coupling, exact
  rotational symmetry) the tangential eigenvalue would be exactly $0$ → a genuine **line/ring
  attractor** (continuum of fixed points).
- Real trained nets are anisotropic ($\lambda_0\neq\lambda_1$, $\lambda_{01}\neq0$): the ring tilts
  into a shallow landscape with a few discrete shallow wells and a **directed but slow tangential
  drift** $\propto$ the angular gradient of $V$ on the ring. Measured drift $\sim\!0.008\,\kappa$/step;
  the radial (across‑ring) velocity is $\sim\!10\times$ smaller than the tangential one — i.e. the
  ring is radially trapping, slowly sliding toward the discrete wells. On finite‑trial timescales any
  point on the arc behaves as a **quasi‑attractor**.

A fixed point whose slowest map eigenvalue is within $\sim$few % of the unit circle is therefore
better described as a *slow‑manifold sample* than a sharp attractor; this is what the `slow_attractor`
(orange) and `marginal` (gold) labels in the flow plots flag.

---

## 7. What each training stage does to the landscape

Throughout, the **autonomous** landscape ($x=0$) is the memory landscape; **input‑conditioned**
landscapes ($x$ clamped) shift the operating point inside $\phi$ and so reshape $V$ via $b(x)$.

### DPA — build the symmetric memory ring
- Trains $m_0,n_0$ so $g\lambda_0>1$: a **supercritical pitchfork** opens a symmetric double well in
  $\kappa_0$ (the A/B memory poles at $\pm\kappa_0^{*}$). $V$ even ⇒ wells symmetric about the origin.
- Trains the decision readout $n_1$ and the post‑test map (sample×test → match/nonmatch on $\kappa_1$).
- **Landscape:** even double well in $\kappa_0$, $\kappa_1$ slaved to the decision readout.
  **Attractors:** $\pm\kappa_0^{*}$ (memory). **Slow manifold:** present if training stops near
  critical $g\lambda_0\!\approx\!1$ (a soft ring).

### GNG — deform the ring, add the go/nogo decision (rank‑0 frozen)
- Rank‑0 of $m,n$ and the DPA input dims are **frozen**, so the $\kappa_0$ memory well is *protected*.
  Learning happens in $\kappa_1$ and in the off‑diagonal couplings $\lambda_{01},\lambda_{10}$.
- The go/nogo **input columns** $W_i[:,\text{go}],W_i[:,\text{nogo}]$ acquire large projections on the
  decision readout $n_1$ (measured: $\langle b_{\rm nogo},\hat n_1\rangle\!\approx\!-12.8$ vs
  $\langle b_{\rm go},\hat n_1\rangle\!\approx\!+3.9$). Under a clamped input the operating point shifts
  by $b$, which (i) drives $\kappa_1$ up (go) or down (nogo), and (ii) changes $\phi'$ (saturation).
- The off‑diagonal coupling **tilts/rotates** the wells (§4) — the ring *deforms* rather than breaks,
  exactly the desired "deformed but not destroyed". The shallow tilt is what makes the ring a slow
  manifold (§6).

### Dual — input‑driven attractors and the saturation collapse
- Combines DPA + GNG; now the *response‑period* input (cue/go/nogo) sets the operating point.
- **Saturation collapse (the key Dual mechanism).** Under a strong clamped input $b$, units saturate,
  $\phi'_i=1-\tanh^2(g(b+M\kappa)_i)\to0$ for the saturated population. The effective memory
  self‑coupling
  $$G_{00}=\tfrac{g}{N}\sum_i n_0[i]\,\phi'_i\,m_0[i]$$
  drops; once $G_{00}<1$ the quadratic coefficient $1-gG_{00}$ in §4 turns positive → the $\kappa_0$
  **double well flattens into a single well** → the condition's attractor becomes **unimodal** (memory
  collapsed *for that input*), while the autonomous field (no $b$) keeps $G_{00}>1$ and stays bistable.
  Measured: autonomous $G_{00}=2.58$, go $1.80$ (still bimodal), nogo $0.94$ (collapsed). Scaling all
  inputs $\times3$ pushes go below 1 too → go unimodal, autonomous bistable.
- So Dual attractors are **input‑conditioned minima** of $V(\kappa; b(x))$: go/nogo sit at $\kappa_1\gtrless0$;
  whether $\kappa_0$ stays split (bimodal) or collapses (unimodal) is governed by $G_{00}(b)\lessgtr1$.

---

## 8. Corollary: what can move the wells into the lower plane

Because the wells are minima of an **even** $V$, lowering them all to $\kappa_1<0$ requires breaking
$V(-\kappa)=V(\kappa)$ **in the readout plane**. Three routes, and what the theory predicts:

1. **A constant readout‑aligned bias** $b$ with $\langle b,n_1\rangle<0$: adds an *even* correction
   $\sim\tfrac{g}{N}n^{\!\top}\!\mathrm{diag}(\phi')b$ to $\Psi$ that shifts the ring centre down.
   *Imposed*; also makes the resting state carry a standing decision. (A generic/random bias the
   network can rotate $\perp n$ → no shift, confirmed by `unit_bias`.)
2. **An even component in $\phi$** (e.g. `tanh_asym` $\phi=\tanh+\gamma\tanh^2$): adds
   $\tfrac{\gamma}{N}n^{\!\top}\tanh^2(gM\kappa)$, an even‑in‑$\kappa$ term that is *non‑removable*
   (tied to $m,n$, not a free vector). Breaks the $\pm$ pairing — but with **no preferred sign**, so the
   ring de‑centres without a systematic downward shift (confirmed empirically). Saturating ⇒ no spiral.
3. **Non‑negative rates (EI / relu):** $\phi\ge0$ makes $\Phi$ non‑even, breaking the symmetry with a
   *fixed preferred side* set by the structure. This is the only route that is both directional and
   structural — at the cost of relu's spiraling/non‑saturation, which is why the EI model uses a strong
   static backbone (to bound the dynamics) and, ultimately, short‑term plasticity (to turn the drifting
   transient into a persistent attractor; see [`ring_lowerplane_log.md`](ring_lowerplane_log.md) §9d).

---

## Assumptions & caveats
- **Adiabatic reduction** $h=M\kappa$: exact at fixed points; transiently the two timescales
  ($\alpha$ rates, $\alpha_{\rm rec}$ recurrent) differ, which can shift *stability* (not location).
- **Gradient/potential picture** is exact only for $n=M$; for $n\neq M$ keep the symmetric part of $G$
  for wells and the antisymmetric part for rotation (§5).
- **Bias drops out** of the symmetry argument only because $b\perp n$ in trained nets ($\Psi(0)\approx0$);
  a readout‑aligned bias is precisely route (1) above.
- The **EI model** has a dense static backbone, so its κ‑reduction is *not* the clean rank‑2 form here;
  the qualitative landscape language (wells, slow manifold, saturation collapse) still applies but the
  $G$/$V$ formulas must include the backbone.
