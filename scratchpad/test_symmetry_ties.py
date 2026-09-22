"""Test every tie kind of project_symmetry / symmetrize_init against the group action it is supposed to implement
(not against its own code): exact field equivariance F(Dκ; Sx) = D F(κ; x) with random inputs and bias, idempotence,
orthogonality of the projection, and an untied control that must fail. Run: LD_PRELOAD=... python scratchpad/test_symmetry_ties.py
(2026-09-22, ring log §37h addendum). All kinds pass at float32 round-off (~3e-8); the untied control gives ~0.25."""
import sys, math, torch; sys.path.insert(0, "/home/leon/rnn"); from src.train import project_symmetry, symmetrize_init
from src.models import LowRankModel
torch.manual_seed(0); N, C = 64, 8
phi = lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2)))
def field(model, k, x):
    m, n, W, b, g = model.m, model.n, model.wi.weight, model.wi.bias, float(model.gain)
    return phi(g * (k @ m.T + x @ W.T + b)) @ n / N - k
def perm_swap(C, pairs):
    S = torch.eye(C)
    for a, b in pairs: S[[a, b]] = S[[b, a]]
    return S
D1, D3, D2 = torch.diag(torch.tensor([-1., 1.])), torch.diag(torch.tensor([1., -1.])), -torch.eye(2)
KINDS = {"pair": ([D1], [perm_swap(C, [(0, 1), (2, 3)])]), "test": ([D3], [perm_swap(C, [(2, 3)])]), "inv": ([D2], [perm_swap(C, [(0, 1)])]),
         "gng": ([D2], [perm_swap(C, [(4, 5)])]), "gng_dec": ([D3], [perm_swap(C, [(4, 5)])]), "gng_mem": ([D1], [perm_swap(C, [(4, 5)])]),
         "klein": ([D1, D3], [perm_swap(C, [(0, 1), (2, 3)]), perm_swap(C, [(2, 3)])]), "gng_klein": ([D1, D3], [torch.eye(C), perm_swap(C, [(4, 5)])])}
def fresh():
    mdl = LowRankModel(input_size=C, hidden_size=N, output_size=0, rank=2, gain=1.0, alpha=0.075, alpha_rec=0.1, noise=0.0, nonlinearity="lif", use_unit_bias=False, unit_bias_trainable=False, device="cpu")
    with torch.no_grad():
        for p in (mdl.m, mdl.n, mdl.wi.weight, mdl.wi.bias): p.copy_(torch.randn_like(p) * 2)
    return mdl
def flat(mdl): return torch.cat([p.detach().flatten() for p in (mdl.m, mdl.n, mdl.wi.weight, mdl.wi.bias)])
k = torch.randn(200, 2); x = torch.randn(200, C); ok = True
for kind, (Ds, Ss) in KINDS.items():
    mdl = fresh(); theta0 = flat(mdl).clone(); project_symmetry(mdl, kind); theta1 = flat(mdl).clone()
    project_symmetry(mdl, kind); idem = (flat(mdl) - theta1).abs().max().item()
    mdl2 = fresh(); project_symmetry(mdl2, kind); orth = float(torch.dot(theta0 - theta1, flat(mdl2)) / (theta0 - theta1).norm() / flat(mdl2).norm())
    mdl3 = fresh(); symmetrize_init(mdl3, kind)
    errs = [((field(m_, k @ D.T, x @ S.T) - field(m_, k, x) @ D.T).norm() / (field(m_, k, x) @ D.T).norm()).item() for m_ in (mdl, mdl3) for D, S in zip(Ds, Ss)]
    good = max(errs) < 1e-5 and idem == 0.0 and abs(orth) < 1e-6; ok &= good
    print(f"{kind:10s} equivariance {['%.1e' % e for e in errs]}  idempotent {idem:.1e}  orthogonal cos {orth:+.1e}  {'OK' if good else 'FAIL'}")
mdl = fresh(); D, S = KINDS["pair"][0][0], KINDS["pair"][1][0]; ctrl = ((field(mdl, k @ D.T, x @ S.T) - field(mdl, k, x) @ D.T).norm() / (field(mdl, k, x) @ D.T).norm()).item()
print(f"untied control (pair): {ctrl:.2f} (must be large)"); ok &= ctrl > 0.05
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
