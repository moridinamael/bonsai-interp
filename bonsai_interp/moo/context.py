"""Controlled context experiments: which parts of a passage's context cause a pole to fire, for up to 99 poles at once.

Corpus text can only say what goes with a pole (compose@16: mostly not what causes it). Here the read token and its context
are the experiment's factors. A snippet is the CTX tokens ending at one of a pole's sampled strong activations, read at
its last token. Every token belongs to one factor and has a fixed random rank in (0, 1]. An input g in [0, 1]^F says how
much of each factor stays: a token stays when its rank is <= its factor's g, and otherwise it is replaced by the token at the
same position of a donor passage (a random passage of the same domain). So every input is real tokens: g = 1 is the original
text, g = 0 is the donor passage, and between them a factor's tokens are swapped one by one. There is no gradient to g.

Factors (24): the read token; the previous token; other pieces of the read token's word; earlier copies of the read token;
then, for 2-4, 5-16, 17-48 and 49-95 tokens back, five classes: content, name, function, punct, other (digits, layout).

Output per pole: the mean activation at the read token over its snippets, as a share of the same snippet at g = 1.
"""
import json
import time

import numpy as np

from bonsai_interp import explore
from bonsai_interp.auto import context_tier as ct
from bonsai_interp.events import sample as EV

CTX = 96
CLASSES = ["content", "name", "function", "punct", "other"]
BANDS = [(2, 4), (5, 16), (17, 48), (49, 95)]
FACTORS = ["read", "prev", "same word", "copy"] + [f"{a}-{b}:{c}" for a, b in BANDS for c in CLASSES]
FUNCTION = set("""a an the this that these those of in on at by for from to with without into onto over under about after before
between through during against and or but nor so yet if then than as because while although though when where which who whom
whose what why how i you he she it we they me him her us them my your his its our their mine yours is am are was were be been
being do does did have has had will would can could shall should may might must not no s t re ve ll d""".split())
N = 12288


def _class(text):
    t = text.strip()
    if not t or t.isdigit():
        return 4
    if not any(c.isalnum() for c in t):
        return 3
    if t.lower() in FUNCTION:
        return 2
    return 1 if t[0].isupper() else 0


def _continues(text):
    """True when a token continues the previous token's word (no leading space, starts a letter or digit)."""
    return bool(text) and text[0].isalnum()


def _lut(ids, fn):
    tk = explore.tokenizer()
    u, inv = np.unique(ids, return_inverse=True)
    return np.array([fn(tk.decode([int(t)])) for t in u])[inv].reshape(ids.shape)


def factor_ids(snips):
    """(S, CTX) factor index of every token of every snippet."""
    S, T = snips.shape
    cl, gl = _lut(snips, _class), _lut(snips, _continues)
    F = np.zeros((S, T), np.int8)
    for b, (lo, hi) in enumerate(BANDS):
        for off in range(lo, hi + 1):
            F[:, T - 1 - off] = 4 + 5 * b + cl[:, T - 1 - off]
    F[snips == snips[:, -1:]] = 3  # copies of the read token, at any distance
    for s in range(S):  # pieces of the read token's word, back from the previous token
        k = T - 1
        while k >= 1 and gl[s, k]:
            F[s, k - 1] = 2
            k -= 1
    F[:, -2] = 1
    F[:, -1] = 0
    return F


class ContextExperiment:
    """The model forward for g, reading one layer's poles at the last token of each snippet."""

    def __init__(self, model, rec, L, poles, snips, wins, seed=0, batch=96):
        """poles: (S, 2) array of neuron, sign per snippet; snips: (S, CTX) token ids; wins: the corpus window of each."""
        import torch
        self.torch, self.model, self.rec, self.L, self.batch = torch, model, rec, L, batch
        self.dev = next(model.parameters()).device
        self.ids = torch.as_tensor(snips.astype(np.int64), device=self.dev)
        self.F = torch.as_tensor(factor_ids(snips).astype(np.int64), device=self.dev)
        self.n = torch.as_tensor(poles[:, 0].astype(np.int64), device=self.dev)
        self.sign = torch.as_tensor(poles[:, 1].astype(np.float32), device=self.dev)
        self.donor = torch.as_tensor(donors(snips, wins, seed).astype(np.int64), device=self.dev)
        rank = 1 - np.random.default_rng([53, seed]).random(snips.shape)  # (0, 1]
        self.rank = torch.as_tensor(rank.astype(np.float32), device=self.dev)

    def read(self, g, idx=None):
        """Activation (signed to the pole) of the snippets idx (default all) at g, a vector of len(FACTORS)."""
        torch, rec, L = self.torch, self.rec, self.L
        from bonsai_interp.model import StopForward
        gt = torch.as_tensor(np.asarray(g, np.float32), device=self.dev)
        idx = torch.arange(len(self.ids), device=self.dev) if idx is None else torch.as_tensor(idx, device=self.dev)
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), self.batch):
                k = idx[s:s + self.batch]
                x = torch.where(self.rank[k] <= gt[self.F[k]], self.ids[k], self.donor[k])
                rec.mode, rec.layers, rec.stop_at = "act", {L}, L
                try:
                    self.model.model(x)
                except StopForward:
                    pass
                finally:
                    rec.mode, rec.stop_at = None, None
                a = rec.store.pop(L)[:, -1, :].float()
                rec.clear()
                out.append(a[torch.arange(len(k), device=self.dev), self.n[k]] * self.sign[k])
        return torch.cat(out).cpu().numpy()


def donors(snips, wins, seed=0):
    """A donor passage per snippet: CTX tokens from a random window of the same domain as the snippet's own window."""
    toks = explore.corpus()
    dom = ct._domains()
    rng = np.random.default_rng([47, seed])
    pool = {d: np.nonzero(dom == d)[0] for d in np.unique(dom)}
    out = np.zeros_like(snips)
    for s, w0 in enumerate(wins):
        w = rng.choice(pool[dom[w0]])
        q = int(rng.integers(CTX, 1024))
        out[s] = toks[w, q - CTX:q]
    return out


def pick(L, n_poles, per_pole, seed):
    """Up to 4 x n_poles unassigned poles of layer L with >= 32 sampled activations, each with enough events far enough
    into a window: [(neuron, sign, event ids, event values)], in a seeded order."""
    from bonsai_interp.compose import round3 as R3
    z = EV.load()
    ids, vals = z["ids"][L], z["vals"][L]
    asg = R3._assigned()
    rng = np.random.default_rng([41, L, seed])
    out = []
    for i in rng.permutation(2 * N):
        p, n = int(i) // N, int(i) % N
        sg = "+" if p == 0 else "-"
        if (L, n, sg) in asg:
            continue
        e, v = ids[p, n], vals[p, n]
        keep = (e >= 0) & (e % 1024 >= CTX - 1)
        if (e >= 0).sum() < 32 or keep.sum() < 2 * per_pole:
            continue
        out.append((n, sg, e[keep], np.asarray(v[keep], float)))
        if len(out) >= 4 * n_poles:
            break
    return out


def setup(model, rec, L, n_poles=99, per_pole=6, seed=0):
    """(experiment, baseline per snippet, poles): poles whose truncated snippets keep firing (the last token at CTX context
    reaches >= 0.5 of its full-window activation on per_pole snippets)."""
    toks = explore.corpus()
    cand = pick(L, n_poles, per_pole, seed)
    snips, pol, ev = [], [], []
    for c, (n, sg, e, v) in enumerate(cand):
        for x, val in zip(e[:2 * per_pole], v[:2 * per_pole]):
            w, q = int(x) // 1024, int(x) % 1024
            snips.append(toks[w, q - CTX + 1:q + 1])
            pol.append((n, 1.0 if sg == "+" else -1.0))
            ev.append((c, val, w))
    snips, pol, ev = np.array(snips), np.array(pol), np.array(ev)
    b = ContextExperiment(model, rec, L, pol, snips, ev[:, 2].astype(np.int64), seed).read(np.ones(len(FACTORS)))
    top = np.load(ct.ROOT / "data" / "scan_bonsai1.npz")["top_val"][L]
    full = ev[:, 1] * np.abs(top[[0 if s > 0 else 1 for s in pol[:, 1]], 0, pol[:, 0].astype(int)])
    kept = b >= 0.5 * full
    print(f"truncated to {CTX} tokens: {kept.mean():.0%} of {len(b)} snippets keep >= 0.5 of the full-window activation; "
          f"median ratio {np.median(b / full):.2f}", flush=True)
    sel, poles = [], []
    for c, (n, sg, _, _) in enumerate(cand):
        k = np.nonzero((ev[:, 0] == c) & kept)[0][:per_pole]
        if len(k) == per_pole and len(poles) < n_poles:
            poles.append((n, sg))
            sel.append(k)
    sel = np.concatenate(sel)
    return ContextExperiment(model, rec, L, pol[sel], snips[sel], ev[sel, 2].astype(np.int64), seed), b[sel], poles


def outputs(a, base, per_pole):
    """(poles,) mean share of the baseline over each pole's snippets."""
    return (a / base).reshape(-1, per_pole).mean(1)


def precheck(L, n_poles=99, per_pole=6, seed=0, out=None):
    """No globalMOO: is there anything for a solver to find? One knockout per factor, block knockouts and random designs."""
    t0 = time.time()
    model, tok, rec = ct._gpu_load()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    X, base, poles = setup(model, rec, L, n_poles, per_pole, seed)
    print(f"{len(poles)} poles, {len(base)} snippets", flush=True)
    t1 = time.time()
    one = np.ones(len(FACTORS))
    y1 = outputs(X.read(one), base, per_pole)
    sec = time.time() - t1
    print(f"one evaluation: {sec:.1f} s; g = 1 output mean {y1.mean():.3f}", flush=True)
    res = {"poles": [f"N{n}{s}" for n, s in poles], "factors": FACTORS, "sec_per_eval": sec}
    knock = {}
    for i, f in enumerate(FACTORS):
        g = one.copy()
        g[i] = 0
        knock[f] = outputs(X.read(g), base, per_pole)
    only_read = np.zeros(len(FACTORS))
    only_read[0] = 1
    near = one.copy()
    near[1:14] = 0
    far = one.copy()
    far[14:] = 0
    local = one.copy()
    local[4:] = 0
    blocks = {"all donor": 0 * one, "read token only": only_read, "no read token": 1 - only_read, "read, prev, word, copy only": local,
              "no 17+ back": far, "no prev to 16 back": near}
    bres = {k: outputs(X.read(v), base, per_pole) for k, v in blocks.items()}
    rng = np.random.default_rng([43, L, seed])
    G = (rng.random((96, len(FACTORS))) < 0.5).astype(float)
    G[:, 0] = 1  # the read token stays: its effect is known, and the rest is the question
    Y = np.array([outputs(X.read(g), base, per_pole) for g in G])
    A = np.c_[np.ones(64), G[:64, 1:]]
    pred = np.c_[np.ones(32), G[64:, 1:]] @ np.linalg.lstsq(A, Y[:64], rcond=None)[0]
    r2 = 1 - ((Y[64:] - pred) ** 2).sum(0) / np.maximum(((Y[64:] - Y[64:].mean(0)) ** 2).sum(0), 1e-9)
    res.update({"knockout": {k: v.tolist() for k, v in knock.items()}, "blocks": {k: v.tolist() for k, v in bres.items()},
                "G": G.tolist(), "Y": Y.tolist(), "additive_r2": r2.tolist()})
    if out:
        json.dump(res, open(out, "w", encoding="utf-8"))
    print(f"\nmean output by single knockout (1 = no effect), L{L}:")
    for f in FACTORS:
        print(f"  {f:18s} mean {knock[f].mean():.2f}  poles below 0.5: {(knock[f] < 0.5).sum()}")
    print("blocks:")
    for k, v in bres.items():
        print(f"  {k:28s} median {np.median(v):.2f}  poles below 0.5: {(v < 0.5).sum()}  above 0.8: {(v >= 0.8).sum()}")
    print(f"random designs: output sd median {np.median(Y.std(0)):.3f}, mean output {Y.mean():.2f}; held-out additive R2 median "
          f"{np.median(r2):.2f}, poles under 0.5: {(r2 < 0.5).sum()}")
    single_ok = np.min([knock[f] for f in FACTORS[1:]], axis=0) >= 0.7
    print(f"redundancy: {single_ok.sum()} poles survive every single context knockout (>= 0.7); of those, "
          f"{(single_ok & (bres['read, prev, word, copy only'] < 0.5)).sum()} fall below 0.5 with all 2+ back context off, "
          f"{(single_ok & (bres['read token only'] < 0.5)).sum()} with all context off")
    print(f"total {time.time() - t0:.0f} s", flush=True)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--poles", type=int, default=99)
    ap.add_argument("--per-pole", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args()
    precheck(a.layer, a.poles, a.per_pole, out=a.out)
