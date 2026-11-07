"""steer@3: where in depth to steer. One globalMOO model over 4 concepts x 6 depth bands, against an additive model of the bands.

  python -m bonsai_interp.moo.steer3 protocol   # writes steer3_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer3 train      # the design of experiments and one model
  python -m bonsai_interp.moo.steer3 gmoo       # every task at every tier through the model
  python -m bonsai_interp.moo.steer3 baselines  # uniform line, band lines, the additive model, differential evolution
  python -m bonsai_interp.moo.steer3 heldout    # each method's best designs on the held-out prompts
  python -m bonsai_interp.moo.steer3 report
  python -m bonsai_interp.moo.steer3 protocol3b / requery / heldout / report3b   # steer@3b, below

steer@2's 24 concept knobs added up: in its design an additive fit predicted every output as well as a quadratic one, so a plain
gain line found the sparse answers. Here each concept's bundle is split by depth. Pilots on the training prompts showed the bands
differ by an order of magnitude in how much steering they buy per unit of lost text quality, and their responses rise and fall,
so the best allocation over depth is not a line, and whether bands add or interact is the open question.

Inputs: u in [-1, 1] per band; the gain added to the band's poles is SCALE x HI[band] x sign(u) |u|^3. Outputs: log10 soft rate
of all 24 concepts, the distinct-token ratio, coherence under the clean model, and drift (mean absolute log10 change of the 20
concepts outside the panel against zero gain). A task raises some panel concepts and lowers others. Each cell is one task at one
tier of text quality, with drift at most DRIFT: four objectives pulling against each other. Its score is the smallest log10 fold
change over the task's concepts, on the best evaluated point that meets the cell, the same for every method.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal, InverseProblem, Var
from bonsai_interp.moo import steer as T
from bonsai_interp.moo import steer_moo as M

OUT = ct.ROOT / "data" / "moo"
PROTOCOL = Path(__file__).parent / "steer3_protocol.json"
EVALS = OUT / "steer3.evals.jsonl"
META = OUT / "steer3.model.json"
CON = ["medicine", "weather", "money", "travel"]
BANDS = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 24), (24, 36)]
HI = [.15, .15, .6, 1.6, 1.0, .5]  # per band: about where one band alone starts to break the text (pilot 2)
SCALE, POWER, NEW, BATCH, DRIFT = .6, 3, 24, 6, .20
TASKS = [{"up": ["medicine"], "down": []}, {"up": ["weather"], "down": []}, {"up": ["money"], "down": []}, {"up": ["travel"], "down": []},
         {"up": ["medicine", "weather"], "down": []}, {"up": ["money", "travel"], "down": []},
         {"up": ["weather"], "down": ["medicine"]}, {"up": ["travel"], "down": ["money"]}]
QUERY_ITER, DE_MAXITER = 12, 3
LINE = tuple(round(.01 * i, 2) for i in range(1, 13))  # steer@2's knob: one raw gain on every band, may leave the box
BAND_U = (-1, -.6, .2, .4, .6, .8, 1)  # one band alone, the rest at zero: the additive model is built from these


def gains(U):
    """Raw gains (designs, 24) from inputs u."""
    U = np.asarray(U, float)
    return SCALE * np.tile(HI, len(CON)) * np.sign(U) * np.abs(U) ** POWER


def to_u(G):
    G = np.asarray(G, float)
    return np.sign(G) * (np.abs(G) / (SCALE * np.tile(HI, len(CON)))) ** (1 / POWER)


def knob(c, b):
    return CON.index(c) * len(BANDS) + b


def protocol(a):
    from bonsai_interp import explore
    from bonsai_interp.compose import round3 as R3
    B, _ = T.bundles(explore.tokenizer(), R3._assigned())
    split = [sorted(q for q in B[c] if lo <= q[0] < hi) for c in CON for lo, hi in BANDS]
    sha = hashlib.sha256(json.dumps(split).encode()).hexdigest()
    json.dump({"name": "steer@3", "concepts": CON, "bands": BANDS, "hi": HI, "scale": SCALE, "power": POWER, "new_tokens": NEW,
               "drift_max": DRIFT, "tasks": TASKS, "tiers": M.TIERS, "query_iter": QUERY_ITER, "de_maxiter": DE_MAXITER, "line": LINE,
               "band_u": BAND_U, "train_prompts": M.TRAIN, "heldout_prompts": M.HELD, "bundles_sha256": sha,
               "pilots": "training prompts only: per-band gain ladders for the 4 concepts set HI; five box scalings on 24 random designs "
                         "each chose SCALE .6 with a cube (uniq 5/50/95 .48/.62/.72; tiers met 4, 13, 23 of 24)",
               "methods": {"gmoo": "design of experiments, one model, then each cell as one query: maximize each raised concept, minimize "
                                   "each lowered one, uniq and coh at least the tier, drift at most drift_max; query_iter evaluations",
                           "additive": "the band lines give each output as base plus a sum of one-band curves (linear interpolation in u); "
                                       "each cell is solved on that model by differential evolution on the baselines' loss, the answer evaluated, the model shifted "
                                       "by the observed residual and solved again, query_iter evaluations per cell",
                           "line": "steer@2's knob: one raw gain on every band of the task's concepts (negative for a lowered one), "
                                   "the line grid per task",
                           "band": "the best band-line point, one band of one concept at a time",
                           "de": "differential evolution on the cell at tier 2 (the baselines' loss: minus the score plus a steep penalty outside the cell), de_maxiter generations of 24, per task",
                           "lookup": "the best point of the design of experiments"},
               "claims": {"depth": "at tier 2, on the held-out prompts, for at least 3 of the 4 single-concept tasks the best of the other "
                                   "methods' designs beats the uniform line's best design by at least log10(2)",
                          "model": "over the 24 task-tier cells on the held-out prompts, globalMOO beats the additive model in more cells than "
                                   "it loses; the same count against line, band, de and lookup is reported whichever way it falls"}},
              open(PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("wrote", PROTOCOL, "bundles", sha[:12])


class Lab:
    def __init__(self):
        from bonsai_interp.compose import round3 as R3
        self.t0 = time.time()
        self.p = json.load(open(PROTOCOL, encoding="utf-8"))
        model, self.tok, rec = ct._gpu_load(lm_head=True)
        B, toks = T.bundles(self.tok, R3._assigned())
        split = [sorted(q for q in B[c] if lo <= q[0] < hi) for c in CON for lo, hi in BANDS]
        assert hashlib.sha256(json.dumps(split).encode()).hexdigest() == self.p["bundles_sha256"], "the bundles differ from the protocol's"
        pm = np.stack([np.stack(ct._pole_max(L)) for L in range(36)]).astype(np.float32)
        self.panel = T.Panel(model, rec, pm, split)
        self.names = list(T.CONCEPTS)
        self.ids = [torch.tensor(toks[c]) for c in self.names]
        self.other = [i for i, c in enumerate(self.names) if c not in CON]
        self.outputs = [f"s_{c}" for c in self.names] + ["uniq", "coh", "drift"]
        self.cache, self.tag, self.prompts, self.zero = {}, "", M.TRAIN, {}
        OUT.mkdir(exist_ok=True)
        if EVALS.exists():
            for line in open(EVALS, encoding="utf-8"):
                r = json.loads(line)
                self.cache[(r["set"], tuple(r["x"]))] = np.array(r["y"])
        self.problem = InverseProblem("steer3", [Var(f"{c}_L{lo}", -1.0, 1.0, "float") for c in CON for lo, _ in BANDS],
                                      self.outputs, self.evaluate)

    def _read(self, U):
        D = len(U)
        e, g, lp = T.generate(self.panel, self.tok, self.prompts, gains(U), NEW)
        P = g.shape[1]
        soft = torch.stack([lp[:, :, i].exp().sum(-1).mean(1) for i in self.ids], 1).view(D, P, -1).mean(1).numpy()
        uniq = np.array([[len(set(x.tolist())) / g.shape[2] for x in row] for row in g.cpu()]).mean(1)
        coh = T.clean_logp(self.panel, e, g).mean(1).cpu().numpy()
        return np.c_[np.log10(soft + M.FLOOR), uniq, coh]

    def evaluate(self, U):
        """Outputs for designs given as u (designs, 24); drift is against the zero design on the same prompts."""
        U = np.round(np.asarray(U, float), 4)
        s = "train" if self.prompts is M.TRAIN else "held"
        if s not in self.zero:
            k = (s, tuple([0.0] * 24))
            if k not in self.cache:
                y = self._read(np.zeros((1, 24)))[0]
                self.cache[k] = np.r_[y, 0.0]
                self._log(s, np.zeros(24), self.cache[k])
            self.zero[s] = self.cache[k]
        z = self.zero[s]
        out = np.zeros((len(U), 27))
        todo = []
        for i, u in enumerate(U):
            k = (s, tuple(u.tolist()))
            if k in self.cache:
                out[i] = self.cache[k]
            else:
                todo.append(i)
        for j in range(0, len(todo), BATCH):
            ix = todo[j:j + BATCH]
            y = self._read(U[ix])
            y = np.c_[y, np.abs(y[:, self.other] - z[self.other]).mean(1)]
            out[ix] = y
            for i, v in zip(ix, y):
                self.cache[(s, tuple(U[i].tolist()))] = v
                self._log(s, U[i], v)
            if j % 120 == 0 and len(todo) > 12:
                print(f"{j} of {len(todo)} ({time.time() - self.t0:.0f} s)", flush=True)
        return out

    def _log(self, s, u, y):
        with open(EVALS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"set": s, "tag": self.tag, "x": [float(v) for v in u], "y": [round(float(w), 4) for w in y]}) + "\n")


def cell_goals(lab, task, tier):
    g = [Goal(f"s_{c}", "max", 0.0) for c in task["up"]] + [Goal(f"s_{c}", "min", 0.0) for c in task["down"]]
    return g + [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", DRIFT)]


def meets(y, tier):
    return y[24] >= tier["uniq"] and y[25] >= tier["coh"] and y[26] <= DRIFT


def score(y, task, base, names):
    """Smallest log10 fold change over the task's concepts: up for a raised one, down for a lowered one."""
    return min([y[names.index(c)] - base[names.index(c)] for c in task["up"]] + [base[names.index(c)] - y[names.index(c)] for c in task["down"]])


def train(a):
    lab = Lab()
    lab.tag = "doe"
    y0 = lab.problem.evaluate(np.zeros((1, 24)))[0]
    print("base uniq, coh", np.round(y0[24:26], 3), flush=True)
    inv = S.GmooInverse(lab.problem, "mono-steer3")
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals,
               "doe_seconds": inv.doe_seconds}, open(META, "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def gmoo(a):
    lab = Lab()
    m = json.load(open(META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    for ti, task in enumerate(TASKS):
        if ti < a.start:
            continue
        for tr, tier in enumerate(M.TIERS):
            lab.tag = f"gmoo/{ti}/{tr}"
            n0 = len(lab.problem.history)
            res = inv.query(cell_goals(lab, task, tier), max_iter=QUERY_ITER)
            print(lab.tag, res.extra.get("stopped"), len(lab.problem.history) - n0, "evals", flush=True)


class Additive:
    """Each output as the zero design plus a sum of one-band curves, from the band lines."""

    def __init__(self, lab):
        self.z = lab.problem.evaluate(np.zeros((1, 24)))[0]
        us = np.array(sorted(set(BAND_U) | {0.0}))
        self.us, self.curve = us, np.zeros((24, len(us), 27))
        for k in range(24):
            X = np.zeros((len(us), 24))
            X[:, k] = us
            self.curve[k] = lab.problem.evaluate(X) - self.z
        self.shift = np.zeros(27)

    def __call__(self, U):
        U = np.atleast_2d(U)
        out = np.tile(self.z + self.shift, (len(U), 1))
        for k in range(24):
            for j in range(27):
                out[:, j] += np.interp(U[:, k], self.us, self.curve[k, :, j])
        return out


def cell_loss(y, task, tier, base, names):
    """What the baselines minimize: minus the cell's score, plus a steep penalty for each unit outside the cell."""
    out = max(0, tier["uniq"] - y[24]) / .05 + max(0, tier["coh"] - y[25]) / .2 + max(0, y[26] - DRIFT) / .05
    return -score(y, task, base, names) + 50 * out


def _de(f, task, tier, base, names, seed, maxiter, popsize):
    from scipy import optimize
    r = optimize.differential_evolution(lambda X: np.array([cell_loss(y, task, tier, base, names) for y in f(X.T)]), [(-1, 1)] * 24,
                                        maxiter=maxiter, popsize=popsize, vectorized=True, seed=seed, polish=False, updating="deferred", tol=0)
    return r.x


def baselines(a):
    lab = Lab()
    names = lab.names
    lab.tag = "band"
    add = Additive(lab)
    for ti, task in enumerate(TASKS):
        lab.tag = f"line/{ti}"
        G = np.zeros((len(LINE), 24))
        for i, g in enumerate(LINE):
            for c in task["up"]:
                G[i, [knob(c, b) for b in range(len(BANDS))]] = g
            for c in task["down"]:
                G[i, [knob(c, b) for b in range(len(BANDS))]] = -g
        lab.problem.evaluate(np.round(to_u(G), 4))
        for tr, tier in enumerate(M.TIERS):
            lab.tag = f"additive/{ti}/{tr}"
            add.shift[:] = 0
            for it in range(QUERY_ITER):
                u = _de(add, task, tier, add.z, names, 1000 * ti + 10 * tr + it, 60, 10)
                y = lab.problem.evaluate(u[None])[0]
                add.shift += y - add(u)[0]
        lab.tag = f"de/{ti}"
        _de(lab.problem.evaluate, task, M.TIERS[1], add.z, names, ti, DE_MAXITER, 1)
        print("task", ti, "done", round(time.time() - lab.t0), "s", flush=True)


def best_points():
    """{(method, task, tier): (score, u)} over each method's training evaluations."""
    names, rows = list(T.CONCEPTS), [json.loads(s) for s in open(EVALS, encoding="utf-8")]
    rows = [r for r in rows if r["set"] == "train"]
    base = next(np.array(r["y"]) for r in rows if not any(r["x"]))
    doe = [r for r in rows if r["tag"] == "doe"]
    best = {}
    for ti, task in enumerate(TASKS):
        pools = {"lookup": doe, "gmoo": doe + [r for r in rows if r["tag"].startswith(f"gmoo/{ti}/")],
                 "additive": [r for r in rows if r["tag"] == "band" or r["tag"].startswith(f"additive/{ti}/")],
                 "band": [r for r in rows if r["tag"] == "band"], "line": [r for r in rows if r["tag"] == f"line/{ti}"],
                 "de": [r for r in rows if r["tag"] == f"de/{ti}"],
                 "target": doe + [r for r in rows if r["tag"].startswith(f"target/{ti}/")],
                 "center": [r for r in rows if r["tag"].startswith(f"center/{ti}/")]}
        for m, pool in pools.items():
            for tr, tier in enumerate(M.TIERS):
                ok = [(score(r["y"], task, base, names), r["x"]) for r in pool if meets(r["y"], tier)]
                best[(m, ti, tr)] = max(ok, key=lambda v: v[0]) if ok else (None, None)
    return names, base, best


METHODS = ("gmoo", "additive", "line", "band", "de", "lookup")


def heldout(a):
    _, _, best = best_points()
    lab = Lab()
    lab.prompts, lab.tag = M.HELD, "heldout"
    X = sorted({tuple(v[1]) for v in best.values() if v[1] is not None})
    print(len(X), "designs on held-out prompts", flush=True)
    lab.problem.evaluate(np.array(X))


def report(a):
    names, base, best = best_points()
    held = {tuple(r["x"]): np.array(r["y"]) for r in map(json.loads, open(EVALS, encoding="utf-8")) if r["set"] == "held"}
    b0 = held[tuple([0.0] * 24)]
    tags = [json.loads(s)["tag"].split("/")[0] for s in open(EVALS, encoding="utf-8") if json.loads(s)["set"] == "train"]
    print("training evaluations by tag", {t: tags.count(t) for t in sorted(set(tags))})

    def hs(m, t, tr):
        u = best[(m, t, tr)][1]
        return None if u is None else score(held[tuple(u)], TASKS[t], b0, names)
    print("\nscore by cell, training / held-out (None: no design met the cell)")
    for t, task in enumerate(TASKS):
        for tr in range(3):
            print(f"  task {t} {'+'.join(task['up'])}{' -' + '-'.join(task['down']) if task['down'] else ''} tier {tr + 1}: " +
                  "  ".join(f"{m} {_f(best[(m, t, tr)][0])}/{_f(hs(m, t, tr))}" for m in METHODS))
    print("\ngmoo against each method, cells won / lost / tied (no design loses to any design)")
    for nm, f in (("training", lambda m, t, tr: best[(m, t, tr)][0]), ("held-out", hs)):
        for m in METHODS[1:]:
            w = l = e = 0
            for t in range(len(TASKS)):
                for tr in range(3):
                    x, y = f("gmoo", t, tr), f(m, t, tr)
                    x, y = -9 if x is None else x, -9 if y is None else y
                    w, l, e = w + (x > y + 1e-3), l + (y > x + 1e-3), e + (abs(x - y) <= 1e-3)
            print(f"  {nm} vs {m}: {w} / {l} / {e}")
    print("\ndepth claim: single-concept tasks at tier 2, held-out: best other method vs the uniform line")
    passed = 0
    for t in range(4):
        other = max((hs(m, t, 1) for m in METHODS if m != "line" and hs(m, t, 1) is not None), default=None)
        ln = hs("line", t, 1)
        ok = other is not None and (ln is None or other - ln >= np.log10(2))
        passed += ok
        print(f"  {CON[t]}: other {_f(other)} line {_f(ln)} {'pass' if ok else 'miss'}")
    print(f"depth: {passed} of 4 (claim needs 3)")
    print("\nhow the best tier-2 designs spread gain over depth (mean |u| per band over the task's concepts)")
    for m in ("gmoo", "additive", "de"):
        for t in range(4):
            u = best[(m, t, 1)][1]
            if u is not None:
                print(f"  {m:8s} {CON[t]:9s}", np.round(np.abs(np.array(u)[[knob(CON[t], b) for b in range(6)]]), 2))


# steer@3b: steer@3's queries tied the design lookup in 23 of 24 cells; are maximize objectives the reason? Same trained model,
# same cells, each asked for a value just past the best design point instead. "target" starts at that design point; "center"
# starts at the zero design, so only the inverse can find a good design.

PROTOCOL3B = Path(__file__).parent / "steer3b_protocol.json"
REACH = np.log10(1.5)  # aim this far past the best design point's score


def protocol3b(a):
    json.dump({"name": "steer@3b", "after": "steer@3, merged in PR 1809; same model, trial, cells, tiers and prompts",
               "reach": REACH, "query_iter": QUERY_ITER,
               "variants": {"target": "each raised concept at least base + t, each lowered one at most base - t, where t is the cell's "
                                      "best design-point score plus reach (log10(2) where no design point meets the cell); uniq, coh and "
                                      "drift constraints as in steer@3; starts at the best design point",
                            "center": "the same goals, starting at the zero design"},
               "claims": {"target": "on training prompts, target queries beat the design lookup in at least 8 of the 24 cells",
                          "center": "reported: cells where center queries meet the cell at all, and their score against lookup"}},
              open(PROTOCOL3B, "w", encoding="utf-8"), indent=1)
    print("wrote", PROTOCOL3B)


def requery(a):
    assert PROTOCOL3B.exists()
    _, _, best = best_points()
    lab = Lab()
    m = json.load(open(META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    z = lab.problem.evaluate(np.zeros((1, 24)))[0]
    for v in ("target", "center"):
        for ti, task in enumerate(TASKS):
            for tr, tier in enumerate(M.TIERS):
                lab.tag = f"{v}/{ti}/{tr}"
                sc = best[("lookup", ti, tr)][0]
                t = (np.log10(2) if sc is None else sc) + REACH
                g = [Goal(f"s_{c}", ">=", float(z[lab.names.index(c)] + t)) for c in task["up"]]
                g += [Goal(f"s_{c}", "<=", float(z[lab.names.index(c)] - t)) for c in task["down"]]
                g += [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", DRIFT)]
                n0 = len(lab.problem.history)
                res = inv.query(g, max_iter=QUERY_ITER, start="best" if v == "target" else "center")
                print(lab.tag, res.extra.get("stopped"), len(lab.problem.history) - n0, "evals", flush=True)


def report3b(a):
    names, base, best = best_points()
    held = {tuple(r["x"]): np.array(r["y"]) for r in map(json.loads, open(EVALS, encoding="utf-8")) if r["set"] == "held"}
    b0 = held[tuple([0.0] * 24)]
    rows = [json.loads(s) for s in open(EVALS, encoding="utf-8")]
    for v in ("gmoo", "target", "center"):
        q = [r for r in rows if r["set"] == "train" and r["tag"].startswith(v + "/")]
        hit = sum(meets(r["y"], M.TIERS[int(r["tag"].split("/")[2])]) for r in q)
        print(f"{v}: {len(q)} query evaluations, {hit} met their cell")

    def hs(m, t, tr):
        u = best[(m, t, tr)][1]
        return None if u is None or tuple(u) not in held else score(held[tuple(u)], TASKS[t], b0, names)
    print("\nwon / lost / tied against design lookup, training and held-out")
    for v in ("gmoo", "target", "center"):
        for nm, f in (("training", lambda m, t, tr: best[(m, t, tr)][0]), ("held-out", hs)):
            w = l = e = 0
            for t in range(len(TASKS)):
                for tr in range(3):
                    x, y = f(v, t, tr), f("lookup", t, tr)
                    x, y = -9 if x is None else x, -9 if y is None else y
                    w, l, e = w + (x > y + 1e-3), l + (y > x + 1e-3), e + (abs(x - y) <= 1e-3)
            print(f"  {v} {nm}: {w} / {l} / {e}")
    print("\ncenter queries meeting their cell:", sum(best[("center", t, tr)][0] is not None for t in range(len(TASKS)) for tr in range(3)), "of 24")
    print("held-out, target vs additive:", end=" ")
    w = l = 0
    for t in range(len(TASKS)):
        for tr in range(3):
            x, y = hs("target", t, tr), hs("additive", t, tr)
            x, y = -9 if x is None else x, -9 if y is None else y
            w, l = w + (x > y + 1e-3), l + (y > x + 1e-3)
    print(f"{w} / {l}")
    target_wins = sum((best[("target", t, tr)][0] or -9) > (best[("lookup", t, tr)][0] or -9) + 1e-3 for t in range(len(TASKS)) for tr in range(3))
    print(f"target claim: {target_wins} of 24 training cells beat lookup (needs 8)")


def _f(v):
    return "  -  " if v is None else f"{v:+.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "gmoo", "baselines", "heldout", "report", "protocol3b", "requery", "report3b"])
    ap.add_argument("--start", type=int, default=0, help="gmoo: resume at this task, after an interruption")
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "gmoo": gmoo, "baselines": baselines, "heldout": heldout, "report": report,
     "protocol3b": protocol3b, "requery": requery, "report3b": report3b}[a.cmd](a)
