"""One globalMOO model for a family of 99 poles: what each pole's firing needs from its context, as a front.

  python -m bonsai_interp.moo.family train --layer 15       # the design of experiments and one model
  python -m bonsai_interp.moo.family fronts --layer 15      # per pole, the fewest factors that keep it at each level
  python -m bonsai_interp.moo.family greedy --layer 15      # the same fronts by greedy elimination, no model

Inputs: 23 factors, each the share of its tokens kept (context.py; the read token stays). Outputs: the 99 poles (share of
their activation at g = 1, each read on its own snippets) and "kept", the mean of the inputs. A front is, per level in LEVELS,
the least context that keeps the pole at that level, so a pole's fronts say which parts of its context it can't do without.
The front is traced by tightening a cap: pole >= level, kept <= cap, for the caps in CAPS while they can be met.

One evaluation of all 99 poles reads every snippet. A query about one pole reads that pole's snippets only; its other
outputs are filled from the nearest design point, and they are unconstrained in the query.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal, InverseProblem, Var
from bonsai_interp.moo import context as X

OUT = ct.ROOT / "data" / "moo"
PER_POLE, N_POLES = 4, 99
LEVELS = (0.8, 0.5, 0.3)
CAPS = (0.7, 0.5, 0.35, 0.25, 0.15)
FREE = X.FACTORS[1:]


class Family:
    def __init__(self, L, seed=0):
        self.t0, self.L = time.time(), L
        model, tok, self.rec = ct._gpu_load()
        self.X, self.base, self.poles = X.setup(model, self.rec, L, N_POLES, PER_POLE, seed)
        self.names = [f"N{n}{s}" for n, s in self.poles] + ["kept"]
        self.reads, self.focus, self.design = 0, None, None
        OUT.mkdir(exist_ok=True)
        self.log = OUT / f"family_L{L}.evals.jsonl"
        self.cache = {}
        if self.log.exists():
            for line in open(self.log, encoding="utf-8"):
                r = json.loads(line)
                self.cache[tuple(r["x"])] = np.array(r["y"])
        self.problem = InverseProblem(f"family-L{L}", [Var(f, 0.0, 1.0, "float") for f in FREE], self.names, self.evaluate)

    def _g(self, x):
        return np.r_[1.0, x]

    def _read(self, x):
        """All outputs at x, resumable: a full evaluation is kept in the log."""
        k = tuple(round(float(v), 3) for v in x)
        if k in self.cache:
            return self.cache[k]
        for attempt in range(3):
            try:
                y = X.outputs(self.X.read(self._g(k)), self.base, PER_POLE)
                break
            except OSError as e:  # a transient file error in the GPU queue must not lose a training run
                print(f"retrying after {e}", flush=True)
                time.sleep(2)
        self.reads += len(self.base)
        y = np.r_[y, np.mean(k)]
        self.cache[k] = y
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"x": list(k), "y": [round(float(v), 4) for v in y]}) + "\n")
        return y

    def _read_focus(self, x):
        """The focus pole's output at x, and the other poles from the nearest full evaluation."""
        k = tuple(round(float(v), 3) for v in x)
        i = self.focus
        idx = np.arange(i * PER_POLE, (i + 1) * PER_POLE)
        a = self.X.read(self._g(k), idx)
        self.reads += PER_POLE
        ks = np.array(list(self.cache))
        y = self.cache[tuple(ks[np.argmin(np.abs(ks - np.array(k)).sum(1))])].copy()
        y[i] = float((a / self.base[idx]).mean())
        y[-1] = float(np.mean(k))
        return y

    def evaluate(self, Xs):
        out = []
        for n, x in enumerate(Xs):
            out.append(self._read(x) if self.focus is None else self._read_focus(x))
            if self.focus is None and n % 25 == 0:
                print(f"{n + 1} of {len(Xs)} ({time.time() - self.t0:.0f} s)", flush=True)
        return np.array(out)


def _meta(L):
    return OUT / f"family_L{L}.model.json"


def train(a):
    fam = Family(a.layer)
    print(f"L{a.layer}: {len(fam.poles)} poles, {len(fam.base)} snippets", flush=True)
    t = time.time()
    fam.X.read(np.ones(len(X.FACTORS)))
    print(f"one evaluation {time.time() - t:.1f} s", flush=True)
    inv = S.GmooInverse(fam.problem, f"mono-family-L{a.layer}")
    fam.problem.scales = None
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals,
               "doe_seconds": inv.doe_seconds, "poles": fam.names[:-1]}, open(_meta(a.layer), "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def fronts(a):
    fam = Family(a.layer)
    m = json.load(open(_meta(a.layer)))
    assert m["poles"] == fam.names[:-1], "the pole set differs from the trained one"
    inv = S.GmooInverse.attach(fam.problem, m["model_id"], m["trial_id"])
    fam.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    path = OUT / f"family_L{a.layer}.fronts.jsonl"
    done = {json.loads(s)["pole"] for s in open(path, encoding="utf-8")} if path.exists() else set()
    for i, name in enumerate(fam.names[:-1]):
        if name in done or (a.max and i >= a.max):
            continue
        fam.focus, row, r0, t0 = i, {"pole": name, "levels": {}}, fam.reads, time.time()
        for lv in LEVELS:
            best, evals = None, 0
            for cap in CAPS:
                res = inv.query([Goal(name, ">=", lv), Goal("kept", "<=", cap)], max_iter=a.iter)
                evals += res.n_evals
                y = res.y[fam.names.index(name)]
                if not (y >= lv and res.y[-1] <= cap):
                    break
                best = {"retained": round(float(y), 3), "kept": round(float(res.x.sum()), 2), "x": [round(float(v), 3) for v in res.x]}
            row["levels"][str(lv)] = {"met": best is not None, **(best or {}), "evals": evals, "stopped": res.extra.get("stopped")}
        row["reads"], row["seconds"] = fam.reads - r0, round(time.time() - t0, 1)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(name, {k: (v["met"], v.get("kept"), v["evals"]) for k, v in row["levels"].items()}, flush=True)


def _greedy(fam, i, level):
    """Backward elimination for pole i: drop the factor whose loss is least while the pole stays at the level."""
    idx = np.arange(i * PER_POLE, (i + 1) * PER_POLE)
    x, evals = np.ones(len(FREE), int), 0

    def val(z):
        return float((fam.X.read(fam._g(z), idx) / fam.base[idx]).mean())
    while x.sum():
        best, bv = None, -1e9
        for k in np.nonzero(x)[0]:
            z = x.copy()
            z[k] = 0
            v = val(z)
            evals += 1
            if v > bv:
                best, bv = k, v
        if bv < level:
            break
        x[best] = 0
    return x, val(x), evals + 1


def greedy(a):
    fam = Family(a.layer)
    path = OUT / f"family_L{a.layer}.greedy.jsonl"
    done = {json.loads(s)["pole"] for s in open(path, encoding="utf-8")} if path.exists() else set()
    for i, name in enumerate(fam.names[:-1]):
        if name in done or (a.max and i >= a.max):
            continue
        row, t0 = {"pole": name, "levels": {}}, time.time()
        for lv in LEVELS:
            x, v, ev = _greedy(fam, i, lv)
            row["levels"][str(lv)] = {"met": bool(v >= lv), "retained": round(v, 3), "kept": int(x.sum()), "x": x.tolist(), "evals": ev}
        row["seconds"] = round(time.time() - t0, 1)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(name, {k: (v["met"], v["kept"], v["evals"]) for k, v in row["levels"].items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "fronts", "greedy"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--max", type=int)
    ap.add_argument("--iter", type=int, default=40)
    a = ap.parse_args()
    {"train": train, "fronts": fronts, "greedy": greedy}[a.cmd](a)
