"""One globalMOO model of the steering panel: 24 knobs, 26 outputs, goals that pull against each other.

  python -m bonsai_interp.moo.steer_moo protocol   # writes steer_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer_moo train      # the design of experiments and one model
  python -m bonsai_interp.moo.steer_moo gmoo       # every task at every tier through the model
  python -m bonsai_interp.moo.steer_moo baselines  # design lookup, a gain grid and differential evolution
  python -m bonsai_interp.moo.steer_moo heldout    # each method's best designs on the held-out prompts, with placebo bundles
  python -m bonsai_interp.moo.steer_moo report

Inputs: one gain per concept bundle, in +-BOX (steer.py). Outputs: log10 soft rate of each concept, the distinct-token ratio
and the coherence of the text under the clean model. Steering pulls against both of those, and a task with several targets
pulls the knobs against each other, so a task has a front: the most steering at each tier of text quality.

A task's score is its smallest log10 fold change against the zero-gain text: up for a raised concept, down for a lowered one.
Every method is scored the same way, on the best evaluated point that meets the tier.
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

OUT = ct.ROOT / "data" / "moo"
PROTOCOL = Path(__file__).parent / "steer_protocol.json"
EVALS = OUT / "steer2.evals.jsonl"
META = OUT / "steer2.model.json"
BOX, NEW, BATCH, FLOOR = 0.04, 24, 2, 1e-5
TRAIN = ["The thing I remember most about that day is", "After lunch we walked down to the", "The most important thing to know about this is",
         "He looked at the results and said", "When I got home that evening,", "She opened the letter and", "It was the first time anyone had",
         "What surprised me most was", "In the years that followed,", "According to the people who were there,", "We spent most of the morning",
         "There are a few reasons why", "I have always believed that", "The next step is to", "Nobody expected that", "Most people don't realize that"]
HELD = ["On the way back I stopped to", "My neighbour told me about the", "For a long time there was nothing but", "The letter said only that",
        "As soon as the door opened,", "Her first thought was that", "By the end of the week we had", "One of the strangest things about it was",
        "They asked me what I thought about the", "If you look closely you will see", "Later that night he wrote that", "The old man smiled and told us",
        "Everyone in the room agreed that", "I never did find out why", "Sooner or later you have to", "That summer we finally decided to"]
TIERS = [{"uniq": 0.68, "coh": -1.40}, {"uniq": 0.58, "coh": -1.70}, {"uniq": 0.45, "coh": -2.20}]
LEVELS = (8, 15, 25)  # query target per tier, as a fold change against zero gain; a lowered concept aims at a third of it
QUERY_ITER, DE_MAXITER, GRID = 12, 3, tuple(round(.01 * i, 2) for i in range(1, 13))  # the grid may leave the box


def protocol(a):
    from bonsai_interp import explore
    from bonsai_interp.compose import round3 as R3
    names = list(T.CONCEPTS)
    r = np.random.default_rng(17)
    draw = lambda n: [names[i] for i in r.permutation(24)[:n]]
    tasks = [{"up": draw(1), "down": []} for _ in range(2)] + [{"up": draw(2), "down": []} for _ in range(4)]
    tasks += [{"up": draw(3), "down": []} for _ in range(3)]
    for _ in range(3):
        d = draw(2)
        tasks.append({"up": d[:1], "down": d[1:]})
    B, _ = T.bundles(explore.tokenizer(), R3._assigned())
    sha = hashlib.sha256(json.dumps({c: sorted(B[c]) for c in names}).encode()).hexdigest()
    json.dump({"name": "steer@2", "supersedes": "steer@1, stopped after task 0: only the zero-gain design of 687 met any tier, because random gains on all 24 knobs at +-0.12 collapse the text; a pilot of six scalings on training prompts chose this box", "box": BOX, "new_tokens": NEW, "tasks": tasks, "tiers": TIERS, "levels": LEVELS, "query_iter": QUERY_ITER,
               "de_maxiter": DE_MAXITER, "grid": GRID, "train_prompts": TRAIN, "heldout_prompts": HELD, "bundles_sha256": sha,
               "claims": {"composition": "at tier 2, on the held-out prompts, at least 9 of 12 tasks reach a score of log10(3) with labelled bundles while the same gains on placebo bundles stay below it",
                          "model": "over the 36 task-tier cells, globalMOO beats each baseline in more cells than it loses, on training and on held-out prompts; reported whichever way it falls"}},
              open(PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("wrote", PROTOCOL, "bundles", sha[:12])


class Lab:
    def __init__(self):
        from bonsai_interp.compose import round3 as R3
        self.t0 = time.time()
        self.p = json.load(open(PROTOCOL, encoding="utf-8"))
        model, self.tok, rec = ct._gpu_load(lm_head=True)
        asg = R3._assigned()
        B, toks = T.bundles(self.tok, asg)
        self.names = list(T.CONCEPTS)
        sha = hashlib.sha256(json.dumps({c: sorted(B[c]) for c in self.names}).encode()).hexdigest()
        assert sha == self.p["bundles_sha256"], "the bundles differ from the protocol's"
        pm = np.stack([np.stack(ct._pole_max(L)) for L in range(36)]).astype(np.float32)
        rng = np.random.default_rng(5)
        by = {}
        for q in sorted(asg):
            by.setdefault(q[0], []).append(q)
        placebo = lambda b: [by[L][rng.integers(len(by[L]))] for L, _, _ in b]
        self.panel = T.Panel(model, rec, pm, [B[c] for c in self.names] + [placebo(B[c]) for c in self.names])
        self.ids = [torch.tensor(toks[c]) for c in self.names]
        self.outputs = [f"s_{c}" for c in self.names] + ["uniq", "coh"]
        self.cache, self.tag, self.prompts, self.place = {}, "", TRAIN, 0
        OUT.mkdir(exist_ok=True)
        if EVALS.exists():
            for line in open(EVALS, encoding="utf-8"):
                r = json.loads(line)
                self.cache[(r["set"], tuple(r["x"]))] = np.array(r["y"])
        self.problem = InverseProblem("steer", [Var(c, -BOX, BOX, "float") for c in self.names], self.outputs, self.evaluate)

    def _read(self, X):
        """Outputs for a batch of designs on the current prompts; placebo bundles when self.place."""
        D = len(X)
        th = np.zeros((D, 48))
        th[:, self.place:self.place + 24] = X
        e, g, lp = T.generate(self.panel, self.tok, self.prompts, th, NEW)
        P = g.shape[1]
        soft = torch.stack([lp[:, :, i].exp().sum(-1).mean(1) for i in self.ids], 1).view(D, P, -1).mean(1).numpy()
        gc = g.cpu()
        uniq = np.array([[len(set(x.tolist())) / g.shape[2] for x in row] for row in gc]).mean(1)
        coh = T.clean_logp(self.panel, e, g).mean(1).cpu().numpy()
        return np.c_[np.log10(soft + FLOOR), uniq, coh]

    def evaluate(self, X):
        X = np.round(np.asarray(X, float), 4)
        out = np.zeros((len(X), 26))
        s = "train" if self.prompts is TRAIN else "held"
        s += "-placebo" if self.place else ""
        todo = []
        for i, x in enumerate(X):
            k = (s, tuple(x.tolist()))
            if k in self.cache:
                out[i] = self.cache[k]
            else:
                todo.append(i)
        with open(EVALS, "a", encoding="utf-8") as f:
            for j in range(0, len(todo), BATCH):
                ix = todo[j:j + BATCH]
                y = self._read(X[ix])
                out[ix] = y
                for i, v in zip(ix, y):
                    self.cache[(s, tuple(X[i].tolist()))] = v
                    f.write(json.dumps({"set": s, "tag": self.tag, "x": X[i].tolist(), "y": [round(float(w), 4) for w in v]}) + "\n")
                f.flush()
                if j % 100 == 0 and len(todo) > 8:
                    print(f"{j} of {len(todo)} ({time.time() - self.t0:.0f} s)", flush=True)
        return out

    def base(self):
        return self.problem.evaluate(np.zeros((1, 24)))[0]


def goals_for(lab, task, tier, fold, base):
    g = [Goal(f"s_{c}", ">=", float(base[lab.names.index(c)] + np.log10(fold))) for c in task["up"]]
    g += [Goal(f"s_{c}", "<=", float(base[lab.names.index(c)] - np.log10(3))) for c in task["down"]]
    return g + [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"])]


def train(a):
    lab = Lab()
    lab.tag = "doe"
    t = time.time()
    y0 = lab.base()
    print("base", np.round(y0[-2:], 3), f"{time.time() - t:.1f} s", flush=True)
    inv = S.GmooInverse(lab.problem, "mono-steer2")
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals, "doe_seconds": inv.doe_seconds}, open(META, "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def gmoo(a):
    lab = Lab()
    m = json.load(open(META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    base = lab.base()
    for ti, task in enumerate(lab.p["tasks"]):
        for tr, tier in enumerate(TIERS):
            lab.tag = f"gmoo/{ti}/{tr}"
            n0 = len(lab.problem.history)
            res = inv.query(goals_for(lab, task, tier, LEVELS[tr], base), max_iter=QUERY_ITER)
            print(lab.tag, res.satisfied, res.extra.get("stopped"), len(lab.problem.history) - n0, "evals", flush=True)


def baselines(a):
    lab = Lab()
    base = lab.base()
    lab.problem.scales = np.maximum(np.array([json.loads(s)["y"] for s in open(EVALS, encoding="utf-8")]).std(0), 1e-3)
    for ti, task in enumerate(lab.p["tasks"]):
        lab.tag = f"grid/{ti}"
        X = np.zeros((len(GRID), 24))
        for i, c in enumerate(GRID):
            for n in task["up"]:
                X[i, lab.names.index(n)] = c
            for n in task["down"]:
                X[i, lab.names.index(n)] = -c
        lab.problem.evaluate(X)
        lab.tag = f"de/{ti}"
        S.differential_evolution(lab.problem, goals_for(lab, task, TIERS[1], LEVELS[1], base), maxiter=DE_MAXITER, popsize=1, seed=ti)
        print("task", ti, "done", round(time.time() - lab.t0), "s", flush=True)


def score(y, task, base, names):
    """Smallest log10 fold change over the task's concepts."""
    up = [y[names.index(c)] - base[names.index(c)] for c in task["up"]]
    dn = [base[names.index(c)] - y[names.index(c)] for c in task["down"]]
    return min(up + dn)


def best_points():
    """{(method, task, tier): (score, x)} over each method's training evaluations."""
    p = json.load(open(PROTOCOL, encoding="utf-8"))
    names, base, rows = list(T.CONCEPTS), None, []
    for s in open(EVALS, encoding="utf-8"):
        r = json.loads(s)
        if r["set"] == "train":
            rows.append(r)
            if not any(r["x"]):
                base = np.array(r["y"])
    doe = [r for r in rows if r["tag"] == "doe"]
    best = {}
    for ti, task in enumerate(p["tasks"]):
        pools = {"lookup": doe, "gmoo": doe + [r for r in rows if r["tag"].startswith(f"gmoo/{ti}/")],
                 "grid": [r for r in rows if r["tag"] == f"grid/{ti}"], "de": [r for r in rows if r["tag"] == f"de/{ti}"]}
        for m, pool in pools.items():
            for tr, tier in enumerate(TIERS):
                ok = [(score(r["y"], task, base, names), r["x"]) for r in pool if r["y"][24] >= tier["uniq"] and r["y"][25] >= tier["coh"]]
                best[(m, ti, tr)] = max(ok, key=lambda v: v[0]) if ok else (None, None)
    return p, names, base, best


def heldout(a):
    p, names, _, best = best_points()
    lab = Lab()
    lab.prompts, lab.tag = HELD, "heldout"
    lab.problem.evaluate(np.zeros((1, 24)))
    X = sorted({tuple(v[1]) for v in best.values() if v[1] is not None})
    print(len(X), "designs on held-out prompts", flush=True)
    lab.problem.evaluate(np.array(X))
    lab.place = 24
    lab.problem.evaluate(np.array(X))


def report(a):
    p, names, base, best = best_points()
    held = {}
    for s in open(EVALS, encoding="utf-8"):
        r = json.loads(s)
        if r["set"] != "train":
            held[(r["set"], tuple(r["x"]))] = np.array(r["y"])
    b0 = held.get(("held", tuple([0.0] * 24)))
    methods = ("gmoo", "de", "grid", "lookup")
    n = len(p["tasks"])
    print("evaluations", {m: sum(1 for s in open(EVALS, encoding="utf-8") if json.loads(s)["tag"].startswith(m)) for m in ("doe", "gmoo", "de", "grid")})
    print("\ntraining score by method: mean of met cells, cells met of", n)
    for tr in range(3):
        print(f"  tier {tr + 1}:", {m: (round(float(np.mean([best[(m, t, tr)][0] for t in range(n) if best[(m, t, tr)][0] is not None] or [0])), 3),
                                    sum(best[(m, t, tr)][0] is not None for t in range(n))) for m in methods})
    def hs(m, t, tr, st="held"):
        x = best[(m, t, tr)][1]
        if x is None:
            return None
        return score(held[(st, tuple(x))], p["tasks"][t], b0, names), held[(st, tuple(x))]
    print("\ngmoo against each baseline, cells won / lost / tied (a cell with no feasible point loses to one with)")
    for name, f in (("training", lambda m, t, tr: best[(m, t, tr)][0]), ("held-out", lambda m, t, tr: (hs(m, t, tr) or [None])[0])):
        for m in methods[1:]:
            w = l = e = 0
            for t in range(n):
                for tr in range(3):
                    x, y = f("gmoo", t, tr), f(m, t, tr)
                    x, y = -9 if x is None else x, -9 if y is None else y
                    w, l, e = w + (x > y + 1e-3), l + (y > x + 1e-3), e + (abs(x - y) <= 1e-3)
            print(f"  {name} vs {m}: {w} / {l} / {e}")
    print("\ncomposition at tier 2, held-out: score labelled, score placebo, uniq, coh")
    passed = 0
    for t, task in enumerate(p["tasks"]):
        v = hs("gmoo", t, 1)
        if v is None:
            print(f"  task {t} {task}: no design")
            continue
        q = hs("gmoo", t, 1, "held-placebo")
        ok = v[0] >= np.log10(3) and q[0] < np.log10(3)
        passed += ok
        print(f"  task {t} {task}: {v[0]:+.2f} {q[0]:+.2f} uniq {v[1][24]:.2f} coh {v[1][25]:.2f} {'pass' if ok else 'miss'}")
    print(f"composition: {passed} of {n} (claim needs 9)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "gmoo", "baselines", "heldout", "report"])
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "gmoo": gmoo, "baselines": baselines, "heldout": heldout, "report": report}[a.cmd](a)
