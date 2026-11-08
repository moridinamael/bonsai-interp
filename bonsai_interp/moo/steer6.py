"""steer@6: hit a whole profile. Every panel concept at its own target at once, under the text-quality and drift constraints.

  python -m bonsai_interp.moo.steer6 protocol   # writes steer6_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer6 seed       # starts the evaluation cache from steer@5's (its design and band lines)
  python -m bonsai_interp.moo.steer6 query      # each cell: globalMOO, the band around every target tightened whenever met
  python -m bonsai_interp.moo.steer6 baselines  # the additive model (solve and correct) and differential evolution
  python -m bonsai_interp.moo.steer6 heldout
  python -m bonsai_interp.moo.steer6 report

steer@3 to @5 raised or lowered one or two concepts and left the rest free. A profile fixes all four panel concepts (school,
software, war, family) at once, some up, some down, some held where they are: seven objectives (four targets, distinct tokens,
coherence, drift) in one query. It reuses steer@5's trained model (7351), its 32 training prompts and its held-out prompts, so the
model never saw these goals. A cell's score is minus its worst miss: the largest absolute log10 error over the four concepts.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer5 as S5  # noqa: F401  (32 training prompts, steer@5's model)
from bonsai_interp.moo import steer_moo as M

S3.PROTOCOL = Path(__file__).parent / "steer6_protocol.json"
SOURCE = S3.EVALS  # steer@5's cache
S3.EVALS = S3.OUT / "steer6.evals.jsonl"
LEVELS = (-0.3, 0.0, 0.3, 0.6)  # target log10 change per concept
BUDGET, START_TOL, TIGHTEN, PRECISE = 24, 0.3, 0.75, float(np.log10(1.5))
DE_MAXITER = 3


def profiles():
    r = np.random.default_rng(61)
    out = []
    while len(out) < 8:
        t = [float(v) for v in r.choice(LEVELS, len(S3.CON))]
        if sum(v != 0 for v in t) >= 2 and t not in out:
            out.append(t)
    return out


PROFILES = profiles()


def miss(y, prof, base, names):
    return max(abs(y[names.index(c)] - base[names.index(c)] - t) for c, t in zip(S3.CON, prof))


def protocol(a):
    json.dump({"name": "steer@6", "after": "steer@5 (PR 1813): its model 7351, design, band lines, 32 training prompts, held-out prompts, "
                                          "bands, box, tiers and drift limit",
               "concepts": S3.CON, "profiles": PROFILES, "levels": LEVELS, "tiers": M.TIERS, "drift_max": S3.DRIFT, "budget": BUDGET,
               "start_tol": START_TOL, "tighten": TIGHTEN, "de_maxiter": DE_MAXITER,
               "score": "minus the largest absolute log10 change error over the four concepts, against the zero design on the same prompts",
               "methods": {"gmoo": "each cell as globalMOO queries on steer@5's trained model: every concept within tol of its target "
                                   "(an absolute band), uniq and coh at least the tier, drift at most drift_max; tol starts at the smaller "
                                   "of start_tol and tighten x the best design point's miss, and is multiplied by tighten whenever a query "
                                   "meets it, until the cell has used budget evaluations or a query ends without meeting it",
                           "additive": "steer@3's additive model from steer@5's band lines; each cell solved on the model by differential "
                                       "evolution on minus the score plus a steep penalty outside the cell, the answer evaluated, the model "
                                       "shifted by the residual and solved again, budget evaluations per cell",
                           "de": "differential evolution on the same loss at tier 2, de_maxiter generations of 24, per profile",
                           "lookup": "the best point of steer@5's design of experiments"},
               "claims": {"model": "over the 24 profile-tier cells on the held-out prompts, globalMOO beats the additive model in more "
                                   "cells than it loses; the same count against de and lookup is reported whichever way it falls",
                          "precision": "on the training prompts, globalMOO's best design meets the cell with every concept within "
                                       "log10(1.5) of its target in at least 12 of the 24 cells"}},
              open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("wrote", S3.PROTOCOL, PROFILES)


def seed(a):
    if not S3.EVALS.exists():
        shutil.copy(SOURCE, S3.EVALS)
    print("cache rows", sum(1 for _ in open(S3.EVALS, encoding="utf-8")))


def best_points():
    names = list(S3.T.CONCEPTS)
    rows = [r for r in map(json.loads, open(S3.EVALS, encoding="utf-8")) if r["set"] == "train"]
    base = next(np.array(r["y"]) for r in rows if not any(r["x"]))
    doe = [r for r in rows if r["tag"] == "doe"]
    best = {}
    for pi, prof in enumerate(PROFILES):
        pools = {"lookup": doe, "gmoo": doe + [r for r in rows if r["tag"].startswith(f"profile/{pi}/")],
                 "additive": [r for r in rows if r["tag"] == "band" or r["tag"].startswith(f"padd/{pi}/")],
                 "de": [r for r in rows if r["tag"] == f"pde/{pi}"]}
        for m, pool in pools.items():
            for tr, tier in enumerate(M.TIERS):
                ok = [(-miss(r["y"], prof, base, names), r["x"]) for r in pool if S3.meets(r["y"], tier)]
                best[(m, pi, tr)] = max(ok, key=lambda v: v[0]) if ok else (None, None)
    return names, base, best


def query(a):
    _, _, best = best_points()
    lab = S3.Lab()
    m = json.load(open(S3.META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    z = lab.problem.evaluate(np.zeros((1, 24)))[0]
    for pi, prof in enumerate(PROFILES):
        if pi < a.start:
            continue
        for tr, tier in enumerate(M.TIERS):
            lab.tag = f"profile/{pi}/{tr}"
            sc = best[("lookup", pi, tr)][0]
            tol = START_TOL if sc is None else min(START_TOL, TIGHTEN * -sc)
            used = rounds = 0
            while used < BUDGET:
                g = [Goal(f"s_{c}", "=", float(z[lab.names.index(c)] + t), tol=float(tol)) for c, t in zip(S3.CON, prof)]
                g += [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", S3.DRIFT)]
                n0 = len(lab.problem.history)
                res = inv.query(g, max_iter=BUDGET - used)
                used += len(lab.problem.history) - n0
                rounds += 1
                if res.extra.get("stopped") not in ("satisfied", "satisfied_local"):
                    break
                tol *= TIGHTEN
            print(lab.tag, rounds, "rounds", used, "evals, last tol", round(float(tol), 3), flush=True)


def _loss(y, prof, tier, base, names):
    out = max(0, tier["uniq"] - y[24]) / .05 + max(0, tier["coh"] - y[25]) / .2 + max(0, y[26] - S3.DRIFT) / .05
    return miss(y, prof, base, names) + 50 * out


def _de(f, prof, tier, base, names, seed, maxiter, popsize):
    from scipy import optimize
    r = optimize.differential_evolution(lambda X: np.array([_loss(y, prof, tier, base, names) for y in f(X.T)]), [(-1, 1)] * 24,
                                        maxiter=maxiter, popsize=popsize, vectorized=True, seed=seed, polish=False, updating="deferred", tol=0)
    return r.x


def baselines(a):
    lab = S3.Lab()
    names = lab.names
    lab.tag = "band"
    add = S3.Additive(lab)
    for pi, prof in enumerate(PROFILES):
        if pi < a.start:
            continue
        for tr, tier in enumerate(M.TIERS):
            lab.tag = f"padd/{pi}/{tr}"
            add.shift[:] = 0
            for it in range(BUDGET):
                u = _de(add, prof, tier, add.z, names, 1000 * pi + 50 * tr + it, 60, 10)
                y = lab.problem.evaluate(u[None])[0]
                add.shift += y - add(u)[0]
        lab.tag = f"pde/{pi}"
        _de(lab.problem.evaluate, prof, M.TIERS[1], add.z, names, pi, DE_MAXITER, 1)
        print("profile", pi, "done", round(time_since(lab)), "s", flush=True)


def time_since(lab):
    import time
    return time.time() - lab.t0


METHODS = ("gmoo", "additive", "de", "lookup")


def heldout(a):
    _, _, best = best_points()
    lab = S3.Lab()
    lab.prompts, lab.tag = M.HELD, "heldout"
    X = sorted({tuple(v[1]) for v in best.values() if v[1] is not None})
    print(len(X), "designs on held-out prompts", flush=True)
    lab.problem.evaluate(np.array(X))


def report(a):
    names, base, best = best_points()
    rows = [json.loads(s) for s in open(S3.EVALS, encoding="utf-8")]
    held = {tuple(r["x"]): np.array(r["y"]) for r in rows if r["set"] == "held"}
    b0 = held[tuple([0.0] * 24)]
    tags = [r["tag"].split("/")[0] for r in rows if r["set"] == "train" and r["tag"].split("/")[0] in ("profile", "padd", "pde")]
    print("training evaluations by tag", {t: tags.count(t) for t in sorted(set(tags))})

    def hs(m, p, tr):
        u = best[(m, p, tr)][1]
        return None if u is None else -miss(held[tuple(u)], PROFILES[p], b0, names)
    print("\nworst miss by cell (log10), training / held-out")
    for p, prof in enumerate(PROFILES):
        for tr in range(3):
            print(f"  profile {p} {prof} tier {tr + 1}: " + "  ".join(
                f"{m} {S3._f(None if best[(m, p, tr)][0] is None else -best[(m, p, tr)][0])}/{S3._f(None if hs(m, p, tr) is None else -hs(m, p, tr))}"
                for m in METHODS))
    print("\nglobalMOO against each method, cells won / lost / tied (a smaller worst miss wins)")
    for nm, f in (("training", lambda m, p, tr: best[(m, p, tr)][0]), ("held-out", hs)):
        for m in METHODS[1:]:
            w = l = e = 0
            for p in range(len(PROFILES)):
                for tr in range(3):
                    x, y = f("gmoo", p, tr), f(m, p, tr)
                    x, y = -9 if x is None else x, -9 if y is None else y
                    w, l, e = w + (x > y + 1e-3), l + (y > x + 1e-3), e + (abs(x - y) <= 1e-3)
            print(f"  {nm} vs {m}: {w} / {l} / {e}")
    for nm, f in (("training", lambda m, p, tr: best[(m, p, tr)][0]), ("held-out", hs)):
        print(f"cells within log10(1.5) on every concept, {nm}:",
              {m: sum((f(m, p, tr) or -9) >= -PRECISE for p in range(8) for tr in range(3)) for m in METHODS})
    print("mean held-out worst miss:", {m: round(float(np.mean([-(hs(m, p, tr) or -2) for p in range(8) for tr in range(3)])), 3) for m in METHODS})
    prec = sum((best[("gmoo", p, tr)][0] or -9) >= -PRECISE for p in range(8) for tr in range(3))
    print(f"precision claim: {prec} of 24 training cells (needs 12)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "seed", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query or baselines: resume at this profile")
    a = ap.parse_args()
    {"protocol": protocol, "seed": seed, "query": query, "baselines": baselines, "heldout": heldout, "report": report}[a.cmd](a)
