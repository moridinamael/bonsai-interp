"""steer@13: targets the design cannot answer. One comparison: globalMOO's inverse query against the best design point.

  python -m bonsai_interp.moo.steer13 seed | protocol | query | heldout | report
  bash scripts/moo_chain.sh steer13 query heldout report

In steer@12 a design point already lay within 0.02 to 0.08 of every target, because the targets were midpoints of design points.
The query beat the lookup in 7 of 16 profiles and tied it in 9. Here every target lies outside what the design reached:
  - no new model: steer@12's trained model, its 699 design points (copied by `seed`), prompts, measure, tier and drift limit;
  - the 597 design points that meet tier 2 are taken as points in the four panel concepts' log10 changes. 16 vertices of their
    convex hull are drawn (seed 71), and each target is that vertex's change vector times OUT (1.25), rounded to 2 places. The
    zero design is inside the hull, so every target is outside it, and the lookup's miss is the distance back to the design;
  - globalMOO's band starts at 0.75 of the lookup's miss on that profile, so the first query already asks for a design better
    than any design point, and tightens by 0.75 when met. 24 evaluations per profile, as in steer@12.
The lookup spends no evaluations. No additive or DE baseline is run: this round asks one thing.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer6 as S6
from bonsai_interp.moo import steer12 as S12  # noqa: F401  (steer@12's box, prompts, measure, drift limit, model)
from bonsai_interp.moo import steer_moo as M

SOURCE = S3.EVALS  # steer@12's evaluations
S3.PROTOCOL = Path(__file__).parent / "steer13_protocol.json"
S3.EVALS = S3.OUT / "steer13.evals.jsonl"  # S3.META stays steer@12's model
OUT, N, SEED, START_FACTOR = 1.25, 16, 71, 0.75
TIER = M.TIERS[0]


def _rows():
    return [json.loads(s) for s in open(S3.EVALS, encoding="utf-8")]


def _design():
    names = list(S3.T.CONCEPTS)
    ci = [names.index(c) for c in S3.CON]
    rows = [r for r in _rows() if r["set"] == "train"]
    base = np.array(next(r["y"] for r in rows if not any(r["x"])))
    doe = [r for r in rows if r["tag"] == "doe" and S3.meets(r["y"], TIER)]
    return rows, base, ci, doe, np.array([np.array(r["y"])[ci] - base[ci] for r in doe])


def targets():
    from scipy.spatial import ConvexHull
    _, _, _, _, P = _design()
    v = np.random.default_rng(SEED).choice(np.sort(ConvexHull(P).vertices), N, replace=False)
    return [[round(float(x), 2) for x in OUT * P[i]] for i in v], len(P)


def seed(a):
    if not S3.EVALS.exists():
        keep = [s for s in open(SOURCE, encoding="utf-8") if json.loads(s)["set"] == "train" and json.loads(s)["tag"] == "doe"]
        open(S3.EVALS, "w", encoding="utf-8", newline="\n").writelines(keep)
    print("design rows", sum(1 for _ in open(S3.EVALS, encoding="utf-8")))


def _miss(d, t):
    return float(np.abs(np.asarray(d) - np.asarray(t)).max())


def best():
    """Per profile: (miss, x) of the best tier-2 point for the lookup (design only) and globalMOO (design plus its queries)."""
    rows, base, ci, doe, _ = _design()
    T, _ = targets()
    out = {}
    for pi, t in enumerate(T):
        pools = {"lookup": doe, "gmoo": doe + [r for r in rows if r["tag"] == f"profile/{pi}/0" and S3.meets(r["y"], TIER)]}
        for m, pool in pools.items():
            out[(m, pi)] = min(((_miss(np.array(r["y"])[ci] - base[ci], t), r["x"]) for r in pool), key=lambda v: v[0])
    return T, out


def protocol(a):
    T, b = best()
    p = json.load(open(Path(__file__).parent / "steer12_protocol.json", encoding="utf-8"))
    look = [round(b[("lookup", pi)][0], 3) for pi in range(N)]
    q = {"name": "steer@13",
         "after": "steer@12 (PR 1823): its trained model and 699-point design, prompts, measure, tier, drift limit, box and budget",
         "bundles_sha256": p["bundles_sha256"], "concepts": p["concepts"], "tiers": p["tiers"], "drift_max": p["drift_max"],
         "measure": p["measure"], "heldout_prompts": p["heldout_prompts"], "budget": S6.BUDGET, "tighten": S6.TIGHTEN,
         "out": OUT, "seed": SEED, "start_factor": START_FACTOR,
         "targets_rule": "the tier-2 design points as points in the four concepts' log10 changes; 16 vertices of their convex hull drawn "
                         "without replacement (numpy default_rng(seed), from the sorted vertex indices); target = out x the vertex, "
                         "rounded to 2 places",
         "targets": T, "lookup_miss": look,
         "methods": {"lookup": "the tier-2 design point with the smallest worst miss; no evaluations",
                     "gmoo": "queries on steer@12's model: every concept within tol of its target, uniq and coh at least the tier, drift at "
                             "most drift_max; tol starts at start_factor x the lookup's miss and is multiplied by tighten whenever a query "
                             "meets it; asked again at the same tol otherwise, until budget evaluations are used or a query returns no "
                             "new evaluation. Its answer is its best tier-2 evaluation or the lookup's point, whichever misses less"},
         "score": "the worst miss: the largest absolute log10 error over the four concepts, against the zero design on the same prompts",
         "primary": "on the training prompts, the number of the 16 profiles where globalMOO's worst miss is smaller than the lookup's by "
                    "more than 0.001",
         "prediction": {"count": 11, "of": N, "held_if_at_least": 9,
                        "arithmetic": "In steer@12 the query beat the lookup in 6 of the 6 profiles where the lookup missed by 0.07 or "
                                      "more, and in 1 of the 10 where it missed by 0.04 or less: with room, it won every time. Here the "
                                      "lookup's miss on every one of the 16 targets is listed in lookup_miss, so by that rate all 16 "
                                      "have room. Two things cut it: the targets ask for changes 25% beyond anything the model was "
                                      "trained on, and the hull's vertices sit where the drift limit binds. About 6% of steer@12's "
                                      "2,234 later tier-2 evaluations (18 of a sample of 320) fell outside the hull without being aimed "
                                      "there, so outside is reachable, but none of them comes closer to a 1.25x target than the lookup "
                                      "does. I discount the 6-of-6 rate to 0.7: 16 x 0.7 = 11.",
                        "held": "9 or more of 16 (more than half). A tie counts against. globalMOO cannot lose a profile, since its "
                                "answer falls back to the lookup's point"},
         "reported": "the same count on the 64 held-out prompts (smaller / larger / equal), the mean worst miss of both on both prompt "
                     "sets, how many queries met their band, and how many query evaluations met tier 2"}
    json.dump(q, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@13; lookup miss", look)


def query(a):
    T, b = best()
    lab = S3.Lab()
    m = json.load(open(S3.META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    z = lab.problem.evaluate(np.zeros((1, 24)))[0]
    for pi, prof in enumerate(T):
        if pi < a.start:
            continue
        lab.tag = f"profile/{pi}/0"
        tol, used, rounds, met = START_FACTOR * b[("lookup", pi)][0], 0, 0, 0
        while used < S6.BUDGET:
            g = [Goal(f"s_{c}", "=", float(z[lab.names.index(c)] + t), tol=float(tol)) for c, t in zip(S3.CON, prof)]
            g += [Goal("uniq", ">=", TIER["uniq"]), Goal("coh", ">=", TIER["coh"]), Goal("drift", "<=", S3.DRIFT)]
            n0 = len(lab.problem.history)
            res = inv.query(g, max_iter=S6.BUDGET - used)
            new = len(lab.problem.history) - n0
            used += new
            rounds += 1
            if res.extra.get("stopped") in ("satisfied", "satisfied_local"):
                met += 1
                tol *= S6.TIGHTEN
            elif new == 0:
                break
        print(lab.tag, rounds, "queries", met, "met", used, "evals, last tol", round(float(tol), 3), flush=True)


def heldout(a):
    _, b = best()
    lab = S3.Lab()
    lab.prompts, lab.tag = M.HELD, "heldout"
    X = sorted({tuple(v[1]) for v in b.values()} | {tuple([0.0] * 24)})
    print(len(X), "designs on held-out prompts", flush=True)
    lab.problem.evaluate(np.array(X))


def report(a):
    T, b = best()
    rows, base, ci, _, P = _design()
    held = {tuple(r["x"]): np.array(r["y"]) for r in _rows() if r["set"] == "held"}
    h0 = held[tuple([0.0] * 24)]

    def hm(m, pi):
        return _miss(held[tuple(b[(m, pi)][1])][ci] - h0[ci], T[pi])
    print(f"{len(P)} design points meet tier 2; 16 targets at {OUT} x a hull vertex")
    print("worst miss (log10), training / held-out, and the query evaluations that meet tier 2")
    w = l = hw = hl = 0
    for pi, t in enumerate(T):
        q = [r for r in rows if r["tag"] == f"profile/{pi}/0"]
        ok = sum(bool(S3.meets(r["y"], TIER)) for r in q)
        g, k, hg, hk = b[("gmoo", pi)][0], b[("lookup", pi)][0], hm("gmoo", pi), hm("lookup", pi)
        w, l, hw, hl = w + (g < k - 1e-3), l + (g > k + 1e-3), hw + (hg < hk - 1e-3), hl + (hg > hk + 1e-3)
        print(f"  profile {pi} {t}: gmoo {g:.3f}/{hg:.3f}  lookup {k:.3f}/{hk:.3f}  tier-2 {ok} of {len(q)}")
    mean = {m: (round(float(np.mean([b[(m, pi)][0] for pi in range(N)])), 3), round(float(np.mean([hm(m, pi) for pi in range(N)])), 3))
            for m in ("gmoo", "lookup")}
    print(f"training: globalMOO smaller in {w}, larger in {l}, equal in {N - w - l} of {N}")
    print(f"held-out: globalMOO smaller in {hw}, larger in {hl}, equal in {N - hw - hl} of {N}")
    print("mean worst miss, training / held-out:", mean)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))["prediction"]
    verdict = "HELD" if w >= p["held_if_at_least"] else "NOT HELD"
    print(f"primary: {w} of {N} (predicted {p['count']}, held if at least {p['held_if_at_least']}): {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seed", "protocol", "query", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query: resume at this profile")
    a = ap.parse_args()
    {"seed": seed, "protocol": protocol, "query": query, "heldout": heldout, "report": report}[a.cmd](a)
