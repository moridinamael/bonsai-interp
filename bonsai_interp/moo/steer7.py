"""steer@7: whole profiles the panel can reach. steer@6's targets were out of joint reach for every method, so it measured reach,
not precision. Here each target is the midpoint of two feasible design points' profiles: reachable in principle, not in the design.

  python -m bonsai_interp.moo.steer7 protocol | seed | query | baselines | heldout | report

Everything else is steer@6's (model 7351, its design, band lines and prompts, the absolute-band queries, the additive model and DE),
at steer@3's tier 2 only, since steer@6's tiers didn't bind: 16 profiles, one cell each.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer6 as S6
from bonsai_interp.moo import steer_moo as M

S3.PROTOCOL = Path(__file__).parent / "steer7_protocol.json"
S3.EVALS = S3.OUT / "steer7.evals.jsonl"  # seeded from steer@5's cache (S6.SOURCE), so steer@6's tags stay out
M.TIERS = [M.TIERS[1]]


def reachable(n=16, seed=71):
    names = list(S3.T.CONCEPTS)
    rows = [r for r in map(json.loads, open(S6.SOURCE, encoding="utf-8")) if r["set"] == "train"]
    base = next(np.array(r["y"]) for r in rows if not any(r["x"]))
    ok = [np.array(r["y"]) for r in rows if r["tag"] == "doe" and S3.meets(r["y"], M.TIERS[0])]
    prof = np.array([[y[names.index(c)] - base[names.index(c)] for c in S3.CON] for y in ok])
    r = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        i, j = r.choice(len(prof), 2, replace=False)
        t = [round(float(v), 2) for v in (prof[i] + prof[j]) / 2]
        if t not in out:
            out.append(t)
    return out, len(ok)


S6.PROFILES, N_FEASIBLE = reachable()


def protocol(a):
    S6.protocol(a)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))
    p["name"] = "steer@7"
    p["after"] = ("steer@6 (PR 1814): its methods, model 7351, design, band lines and prompts; tier 2 only; targets drawn as the midpoint "
                  "of two of the design's points that meet tier 2 (log10 change per concept, rounded to 0.01)")
    p["feasible_design_points"] = N_FEASIBLE
    p.pop("levels", None)
    p["claims"] = {"precision": "on the training prompts, globalMOO's best design meets the cell with every concept within log10(1.5) of its "
                                "target in at least 8 of the 16 profiles",
                   "model": "over the 16 profiles on the held-out prompts, globalMOO's worst miss is smaller than the additive model's in "
                            "more profiles than it is larger; the same count against de and lookup is reported whichever way it falls"}
    json.dump(p, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@7,", N_FEASIBLE, "feasible design points")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "seed", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query or baselines: resume at this profile")
    a = ap.parse_args()
    {"protocol": protocol, "seed": S6.seed, "query": S6.query, "baselines": S6.baselines, "heldout": S6.heldout, "report": S6.report}[a.cmd](a)
