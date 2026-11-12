"""steer@12: steer@11's profile round with the design drawn inside the usable region, and tight query bands.

  python -m bonsai_interp.moo.steer12 protocol | train | query | baselines | heldout | report
  bash scripts/moo_chain.sh steer12 train query baselines heldout report   # every stage in order, outside the session

In steer@11 only 35 of globalMOO's 687 design points met tier 2: 95% failed the drift limit, so the model was trained mostly where
no profile can be met, and the additive model, which corrects itself for 24 evaluations per profile near the zero design, had the
smaller miss in all 16 profiles. Changes from steer@11:
  - the box is 0.7 of steer@11's on every knob (the gain scale is multiplied by 0.7^3, so u still runs from -1 to 1 for every
    method). A pilot read 40 of steer@11's design points scaled by 0.6, 0.7 and 0.8 on the training prompts: 39, 28 and 16 of 40
    met tier 2, and the spread of the concept changes was 0.059, 0.087 and 0.126. 0.7 keeps most of the design usable and leaves
    the limits binding at the edge;
  - globalMOO's band starts at 0.05 for every profile (the additive model reached 0.04 in steer@11) and tightens by 0.75 when met;
  - a query the platform ends early is asked again, so every profile spends its budget.
The measure, prompts (64 training, steer@9's 64 fresh as held-out), drift limit, tier, methods, budgets and the rule for drawing
profiles are steer@11's. The model, the design and so the profiles are new.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer6 as S6
from bonsai_interp.moo import steer8 as S8
from bonsai_interp.moo import steer11 as S11  # the per-prompt read, 64 + 64 prompts, drift 0.08, 192 sequences per batch
from bonsai_interp.moo import steer_moo as M

S3.PROTOCOL = Path(__file__).parent / "steer12_protocol.json"
S3.EVALS = S3.OUT / "steer12.evals.jsonl"
S3.META = S3.OUT / "steer12.model.json"
S6.SOURCE = S3.EVALS
BOX = 0.7
S3.SCALE = 0.6 * BOX ** S3.POWER
START_TOL = 0.05


def protocol(a):
    S11.protocol(a)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))
    p["name"] = "steer@12"
    p["after"] = ("steer@11 (PR 1820): its measure, prompts, drift limit, tier, methods, budgets and rule for profiles; a box of 0.7 of "
                  "steer@11's on every knob, a new model and design, 16 profiles drawn by steer@7's rule and seed from this round's design")
    p["box"] = BOX
    p["scale"] = S3.SCALE
    p["box_note"] = ("pilot on the training prompts, 40 of steer@11's design points scaled by 0.6 / 0.7 / 0.8: 39 / 28 / 16 met tier 2, "
                     "spread of concept changes 0.059 / 0.087 / 0.126")
    p["start_tol"] = START_TOL
    p.pop("start_factor", None)
    p["methods"]["gmoo"] = ("each profile as globalMOO queries on this round's trained model: every concept within tol of its target (an "
                            "absolute band), uniq and coh at least the tier, drift at most drift_max; tol starts at start_tol and is "
                            "multiplied by tighten whenever a query meets it; a query that ends without meeting it is asked again at the "
                            "same tol, until the profile has used budget evaluations or a query returns no new evaluation")
    p["claims"] = {"precision_heldout": "on the 64 held-out prompts, globalMOO's best design is within log10(1.5) of the target on every "
                                        "concept in at least 8 of the 16 profiles",
                   "query": "on the training prompts, globalMOO's worst miss is smaller than the design lookup's in at least 8 of the 16 "
                            "profiles",
                   "model": "over the 16 profiles on the held-out prompts, globalMOO's worst miss is smaller than the additive model's in "
                            "more profiles than it is larger; de and lookup are reported whichever way they fall",
                   "reported": "the number of design points that meet tier 2, profiles within log10(1.5) on training and held-out for "
                               "every method, and the mean worst miss"}
    json.dump(p, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@12")


def train(a):
    lab = S3.Lab()
    lab.tag = "doe"
    lab.problem.evaluate(np.zeros((1, 24)))
    inv = S.GmooInverse(lab.problem, "mono-steer12")
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals,
               "doe_seconds": inv.doe_seconds}, open(S3.META, "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def query(a):
    lab = S3.Lab()
    m = json.load(open(S3.META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    z = lab.problem.evaluate(np.zeros((1, 24)))[0]
    tier = M.TIERS[0]
    for pi, prof in enumerate(S6.PROFILES):
        if pi < a.start:
            continue
        lab.tag = f"profile/{pi}/0"
        tol, used, rounds, met = START_TOL, 0, 0, 0
        while used < S6.BUDGET:
            g = [Goal(f"s_{c}", "=", float(z[lab.names.index(c)] + t), tol=float(tol)) for c, t in zip(S3.CON, prof)]
            g += [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", S3.DRIFT)]
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query or baselines: resume at this profile")
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "query": S8.run(query), "baselines": S8.run(S6.baselines), "heldout": S8.run(S6.heldout),
     "report": S8.run(S6.report)}[a.cmd](a)
