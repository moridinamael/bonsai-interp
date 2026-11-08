"""steer@5: steer@4's panel with twice the training prompts, and value targets raised each time a query meets them.

  python -m bonsai_interp.moo.steer5 protocol   # writes steer5_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer5 train
  python -m bonsai_interp.moo.steer5 query      # every cell: value targets, raised by reach whenever met, within BUDGET evaluations
  python -m bonsai_interp.moo.steer5 baselines
  python -m bonsai_interp.moo.steer5 heldout
  python -m bonsai_interp.moo.steer5 report

In steer@4 the queries climbed past the design point on the 16 training prompts in 12 of 24 cells, but only 4 of those climbs held on
the held-out prompts: 16 prompts are noisy enough for 12 evaluations to fit. Here the training set is 32 prompts (steer@4's 16 and 16
new ones, none from the held-out set), so every method fits less noise, and each cell gets BUDGET evaluations, spent by raising its
target each time a query meets it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer4 as S4
from bonsai_interp.moo import steer_moo as M

M.TRAIN = M.TRAIN + ["The first thing she noticed was", "Over the course of the afternoon,", "At the far end of the street there was",
                     "My grandfather used to say that", "When the meeting finally ended,", "It took us almost an hour to",
                     "The report published last week found that", "Before anyone could answer,", "I still remember the moment when",
                     "On the second day of the trip,", "What they did not tell us was", "The best part of the whole thing was",
                     "A few minutes later the phone rang and", "Looking back on it now,", "The plan was simple: we would",
                     "Nobody in the village could explain why"]
assert not set(M.TRAIN) & set(M.HELD) and len(set(M.TRAIN)) == 32
S3.PROTOCOL = Path(__file__).parent / "steer5_protocol.json"
S3.EVALS = S3.OUT / "steer5.evals.jsonl"
S3.META = S3.OUT / "steer5.model.json"
BUDGET = 24


def protocol(a):
    S4.protocol(a)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))
    p["name"] = "steer@5"
    p["after"] = ("steer@4 (PR 1812): same concepts, tasks, bands, box, tiers, drift limit, held-out prompts and baselines; 32 training "
                  "prompts (steer@4's 16 plus 16 new); a new model")
    p["budget"] = BUDGET
    p["pilots"] = "none: the box and concepts are steer@4's, whose pilot kept the box"
    p["methods"]["target"] = p["methods"]["target"].replace(
        "; query_iter evaluations", "; whenever a query meets its goals, t rises by reach and the cell is queried again, until the cell "
                                    "has used budget evaluations or a query ends without meeting them")
    p["claims"] = {"transfer": "on the held-out prompts, the target queries beat the design lookup in more of the 24 cells than they lose to it",
                   "model": "over the 24 cells on the held-out prompts, the target queries beat the additive model in more cells than they "
                            "lose; the same count against line, band and de is reported whichever way it falls"}
    json.dump(p, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@5")


def train(a):
    lab = S3.Lab()
    lab.tag = "doe"
    lab.problem.evaluate(np.zeros((1, 24)))
    inv = S.GmooInverse(lab.problem, "mono-steer5")
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals,
               "doe_seconds": inv.doe_seconds}, open(S3.META, "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def query(a):
    _, _, best = S3.best_points()
    lab = S3.Lab()
    m = json.load(open(S3.META))
    inv = S.GmooInverse.attach(lab.problem, m["model_id"], m["trial_id"])
    lab.problem.scales = np.maximum(inv.doe_y.std(0), 1e-3)
    z = lab.problem.evaluate(np.zeros((1, 24)))[0]
    for ti, task in enumerate(S3.TASKS):
        if ti < a.start:
            continue
        for tr, tier in enumerate(M.TIERS):
            lab.tag = f"target/{ti}/{tr}"
            sc = best[("lookup", ti, tr)][0]
            t = (np.log10(2) if sc is None else sc) + S3.REACH
            used, rounds = 0, 0
            while used < BUDGET:
                g = [Goal(f"s_{c}", ">=", float(z[lab.names.index(c)] + t)) for c in task["up"]]
                g += [Goal(f"s_{c}", "<=", float(z[lab.names.index(c)] - t)) for c in task["down"]]
                g += [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", S3.DRIFT)]
                n0 = len(lab.problem.history)
                res = inv.query(g, max_iter=BUDGET - used)
                used += len(lab.problem.history) - n0
                rounds += 1
                if res.extra.get("stopped") not in ("satisfied", "satisfied_local"):
                    break
                t += S3.REACH
            print(lab.tag, rounds, "rounds", used, "evals, last target", round(float(t), 2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query: resume at this task, after an interruption")
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "query": query, "baselines": S3.baselines, "heldout": S3.heldout, "report": S4.report}[a.cmd](a)
