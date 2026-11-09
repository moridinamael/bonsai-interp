"""steer@8: steer@7's reachable whole profiles with 64 training prompts and a new model. In steer@7, profiles hit within 1.5x on the
32 training prompts fell apart on unseen ones for every method, as single pushes had with 16 prompts in steer@4 before steer@5
doubled them.

  python -m bonsai_interp.moo.steer8 protocol | train | query | baselines | heldout | report

The profiles follow steer@7's rule, applied to this round's own design (it doesn't exist until `train`): the midpoint of two design
points that meet tier 2, with the same seed. Everything else is steer@7's.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer6 as S6
from bonsai_interp.moo import steer7 as S7
from bonsai_interp.moo import steer_moo as M

M.TRAIN = M.TRAIN + ["The woman behind the counter looked up and", "By the time the train pulled into the station,", "Every morning before work he would",
                     "The letter from the council arrived on a Tuesday and", "We had been driving for hours when", "There is an old story about a",
                     "The first chapter begins with", "After the storm passed, the neighbours", "Most of what I know about it comes from",
                     "Halfway through the speech, the mayor", "It was late in the season when", "Her brother had always insisted that",
                     "The sign on the door said", "When the results came back, the team", "In a small town near the border,",
                     "I asked the guide what would happen if", "The kitchen smelled of", "For most of the afternoon the two of them",
                     "According to the old map, the path", "The new manager started by", "Nobody had used the room since",
                     "On the radio that night they said", "The children ran to the window because", "He kept a notebook in which he",
                     "At the bottom of the box we found", "The festival had been cancelled twice before", "She wrote back the next day to say that",
                     "The last time I saw him he was", "From the top of the hill you could see", "The rules of the game were simple:",
                     "When the lights came back on,", "My first job after school was"]
assert not set(M.TRAIN) & set(M.HELD) and len(set(M.TRAIN)) == 64
S3.PROTOCOL = Path(__file__).parent / "steer8_protocol.json"
S3.EVALS = S3.OUT / "steer8.evals.jsonl"
S3.META = S3.OUT / "steer8.model.json"
S6.SOURCE = S3.EVALS  # profiles come from this round's own design


def _profiles():
    S6.PROFILES, n = S7.reachable()
    print(n, "design points meet tier 2;", len(S6.PROFILES), "profiles", flush=True)


def protocol(a):
    S6.PROFILES = []
    S6.protocol(a)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))
    p["name"] = "steer@8"
    p["after"] = ("steer@7 (PR 1815): its rule for profiles, methods, budget, tier 2 and held-out prompts; 64 training prompts (steer@5's "
                  "32 plus 32 new); a new model; the profiles are drawn by steer@7's rule and seed from this round's design after training")
    p["profiles"] = "drawn after training: the midpoint of two design points that meet tier 2, steer@7's seed 71, 16 profiles"
    p.pop("levels", None)
    p["claims"] = {"precision_heldout": "on the held-out prompts, globalMOO's mean worst miss over the 16 profiles is below steer@7's 0.49 log10",
                   "model": "over the 16 profiles on the held-out prompts, globalMOO's worst miss is smaller than the additive model's in more "
                            "profiles than it is larger; de and lookup are reported whichever way they fall",
                   "precision": "reported: profiles within log10(1.5) on every concept, training and held-out, for every method"}
    json.dump(p, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@8")


def train(a):
    lab = S3.Lab()
    lab.tag = "doe"
    lab.problem.evaluate(np.zeros((1, 24)))
    inv = S.GmooInverse(lab.problem, "mono-steer8")
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals,
               "doe_seconds": inv.doe_seconds}, open(S3.META, "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def run(f):
    def g(a):
        _profiles()
        f(a)
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query or baselines: resume at this profile")
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "query": run(S6.query), "baselines": run(S6.baselines), "heldout": run(S6.heldout),
     "report": run(S6.report)}[a.cmd](a)
