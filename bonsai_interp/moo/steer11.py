"""steer@11: steer@8's whole-profile round, scored per prompt.

  python -m bonsai_interp.moo.steer11 protocol | train | query | baselines | heldout | report

steer@10 showed that a design's effect, scored as the mean over prompts of each prompt's log10 soft rate (geo), is the same on any
prompt set (split-half correlations 0.88 to 0.96, slope from training to fresh prompts about 1.05), while the measure every round
had used (log10 of the mean soft rate) is not. steer@7 to @9's held-out misses sat at that old measure's own floor. This round is
steer@8 with geo outputs: 64 training prompts, a new model and design, 16 reachable profiles drawn by steer@7's rule and seed from
this round's design, tier 2, and the same methods and budget. Changes besides the measure:
  - drift (the mean |change| of the 20 other concepts) runs at about 0.41 of its old value under geo (median 0.072 against 0.174
    over steer@8's held-out designs, steer@10's reads), so its limit is 0.08 in place of 0.20;
  - the held-out set is steer@9's 64 fresh prompts, which no model has been trained or scored against for selection;
  - globalMOO's band starts at the design lookup's own miss (capped at 0.3) and tightens by 0.75 each time a query meets it; in
    steer@8 it started at 0.75 of that miss and 13 of 16 queries spent the whole budget without meeting it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer6 as S6
from bonsai_interp.moo import steer7 as S7
from bonsai_interp.moo import steer8 as S8
from bonsai_interp.moo import steer9 as S9
from bonsai_interp.moo import steer_moo as M

M.HELD = S9.FRESH
assert len(M.TRAIN) == 64 and not set(M.TRAIN) & set(M.HELD)
S3.PROTOCOL = Path(__file__).parent / "steer11_protocol.json"
S3.EVALS = S3.OUT / "steer11.evals.jsonl"
S3.META = S3.OUT / "steer11.model.json"
S6.SOURCE = S3.EVALS
S3.BATCH = 3  # 192 sequences per forward batch, fixed for the whole round
S3.DRIFT = 0.08
START_FACTOR = 1.0


def _read(self, U):
    """steer@3's outputs with each concept as the mean over prompts of log10 (soft rate + floor)."""
    import torch
    D = len(U)
    e, g, lp = S3.T.generate(self.panel, self.tok, self.prompts, S3.gains(U), S3.NEW)
    P = g.shape[1]
    soft = torch.stack([lp[:, :, i].exp().sum(-1).mean(1) for i in self.ids], 1).view(D, P, -1).cpu().numpy()
    uniq = np.array([[len(set(x.tolist())) / g.shape[2] for x in row] for row in g.cpu()]).mean(1)
    coh = S3.T.clean_logp(self.panel, e, g).mean(1).cpu().numpy()
    return np.c_[np.log10(soft + M.FLOOR).mean(1), uniq, coh]


S3.Lab._read = _read


def protocol(a):
    S6.PROFILES = []
    S8.protocol(a)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))
    p["name"] = "steer@11"
    p["after"] = ("steer@8 (PR 1816) with steer@10's per-prompt measure (PR 1818): 64 training prompts, a new model, 16 profiles drawn "
                  "by steer@7's rule and seed from this round's design, tier 2, methods and budget as steer@8")
    p["measure"] = "each concept: the mean over prompts of log10 of the prompt's soft rate plus the floor (steer@10's geo)"
    p["drift_max"] = S3.DRIFT
    p["drift_note"] = "0.41 x 0.20: geo drift over arith drift, median over steer@8's held-out designs on the training prompts (steer@10's reads)"
    p["heldout_prompts"] = M.HELD
    p["batch_designs"] = S3.BATCH
    p["start_factor"] = START_FACTOR
    p["methods"]["gmoo"] = p["methods"]["gmoo"].replace("the smaller of start_tol and tighten x the best design point's miss",
                                                        "the smaller of start_tol and start_factor x the best design point's miss")
    p["claims"] = {"precision_heldout": "on the 64 held-out prompts, globalMOO's best design is within log10(1.5) of the target on every "
                                        "concept in at least 8 of the 16 profiles",
                   "model": "over the 16 profiles on the held-out prompts, globalMOO's worst miss is smaller than the additive model's in "
                            "more profiles than it is larger; de and lookup are reported whichever way they fall",
                   "reported": "profiles within log10(1.5) on training and held-out for every method, and the mean worst miss"}
    json.dump(p, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@11")


def train(a):
    lab = S3.Lab()
    lab.tag = "doe"
    lab.problem.evaluate(np.zeros((1, 24)))
    inv = S.GmooInverse(lab.problem, "mono-steer11")
    json.dump({"model_id": inv.model_id, "trial_id": inv.trial.id, "project_id": inv.project_id, "doe_evals": inv.doe_evals,
               "doe_seconds": inv.doe_seconds}, open(S3.META, "w"))
    print(f"trained: model {inv.model_id}, trial {inv.trial.id}, {inv.doe_evals} design points in {inv.doe_seconds:.0f} s", flush=True)


def query(a):
    """steer@6's query, with the band starting at start_factor x the lookup's miss."""
    _, _, best = S6.best_points()
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
        sc = best[("lookup", pi, 0)][0]
        tol = S6.START_TOL if sc is None else min(S6.START_TOL, START_FACTOR * -sc)
        used = rounds = 0
        while used < S6.BUDGET:
            g = [Goal(f"s_{c}", "=", float(z[lab.names.index(c)] + t), tol=float(tol)) for c, t in zip(S3.CON, prof)]
            g += [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", S3.DRIFT)]
            n0 = len(lab.problem.history)
            res = inv.query(g, max_iter=S6.BUDGET - used)
            used += len(lab.problem.history) - n0
            rounds += 1
            if res.extra.get("stopped") not in ("satisfied", "satisfied_local"):
                break
            tol *= S6.TIGHTEN
        print(lab.tag, rounds, "rounds", used, "evals, last tol", round(float(tol), 3), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query or baselines: resume at this profile")
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "query": S8.run(query), "baselines": S8.run(S6.baselines), "heldout": S8.run(S6.heldout),
     "report": S8.run(S6.report)}[a.cmd](a)
