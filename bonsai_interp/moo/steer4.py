"""steer@4: steer@3b's value-target queries, replicated on four new concepts with a fresh globalMOO model.

  python -m bonsai_interp.moo.steer4 protocol   # writes steer4_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer4 train      # the design of experiments and one model
  python -m bonsai_interp.moo.steer4 query      # every cell as one value-target query
  python -m bonsai_interp.moo.steer4 baselines  # uniform line, band lines, the additive model, differential evolution
  python -m bonsai_interp.moo.steer4 heldout
  python -m bonsai_interp.moo.steer4 report

Everything but the concepts comes from steer@3 (bands, box, tiers, drift limit, prompts, baselines), and the query is steer@3b's:
each raised concept at least the best design point's score plus log10(1.5), each lowered one at most the mirror of that.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.inverse import solvers as S
from bonsai_interp.inverse.problem import Goal
from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer_moo as M

S3.CON = ["school", "software", "war", "family"]
S3.TASKS = [{"up": ["school"], "down": []}, {"up": ["software"], "down": []}, {"up": ["war"], "down": []}, {"up": ["family"], "down": []},
            {"up": ["school", "software"], "down": []}, {"up": ["war", "family"], "down": []},
            {"up": ["software"], "down": ["school"]}, {"up": ["war"], "down": ["family"]}]
S3.PROTOCOL = Path(__file__).parent / "steer4_protocol.json"
S3.EVALS = S3.OUT / "steer4.evals.jsonl"
S3.META = S3.OUT / "steer4.model.json"
METHODS = ("target", "additive", "line", "band", "de", "lookup")


def protocol(a):
    S3.protocol(a)
    p = json.load(open(S3.PROTOCOL, encoding="utf-8"))
    p["name"] = "steer@4"
    p["after"] = "steer@3 (PR 1809) and steer@3b (PR 1811): same bands, box, tiers, drift limit, prompts and baselines; new concepts and a new model"
    p["reach"] = S3.REACH
    p["methods"]["target"] = p["methods"].pop("gmoo").split(", then")[0] + (
        ", then each cell as one query: each raised concept at least t past the zero design, each lowered one at least t below it, where "
        "t is the cell's best design-point score plus reach (log10(2) where no design point meets the cell); uniq and coh at least the "
        "tier, drift at most drift_max; starts at the best design point; query_iter evaluations")
    p["claims"] = {"model": "over the 24 task-tier cells on the held-out prompts, the target queries beat the additive model in more cells "
                            "than they lose; the same count against line, band, de and lookup is reported whichever way it falls",
                   "climb": "on training prompts, the target queries beat the design lookup in at least 8 of the 24 cells"}
    json.dump(p, open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("protocol is steer@4")


def train(a):
    lab = S3.Lab()
    lab.tag = "doe"
    lab.problem.evaluate(np.zeros((1, 24)))
    inv = S.GmooInverse(lab.problem, "mono-steer4")
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
            g = [Goal(f"s_{c}", ">=", float(z[lab.names.index(c)] + t)) for c in task["up"]]
            g += [Goal(f"s_{c}", "<=", float(z[lab.names.index(c)] - t)) for c in task["down"]]
            g += [Goal("uniq", ">=", tier["uniq"]), Goal("coh", ">=", tier["coh"]), Goal("drift", "<=", S3.DRIFT)]
            n0 = len(lab.problem.history)
            res = inv.query(g, max_iter=S3.QUERY_ITER)
            print(lab.tag, res.extra.get("stopped"), len(lab.problem.history) - n0, "evals", flush=True)


def report(a):
    names, base, best = S3.best_points()
    rows = [json.loads(s) for s in open(S3.EVALS, encoding="utf-8")]
    held = {tuple(r["x"]): np.array(r["y"]) for r in rows if r["set"] == "held"}
    b0 = held[tuple([0.0] * 24)]
    tags = [r["tag"].split("/")[0] for r in rows if r["set"] == "train"]
    print("training evaluations by tag", {t: tags.count(t) for t in sorted(set(tags))})

    def hs(m, t, tr):
        u = best[(m, t, tr)][1]
        return None if u is None else S3.score(held[tuple(u)], S3.TASKS[t], b0, names)
    print("\nscore by cell, training / held-out")
    for t, task in enumerate(S3.TASKS):
        for tr in range(3):
            print(f"  task {t} {'+'.join(task['up'])}{' -' + '-'.join(task['down']) if task['down'] else ''} tier {tr + 1}: " +
                  "  ".join(f"{m} {S3._f(best[(m, t, tr)][0])}/{S3._f(hs(m, t, tr))}" for m in METHODS))
    print("\ntarget queries against each method, cells won / lost / tied")
    for nm, f in (("training", lambda m, t, tr: best[(m, t, tr)][0]), ("held-out", hs)):
        for m in METHODS[1:]:
            w = l = e = 0
            for t in range(len(S3.TASKS)):
                for tr in range(3):
                    x, y = f("target", t, tr), f(m, t, tr)
                    x, y = -9 if x is None else x, -9 if y is None else y
                    w, l, e = w + (x > y + 1e-3), l + (y > x + 1e-3), e + (abs(x - y) <= 1e-3)
            print(f"  {nm} vs {m}: {w} / {l} / {e}")
    print("\nmean held-out score:", {m: round(float(np.mean([hs(m, t, tr) or 0 for t in range(8) for tr in range(3)])), 3) for m in METHODS})
    climb = sum((best[("target", t, tr)][0] or -9) > (best[("lookup", t, tr)][0] or -9) + 1e-3 for t in range(8) for tr in range(3))
    print(f"climb claim: {climb} of 24 training cells beat lookup (needs 8)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "train", "query", "baselines", "heldout", "report"])
    ap.add_argument("--start", type=int, default=0, help="query: resume at this task, after an interruption")
    a = ap.parse_args()
    {"protocol": protocol, "train": train, "query": query, "baselines": S3.baselines, "heldout": S3.heldout, "report": report}[a.cmd](a)
