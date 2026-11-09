"""steer@10: does a per-prompt measure make a design's effect agree across prompt sets?

  python -m bonsai_interp.moo.steer10 protocol   # writes steer10_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer10 read       # steer@8's 52 held-out designs and the zero design, per prompt, on three sets
  python -m bonsai_interp.moo.steer10 report

Every round so far scored a concept as log10 of the mean soft rate over prompts (an arithmetic mean), so the few prompts where a
concept is already likely dominate it. In steer@9 two disjoint halves of 32 fresh prompts disagreed about steer@8's designs by a
median worst-concept 0.45 log10, as much as the training prompts disagreed with any new set. This round keeps every prompt's soft
rate, so both measures come from the same reads: the arithmetic one used so far, and the geometric one (the mean over prompts of
log10 of each prompt's soft rate), which weights every prompt alike. The sets are steer@8's 64 training prompts and steer@9's two
halves of 32 fresh prompts. No optimization: if the geometric measure is the more reliable, the next round optimizes against it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer9 as S9  # steer@8's prompts, designs and cache
from bonsai_interp.moo import steer_moo as M

S3.PROTOCOL = Path(__file__).parent / "steer10_protocol.json"
SETS = {"train": M.TRAIN, "a": S9.HALVES["a"], "b": S9.HALVES["b"]}
READS = S3.OUT / "steer10.reads.jsonl"
SEQ = 192  # sequences per forward batch, steer@8's


def protocol(a):
    _, X = S9.designs()
    p9 = json.load(open(Path(__file__).parent / "steer9_protocol.json", encoding="utf-8"))
    json.dump({"name": "steer@10", "after": "steer@9 (PR 1817): steer@8's 52 held-out designs and the zero design; steer@8's 64 training "
                                            "prompts and steer@9's two halves of 32 fresh prompts",
               "bundles_sha256": p9["bundles_sha256"], "designs": len(X), "sets": {k: len(v) for k, v in SETS.items()},
               "measures": {"arith": "log10 of the mean over prompts of the soft rate, plus the floor (every round so far)",
                            "geo": "the mean over prompts of log10 of the prompt's soft rate plus the floor"},
               "change": "a design's change on a concept is its measure minus the zero design's, on the same prompts",
               "claims": {"reliability": "for at least 3 of the 4 panel concepts, the correlation across the 52 designs between their changes "
                                        "on half a and on half b is higher under geo than under arith",
                          "transfer": "for at least 3 of the 4 panel concepts, the correlation across the designs between their changes on "
                                      "the training prompts and on the 64 fresh prompts is higher under geo than under arith",
                          "reported": "both measures' median worst-concept gaps (half a - half b, training - fresh), slopes and offsets "
                                      "from training to fresh, the spread of the changes across designs, and the arith figures beside "
                                      "steer@9's (the same designs and prompts, read again)"}},
              open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("wrote", S3.PROTOCOL, len(X), "designs")


def read(a):
    _, X = S9.designs()
    X = [tuple([0.0] * 24)] + [x for x in X if any(x)]
    lab = S3.Lab()
    done = set()
    if READS.exists():
        done = {(r["set"], tuple(r["x"])) for r in map(json.loads, open(READS, encoding="utf-8"))}
    for s, prompts in SETS.items():
        todo = [x for x in X if (s, x) not in done]
        per = max(1, SEQ // len(prompts))
        print(s, len(todo), "designs to read on", len(prompts), "prompts", flush=True)
        for j in range(0, len(todo), per):
            U = np.array(todo[j:j + per])
            with torch.no_grad():
                _, g, lp = S3.T.generate(lab.panel, lab.tok, prompts, S3.gains(U), S3.NEW)
            soft = torch.stack([lp[:, :, i].exp().sum(-1).mean(1) for i in lab.ids], 1).view(len(U), len(prompts), -1).cpu().numpy()
            with open(READS, "a", encoding="utf-8") as f:
                for u, sp in zip(U, soft):
                    f.write(json.dumps({"set": s, "x": [float(v) for v in u], "soft": [[float(f"{v:.4g}") for v in row] for row in sp]}) + "\n")
            print(f"  {s} {j + len(U)} of {len(todo)}", flush=True)


def report(a):
    best, X = S9.designs()
    names = list(S3.T.CONCEPTS)
    idx = [names.index(c) for c in S3.CON]
    R = {(r["set"], tuple(r["x"])): np.array(r["soft"])[:, idx] for r in map(json.loads, open(READS, encoding="utf-8"))}
    z = tuple([0.0] * 24)
    D = [x for x in X if any(x)]
    meas = {"arith": lambda sp: np.log10(sp.mean(0) + M.FLOOR), "geo": lambda sp: np.log10(sp + M.FLOOR).mean(0)}
    out = {}
    for mn, f in meas.items():
        def val(s, x):
            if s == "fresh":
                return f(np.r_[R[("a", x)], R[("b", x)]])
            return f(R[(s, x)])
        ch = {s: np.array([val(s, x) - val(s, z) for x in D]) for s in ("train", "a", "b", "fresh")}
        corr = {k: [float(np.corrcoef(ch[u][:, c], ch[v][:, c])[0, 1]) for c in range(4)]
                for k, (u, v) in {"a-b": ("a", "b"), "train-fresh": ("train", "fresh")}.items()}
        gap = {k: float(np.median(np.abs(ch[u] - ch[v]).max(1))) for k, (u, v) in {"a-b": ("a", "b"), "train-fresh": ("train", "fresh")}.items()}
        out[mn] = corr
        print(f"\n{mn}: median worst-concept gap", {k: round(v, 3) for k, v in gap.items()},
              " spread of changes (sd across designs, training):", dict(zip(S3.CON, ch["train"].std(0).round(3))))
        for k, v in corr.items():
            print(f"  correlation {k}: " + "  ".join(f"{c} {r:.2f}" for c, r in zip(S3.CON, v)))
        print("  training -> fresh: " + "  ".join(f"{c} slope {np.polyfit(ch['train'][:, k], ch['fresh'][:, k], 1)[0]:.2f} "
                                                  f"offset {np.mean(ch['fresh'][:, k] - ch['train'][:, k]):+.2f}" for k, c in enumerate(S3.CON)))
    for k, nm in (("a-b", "reliability"), ("train-fresh", "transfer")):
        n = sum(g > r for g, r in zip(out["geo"][k], out["arith"][k]))
        print(f"{nm} claim: geo higher in {n} of 4 concepts:", "HELD" if n >= 3 else "NOT HELD")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "read", "report"])
    a = ap.parse_args()
    {"protocol": protocol, "read": read, "report": report}[a.cmd](a)
