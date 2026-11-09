"""steer@9: how much of steer@8's held-out miss is the 16 held-out prompts? steer@8's designs, re-read on 64 fresh prompts.

  python -m bonsai_interp.moo.steer9 protocol   # writes steer9_protocol.json: commit it alone, before anything is scored
  python -m bonsai_interp.moo.steer9 read       # steer@8's held-out designs on the fresh prompts, as two halves of 32
  python -m bonsai_interp.moo.steer9 report

In steer@8 no method kept a four-concept profile within 1.5x on the 16 held-out prompts, and every method's mean worst miss sat
between 0.45 and 0.58 log10. The design points themselves moved about as much: the median over steer@8's held-out designs of the
worst-concept gap between a design's change on the 64 training prompts and on the 16 held-out ones is 0.43 log10. That gap is
either sampling noise from 16 prompts or a real shift between prompt sets. This round re-reads the same 52 designs (and the zero
design) on 64 fresh prompts, none of them in steer@8's training or held-out sets, scored as two halves of 32 so the halves measure
the sampling noise. No new optimization: no globalMOO queries, no new model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bonsai_interp.moo import steer3 as S3
from bonsai_interp.moo import steer6 as S6
from bonsai_interp.moo import steer8 as S8
from bonsai_interp.moo import steer_moo as M

FRESH = ["The bus was late again, so we", "Under the bridge there used to be", "She had never been good at", "The doctor came in and",
         "At the end of the corridor a door", "The hardest part of the job was", "Two weeks after the move,", "He put down the newspaper and",
         "The garden behind the house had", "On Sundays my aunt would", "The meeting was supposed to start at", "Somewhere in the distance a dog",
         "If I had known then what I know now,", "The shop on the corner sold", "During the long winter the family", "The professor paused and asked",
         "Outside the window the rain", "The first time I tried it,", "In the middle of the night someone", "The village hall was full of",
         "My sister called to tell me", "The road ended at a gate where", "When the bell rang, the students", "We sat on the steps and talked about",
         "The engineer explained that the", "His last words before leaving were", "The box in the attic contained", "Half the town turned out to",
         "The recipe called for", "By noon the heat was", "The captain ordered the crew to", "She kept the photograph because",
         "The old factory by the river", "Nobody answered when I knocked, so", "The museum had a room full of", "After the game the coach told us",
         "In those days a letter took", "The fisherman pulled in his net and", "The sound of the music drifted", "Our teacher once told us that",
         "The farmer looked at the sky and", "Down at the harbour the boats", "The interview lasted less than", "There was a rumour going round that",
         "The price of bread had", "He tried the key in the lock, but", "The nurse wrote something on the chart and", "Across the valley the lights of",
         "The book I was reading at the time", "At the market that morning", "The soldier wrote home to say", "The baby finally fell asleep, and",
         "The parish records show that", "Every summer the river", "The clerk at the desk asked whether", "When the snow melted we found",
         "A small crowd had gathered to", "The computer in the library", "My father never talked about", "The tour guide pointed to",
         "The deadline was moved twice, and", "In the back of the car the", "The chef tasted the soup and", "Long after everyone had gone home,"]
assert len(set(FRESH)) == 64 and not set(FRESH) & (set(M.TRAIN) | set(M.HELD))
HALVES = {"a": FRESH[:32], "b": FRESH[32:]}
SOURCE = S3.EVALS  # steer@8's cache: its designs, training and held-out reads
S3.PROTOCOL = Path(__file__).parent / "steer9_protocol.json"
EVALS = {h: S3.OUT / f"steer9{h}.evals.jsonl" for h in HALVES}
GAP8 = 0.428  # steer@8: median over its held-out designs of the worst-concept |training change - held-out change|, log10


def designs():
    """steer@8's held-out designs: each method's best point per profile, from steer@8's cache."""
    S8._profiles()
    _, _, best = S6.best_points()
    return best, sorted({tuple(v[1]) for v in best.values() if v[1] is not None})


def protocol(a):
    p8 = json.load(open(Path(__file__).parent / "steer8_protocol.json", encoding="utf-8"))
    best, X = designs()
    json.dump({"name": "steer@9", "after": "steer@8 (PR 1816): its 16 profiles, its methods' best designs and its scoring, re-read on fresh prompts",
               "bundles_sha256": p8["bundles_sha256"], "profiles": S6.PROFILES, "designs": len(X), "fresh_prompts": FRESH,
               "halves": "the first 32 fresh prompts and the last 32, read separately; the 64-prompt value of a concept is log10 of the "
                         "mean of the two halves' soft rates (floor as in steer@3)",
               "gap8": GAP8,
               "claims": {"replicate": "on the 64 fresh prompts, globalMOO's mean worst miss over the 16 profiles is below the additive "
                                       "model's, and its worst miss is smaller than the additive model's in more profiles than it is larger",
                          "sampling": "the median over the designs of the worst-concept |change on the training prompts - change on the 64 "
                                      "fresh prompts| is below gap8, the same measure on steer@8's 16 held-out prompts",
                          "reported": "the split-half gap (the same measure between the two halves of 32), per-concept slope and offset "
                                      "from training to fresh prompts, profiles within log10(1.5) on every concept for every method, and "
                                      "globalMOO against de and lookup"}},
              open(S3.PROTOCOL, "w", encoding="utf-8"), indent=1)
    print("wrote", S3.PROTOCOL, len(X), "designs")


def read(a):
    _, X = designs()
    lab = S3.Lab()
    lab.tag = "fresh"
    for h, prompts in HALVES.items():
        S3.EVALS = EVALS[h]
        lab.cache, lab.zero, lab.prompts = {}, {}, prompts
        if S3.EVALS.exists():
            for line in open(S3.EVALS, encoding="utf-8"):
                r = json.loads(line)
                lab.cache[(r["set"], tuple(r["x"]))] = np.array(r["y"])
        print("half", h, len(X), "designs on", len(prompts), "fresh prompts", flush=True)
        lab.problem.evaluate(np.array(X))


def _load(path, s):
    return {tuple(r["x"]): np.array(r["y"]) for r in map(json.loads, open(path, encoding="utf-8")) if r["set"] == s}


def report(a):
    best, X = designs()
    names = list(S3.T.CONCEPTS)
    idx = [names.index(c) for c in S3.CON]
    tr, h16 = _load(SOURCE, "train"), _load(SOURCE, "held")
    A, B = _load(EVALS["a"], "held"), _load(EVALS["b"], "held")

    def join(ya, yb):
        s = (10 ** ya[:24] - M.FLOOR + 10 ** yb[:24] - M.FLOOR) / 2
        return np.log10(s + M.FLOOR)
    F = {x: join(A[x], B[x]) for x in A if x in B}
    z = tuple([0.0] * 24)
    ch = {nm: {x: (d[x][idx] - d[z][idx]) for x in X} for nm, d in (("train", tr), ("held16", h16), ("a", A), ("b", B), ("fresh", F))}
    gap = {k: float(np.median([np.abs(ch[u][x] - ch[v][x]).max() for x in X if any(x)]))
           for k, (u, v) in {"train-held16": ("train", "held16"), "train-fresh64": ("train", "fresh"), "half a-half b": ("a", "b")}.items()}
    print(len(X), "designs; median worst-concept gap (log10):", {k: round(v, 3) for k, v in gap.items()})
    D = np.array([ch["train"][x] for x in X if any(x)])
    for nm in ("held16", "fresh"):
        H = np.array([ch[nm][x] for x in X if any(x)])
        print(f"  training -> {nm}: " + "  ".join(f"{c} slope {np.polyfit(D[:, k], H[:, k], 1)[0]:.2f} offset {np.mean(H[:, k] - D[:, k]):+.2f}"
                                                  for k, c in enumerate(S3.CON)))

    def wm(m, p, nm):
        u = best[(m, p, 0)][1]
        return None if u is None else float(np.abs(ch[nm][tuple(u)] - np.array(S6.PROFILES[p])).max())
    print("\nworst miss by profile (log10), held16 / fresh64")
    for p, prof in enumerate(S6.PROFILES):
        print(f"  profile {p} {prof}: " + "  ".join(f"{m} {S3._f(wm(m, p, 'held16'))}/{S3._f(wm(m, p, 'fresh'))}" for m in S6.METHODS))
    for nm in ("held16", "fresh"):
        print(f"\n{nm}: mean worst miss", {m: round(float(np.mean([wm(m, p, nm) or 2 for p in range(16)])), 3) for m in S6.METHODS})
        print(f"{nm}: profiles within log10(1.5) on every concept",
              {m: sum((wm(m, p, nm) or 9) <= S6.PRECISE for p in range(16)) for m in S6.METHODS})
        for m in S6.METHODS[1:]:
            w = sum((wm("gmoo", p, nm) or 9) < (wm(m, p, nm) or 9) - 1e-3 for p in range(16))
            l = sum((wm(m, p, nm) or 9) < (wm("gmoo", p, nm) or 9) - 1e-3 for p in range(16))
            print(f"  {nm} globalMOO vs {m}: {w} / {l} / {16 - w - l}")
    g, ad = (np.mean([wm(m, p, "fresh") or 2 for p in range(16)]) for m in ("gmoo", "additive"))
    w = sum((wm("gmoo", p, "fresh") or 9) < (wm("additive", p, "fresh") or 9) - 1e-3 for p in range(16))
    l = sum((wm("additive", p, "fresh") or 9) < (wm("gmoo", p, "fresh") or 9) - 1e-3 for p in range(16))
    print(f"\nreplicate claim: mean {g:.3f} vs additive {ad:.3f}, profiles {w} / {l}:", "HELD" if g < ad and w > l else "NOT HELD")
    print(f"sampling claim: train-fresh64 gap {gap['train-fresh64']:.3f} vs gap8 {GAP8}:", "HELD" if gap["train-fresh64"] < GAP8 else "NOT HELD")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["protocol", "read", "report"])
    a = ap.parse_args()
    {"protocol": protocol, "read": read, "report": report}[a.cmd](a)
