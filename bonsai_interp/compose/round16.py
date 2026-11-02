"""compose@16: covers by the neighbouring token and by embedding class. compose@7-12 covered a pole with the read token
(sets of up to 16 tokens, pairs, triples) and ended the n-gram line; most poles left fire on many read tokens. This round
lets the read token vary and fixes something else instead:
  next    the token right after the read token is in S (at most 8 tokens): the pole anticipates what comes next
  prev    the token right before the read token is in S (at most 8 tokens): any token after S
  class   the read token is in a class of 9-256 tokens the model's input embedding puts together (the tokens nearest the
          centroid of the pole's commonest read tokens), a category rather than a list

  python -m bonsai_interp.compose.round16 protocol | select | build | score | report   (same stages and flags as round7.py)

The engine is compose@7's (round7.py): the same event sample, test split, reader and judge (quick-8+8@2), with a placebo.
Tests:
  next, prev   fire: a snippet ending on a token whose next (previous) token is in S; quiet: a snippet ending on the same
               read token whose next (previous) token is not (strata: the read token). A next snippet ends at the read
               token, so the model never sees the next token: the pole has to expect it.
  class        compose@7's word test with T = the class: a token of the class against other tokens of the same frequency
               band and domain.
Selection, per eligible pole not assignment-grade (atlas, after compose@12), with >= 32 sampled activations in >= 8
windows:
  - next, prev: the fewest of the commonest next (previous) tokens over the pole's sample that reach half of it, at most 8,
    each seen at least twice;
  - class: seeds are the read tokens seen at least twice; the class is every token seen >= SEL["class_min_count"] times
    in the corpus with cosine >= tau to the seeds' count-weighted centroid, for the largest tau on a 0.05 grid (0.6 down
    to 0.15) whose class reaches half of the sample; it must hold 9-256 tokens (8 or fewer is a word cover, which
    compose@7-12 had);
  - precision >= SEL["min_precision"]; the pole keeps its best hypothesis by coverage x precision; the best SEL["cap"] are
    tested.
A pole is assignment-grade when its cover held; its coverage is the cover's.
"""
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import conds4 as CD
from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round7 as R7
from bonsai_interp.compose import round13 as R13
from bonsai_interp.events import sample as EV
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR
from bonsai_interp.sweeplib import kinds as SK

SEL = {"min_events": 32, "min_windows": 8, "cov": 0.5, "max_items": 8, "min_seen": 2, "class_min_count": 20,
       "class_size": [9, 256], "tau": [0.6, 0.15, 0.05], "min_precision": 0.2, "cap": 12000}
SHIFT = {"next": 1, "prev": -1}
CAND_MAX = 200_000  # fire and quiet candidates per source and cover, before the per-token cap
EXCLUDED = "assignment-grade poles (atlas, after compose@12)"


def _install():
    R7.PROTOCOL, R7.BASE = "compose@16", "bonsai1.compose16"
    R7.select, R7.build = select, build
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.extend([Path(R13.__file__), Path(__file__)]))


def _set(keys):
    """(sorted set, coverage) of the fewest commonest keys reaching SEL["cov"], each seen >= SEL["min_seen"], or None."""
    vals, cs = np.unique(keys, return_counts=True)
    o = np.argsort(-cs, kind="stable")
    m = int(np.searchsorted(np.cumsum(cs[o]) / len(keys), SEL["cov"]) + 1)
    if m > SEL["max_items"] or cs[o[m - 1]] < SEL["min_seen"]:
        return None
    S = np.sort(vals[o[:m]])
    return S, float(np.isin(keys, S).mean())


def select(a):
    P, man = R3._setup(a)
    if man and man["events"]["sha256"] != R3._sha(P["events"]):
        sys.exit("the event sample differs from the one the protocol registered")
    import torch
    from bonsai_interp import explore
    from bonsai_interp import atlas
    t0 = time.time()
    z = EV.load(P["events"])
    ids, cd, mp = z["ids"], z["count_dom"], int(z["min_pos"])
    toks = explore.corpus()
    _, _, alltok = CD.population(mp)
    at = np.asarray(alltok, np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = R13._embedding().to(dev, torch.float16)
    V = E.shape[0]
    cnt = np.bincount(toks.ravel().astype(np.int64), minlength=V)[:V]
    ok = torch.from_numpy(cnt >= SEL["class_min_count"]).to(dev)
    taus = np.round(np.arange(SEL["tau"][0], SEL["tau"][1] - 1e-9, -SEL["tau"][2]), 2)
    asg = R3._assigned()
    hyps, n_elig = [], 0
    for L in range(36):
        t = atlas.load(L)
        if t is None:
            continue
        d = t.select(["neuron", "sign", "class"]).to_pydict()
        for n, sg, cl in zip(d["neuron"], d["sign"], d["class"]):
            if cl != "eligible" or (L, n, sg) in asg:
                continue
            p = 0 if sg == "+" else 1
            e = ids[L, p, n]
            e = e[e >= 0]
            w, q = e // 1024, e % 1024
            if len(e) < SEL["min_events"] or len(np.unique(w)) < SEL["min_windows"]:
                continue
            n_elig += 1
            count = int(cd[L, p, n].sum())
            cur = toks[w, q].astype(np.int64)
            cands = []
            for kind, s in SHIFT.items():
                got = _set(toks[w, np.clip(q + s, 0, 1023)].astype(np.int64))
                if got:
                    S, cov = got
                    cands.append((kind, S, cov, cov * count / max(at[S].sum(), 1)))
            u, c = np.unique(cur, return_counts=True)
            if (c >= SEL["min_seen"]).any():
                m = c >= SEL["min_seen"]
                cen = (E[torch.from_numpy(u[m]).to(dev)].float() * torch.from_numpy(c[m]).float().to(dev)[:, None]).sum(0)
                cos = (E @ torch.nn.functional.normalize(cen, dim=0).half()).float()
                cos[~ok] = -1
                cos = cos.cpu().numpy()
                for tau in taus:
                    cls = cos >= tau
                    cov = float(cls[cur].mean())
                    if cov >= SEL["cov"]:
                        k = int(cls.sum())
                        if SEL["class_size"][0] <= k <= SEL["class_size"][1]:
                            S = np.nonzero(cls)[0]
                            cands.append(("class", S, cov, cov * count / max(at[S].sum(), 1)))
                        break
            cands = [x for x in cands if x[3] >= SEL["min_precision"]]
            if not cands:
                continue
            kind, S, cov, prec = max(cands, key=lambda x: x[2] * x[3])
            pt = {"kind": kind, "condition": kind, "tokens": [int(x) for x in S], "coverage": round(cov, 3),
                  "precision": round(float(prec), 3)}
            hyps.append({"neuron": f"bonsai1/L{L}/N{n}{sg}", "union_coverage": pt["coverage"], "events": int(len(e)),
                         "score": round(cov * float(prec), 4), "parts": [pt]})
        print(f"L{L}: {len(hyps):,} hypotheses so far, {n_elig:,} eligible poles, {time.time() - t0:.0f} s", flush=True)
    hyps.sort(key=lambda h: -h["score"])
    kept = hyps[::max(1, len(hyps) // a.max)][:a.max] if a.max else hyps[:SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "n_eligible": n_elig,
                                "minutes": round((time.time() - t0) / 60, 1)}])
    by = {k: sum(h["parts"][0]["kind"] == k for h in kept) for k in ("next", "prev", "class")}
    print(f"wrote {P['select']}: {len(hyps)} hypotheses over {n_elig} eligible poles, kept {len(kept)} {by}", flush=True)


def _offset_battery(items, split):
    """{key: pairs, "placebo:key": pairs} for next and prev covers: a read token whose neighbour is in S against the same
    read token whose neighbour is not."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    src = []
    for _, win, ids, M, Mq, can in R3._test_masks(split):
        nb = {}
        for kind, s in SHIFT.items():
            nb[kind] = np.full_like(ids, -1)
            if s > 0:
                nb[kind][:, :-s] = ids[:, s:]
            else:
                nb[kind][:, -s:] = ids[:, :s]
        src.append((win, ids, can, nb))
    out = {}
    for name, (kind, S) in sorted(items.items()):
        seed = [31, sum(map(ord, name)) % 100003, 1 if split == "test" else 0]
        rng = np.random.default_rng(seed + [7])
        Fk, Fw, Fp, Qk, Qw, Qp = [], [], [], [], [], []
        for win, ids, can, nb in src:
            on = np.isin(nb[kind], S)
            f = on & can
            q = ~on & (nb[kind] >= 0) & can & np.isin(ids, np.unique(ids[f]))
            fw, fp = np.nonzero(f)
            qw, qp = np.nonzero(q)
            if len(fw) > CAND_MAX:  # a common neighbour or read token: a seeded sample of the candidates
                k = np.sort(rng.choice(len(fw), CAND_MAX, replace=False))
                fw, fp = fw[k], fp[k]
            if len(qw) > CAND_MAX:
                k = np.sort(rng.choice(len(qw), CAND_MAX, replace=False))
                qw, qp = qw[k], qp[k]
            Fk += ids[fw, fp].tolist(); Fw += win[fw].tolist(); Fp += fp.tolist()  # noqa: E702
            Qk += ids[qw, qp].tolist(); Qw += win[qw].tolist(); Qp += qp.tolist()  # noqa: E702
        rng = np.random.default_rng(seed + [5])
        F = C._cap(rng, Fk, Fw, Fp, R3.CAP["conj"])
        Q = C._cap(rng, Qk, Qw, Qp, R3.CAP["conj"], ["same token"] * len(Qk))
        pairs = SK._pairs(F, Q, C.N_TEST, R3.PER_KEY["conj"]["test"], seed)
        if len(pairs) < C.N_TEST:
            pairs = SK._pairs(F, Q, C.N_TEST, 4 * R3.PER_KEY["conj"]["test"], seed)
        plac = C._placebo(Q, C.N_TEST, seed + [2], 4 * R3.PER_KEY["conj"]["test"])

        def snip(e):
            return SK.Snip(e.w, e.p, int(dom[e.w]), toks[e.w, e.p - C.SNIP + 1:e.p + 1])
        out[name] = [(snip(f), snip(q), st) for f, q, st in pairs]
        out["placebo:" + name] = [(snip(x), snip(y), st) for x, y, st in plac]
    return out


def _label(pt, tk, near):
    words = lambda T: ", ".join(repr(tk.decode([int(t)])) for t in T)
    if pt["kind"] == "next":
        return (f"compose (a cover by the next token): fires on a token followed by {words(pt['tokens'])}, not on the same "
                f"token followed by anything else (the snippet ends before the next token)")
    if pt["kind"] == "prev":
        return f"compose (a cover by the previous token): fires on any token after {words(pt['tokens'])}, not on the same token after another"
    return (f"compose (an embedding class): fires on a token of a class of {len(pt['tokens'])} tokens the input embedding "
            f"groups together (commonest: {words(near)}), not on other tokens of the same frequency")


def build(a):
    P, _ = R3._setup(a)
    from bonsai_interp import explore
    from bonsai_interp.sweeplib import run as R
    split = "select" if a.smoke else "test"
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    off = {R7._key(pt): (pt["kind"], pt["tokens"]) for h in sel for pt in h["parts"] if pt["kind"] in SHIFT}
    cls = {R7._key(pt): pt["tokens"] for h in sel for pt in h["parts"] if pt["kind"] == "class"}
    t0 = time.time()
    bat = {**(_offset_battery(off, split) if off else {}), **(R7._word_battery(cls, split) if cls else {})}
    print(f"batteries for {len(off)} next/prev and {len(cls)} class covers in {time.time() - t0:.0f} s", flush=True)
    tk = explore.tokenizer()
    cnt = np.bincount(explore.corpus().ravel().astype(np.int64))
    pts = {R7._key(pt): pt for h in sel for pt in h["parts"]}
    rows, tested = [], set()
    for key in sorted(set(off) | set(cls)):
        pt = pts[key]
        near = sorted(pt["tokens"], key=lambda t: -(cnt[t] if t < len(cnt) else 0))[:8] if pt["kind"] == "class" else None
        item = {"type": "items", "kind": R7.KIND, "condition": key, "label": _label(pt, tk, near)}
        pairs, plac = bat[key], bat["placebo:" + key]
        if len(pairs) < C.N_TEST:
            rows.append({**item, "class": "untestable", "why": f"only {len(pairs)} held-out pairs"})
            continue
        tested.add(key)
        rows.append({**item, "class": "tested", "fire": [R._text(f) for f, _, _ in pairs], "quiet": [R._text(q) for _, q, _ in pairs],
                     "quiet_kind": [st if isinstance(st, str) else "same token" for _, _, st in pairs],
                     "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
        if len(plac) >= C.N_TEST:
            rows.append({"type": "items", "kind": R7.KIND, "condition": "placebo:" + key, "label": "placebo", "class": "placebo",
                         "fire": [R._text(f) for f, _, _ in plac], "quiet": [R._text(q) for _, q, _ in plac],
                         "quiet_kind": ["placebo"] * len(plac), "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in plac]})
    n_poles = 0
    for h in sel:
        pt = h["parts"][0]
        k = R7._key(pt)
        if k not in tested:
            continue
        n_poles += 1
        rows.append({"type": "test", "neuron": h["neuron"], "kind": R7.KIND, "condition": k,
                     "select": {"part_kind": pt["kind"], "coverage": pt["coverage"], "precision": pt["precision"],
                                "items": len(pt["tokens"]), "union_coverage": h["union_coverage"], "parts": 1}})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": len(tested), "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "poles": n_poles, "tests": sum(r["type"] == "test" for r in rows),
              "by_kind": {k: sum(r["type"] == "test" and r["select"]["part_kind"] == k for r in rows) for k in ("next", "prev", "class")}}
    pre = {"protocol": R7.PROTOCOL, "protocol_commit": GR.registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def main():
    _install()
    R7.main()
    if R7.sys.argv[1:2] == ["protocol"]:
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update({"about": __doc__, "selection": SEL, "excluded": EXCLUDED,
                    "assignment": "a pole is assignment-grade when its cover held; its coverage is the cover's"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
