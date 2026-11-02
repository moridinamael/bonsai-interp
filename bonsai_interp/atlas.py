"""The atlas: one compact, typed table per layer with everything known about each of its 24,576 poles.

  python -m bonsai_interp.atlas build [--auto2 data/auto2] [--auto data/auto]   ->  data/atlas/bonsai1_L<L>.arrow
                                                                                  + data/atlas/vocab.arrow

A full interpretation of the model is 884,736 poles. Gzipped JSON or a file per pole does not scale to that, so the
atlas is an Arrow IPC file with zstd compression: columnar, typed and compressed, and read directly by pyarrow (and so
pandas), polars, DuckDB, Hugging Face datasets and Arrow JS in a browser. Parquet is the same columnar data, but the
reference machine's application-control policy blocks pyarrow's Parquet module, so a Parquet copy, if wanted for a
hosted viewer, is converted on CI. Words are stored as token ids. vocab.arrow maps an id to its text, so readers need
no tokenizer.

    import pyarrow.ipc as ipc; t = ipc.open_file('data/atlas/bonsai1_L4.arrow').read_all(); df = t.to_pandas()

One row per pole, keyed by `pole` = 2 * neuron + (1 for the - pole):
  layer, neuron, sign             int8, int16, "+" / "-"
  class                           stage 1 eligibility: eligible | weak | dense | positional
  strong                          strong activations (>= 0.3 of the pole max) in the selection windows (0..7999)
  hit_tok, hit_n, hit_precision,  the pole's top words by count: token id, strong count, P(strong | token),
  hit_lift, hit_coverage,         lift over the pole's base rate, share of the pole's strong activations,
  hit_trigger                     and whether stage 1 screened that word (lists, at most 5)
  bin, context, test, auroc       the auto-token@2 screen and auto test, when the layer has been screened
  lib_kind, lib_condition,        the pole's best test across the deterministic passes (atlas.LIB_SOURCES): sweeplib@1
  lib_test, lib_auroc, lib_held,  (copy, topic, lang, struct, number, predict), sweeplib@2 (twin), sweeplib@3 (grammar),
  lib_label, lib_coverage         sweeplib@4 (pos), sweeplib@5 (code),
                                  compose@1-2 (compose), compose@3+ (cover); a held one first, then the higher coverage. lib_held lists every
                                  kind:condition it held under, lib_label the claim (scrutiny 0: automatic, never a
                                  survey label), lib_coverage its share of the pole's strong activations when measured
  census_flags                    the weights-only census (docs/CENSUS.md): predict 1, frequency 2, entropy 4, steer 8
Scrutiny 0 throughout: corpus statistics and automatic tests, never a label. Labels and statuses from tested
contributions stay in graph/ (git), keyed the same way, and can be joined on (layer, neuron, sign).
Published in the `atlas` release; `bash scripts/fetch_data.sh atlas` downloads it.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

from bonsai_interp.sensitive import mask, mask_token

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "data" / "atlas"
N = 12288
TOP = 5


def _write(t, path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.OSFile(str(path), "wb") as f, ipc.new_file(f, t.schema, options=ipc.IpcWriteOptions(compression="zstd")) as w:
        w.write_table(t)


def _read(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.OSFile(str(path), "rb") as f:  # read into memory: no memory map, so the file is never held open
        return ipc.open_file(f).read_all()


def _layer_of(p):
    return int(p.name.split("_L")[1].split(".")[0])


def _schema():
    import pyarrow as pa
    lst = lambda t: pa.list_(t)
    return pa.schema([("pole", pa.int32()), ("layer", pa.int8()), ("neuron", pa.int16()), ("sign", pa.dictionary(pa.int8(), pa.string())),
                      ("class", pa.dictionary(pa.int8(), pa.string())), ("strong", pa.int32()),
                      ("hit_tok", lst(pa.int32())), ("hit_n", lst(pa.int32())), ("hit_precision", lst(pa.float32())),
                      ("hit_lift", lst(pa.float32())), ("hit_coverage", lst(pa.float32())), ("hit_trigger", lst(pa.bool_())),
                      ("bin", pa.dictionary(pa.int8(), pa.string())), ("context", pa.string()),
                      ("test", pa.dictionary(pa.int8(), pa.string())), ("auroc", pa.float32()),
                      ("lib_kind", pa.dictionary(pa.int8(), pa.string())), ("lib_condition", pa.dictionary(pa.int16(), pa.string())),
                      ("lib_test", pa.dictionary(pa.int8(), pa.string())), ("lib_auroc", pa.float32()), ("lib_held", lst(pa.string())),
                      ("lib_label", pa.string()), ("lib_coverage", pa.float32()), ("census_flags", pa.int16())])


LIB_SOURCES = {  # every deterministic pass whose per-pole results join the atlas: (file, kind of row)
    "sweeplib@1": ROOT / "data" / "sweeplib" / "bonsai1.sweeplib1.results.jsonl.gz",
    "sweeplib@2": ROOT / "data" / "sweeplib2" / "bonsai1.sweeplib2.results.jsonl.gz",
    "sweeplib@3": ROOT / "data" / "grammar" / "bonsai1.sweeplib3.results.jsonl.gz",
    "compose@1": ROOT / "data" / "compose" / "bonsai1.compose1.results.jsonl.gz",
    "compose@2": ROOT / "data" / "compose" / "bonsai1.compose2.results.jsonl.gz",
    "sweeplib@4": ROOT / "data" / "grammar" / "bonsai1.sweeplib4.results.jsonl.gz",
    "sweeplib@5": ROOT / "data" / "codesyn" / "bonsai1.sweeplib5.results.jsonl.gz",
    "compose@3": ROOT / "data" / "compose" / "bonsai1.compose3.results.jsonl.gz",
    "compose@4": ROOT / "data" / "compose" / "bonsai1.compose4.results.jsonl.gz",
    "compose@5": ROOT / "data" / "compose" / "bonsai1.compose5.results.jsonl.gz",
    "compose@6": ROOT / "data" / "compose" / "bonsai1.compose6.results.jsonl.gz",
    "compose@7": ROOT / "data" / "compose" / "bonsai1.compose7.results.jsonl.gz",
    "compose@8": ROOT / "data" / "compose" / "bonsai1.compose8.results.jsonl.gz",
    "compose@9": ROOT / "data" / "compose" / "bonsai1.compose9.results.jsonl.gz",
    "compose@10": ROOT / "data" / "compose" / "bonsai1.compose10.results.jsonl.gz",
    "compose@11": ROOT / "data" / "compose" / "bonsai1.compose11.results.jsonl.gz",
    "compose@12": ROOT / "data" / "compose" / "bonsai1.compose12.results.jsonl.gz",
    "compose@16": ROOT / "data" / "compose" / "bonsai1.compose16.results.jsonl.gz",
}
GRAMMAR_KIND = {"sweeplib@3": "grammar", "sweeplib@4": "pos", "sweeplib@5": "code"}  # passes on the grammar engine


def _cover(proto):
    """compose@3 and later rounds: the cover engine (round3.py), coverage from the selection file."""
    return proto.startswith("compose@") and int(proto.split("@")[1]) >= 3


def _compose3_cov(path):
    """{(pole, item key): coverage} from compose@3's selection file (beside its results). compose@7's union covers: each
    part's own coverage, and the union's for every part of a pole whose parts all held (its coverage file)."""
    from bonsai_interp.compose.round3 import _key
    from bonsai_interp.compose.round7 import _key as _part_key
    sel = Path(str(path).replace(".results.jsonl.gz", ".select.json.gz"))
    if not sel.exists():
        return {}
    with gzip.open(sel, "rt", encoding="utf-8") as f:
        hyps = json.loads(f.readline())["hypotheses"]
    out = {}
    for h in hyps:
        if "parts" in h:
            out.update({(h["neuron"], _part_key(pt)): pt["coverage"] for pt in h["parts"]})
        else:
            out[(h["neuron"], _key(h))] = h["coverage"]
    cp = Path(str(path).replace(".results.", ".coverage."))
    if any("parts" in h for h in hyps) and cp.exists():
        with gzip.open(cp, "rt", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                out.update({(c["neuron"], t["condition"]): c["union_coverage"] for t in c["tests"]})
    return out


def _lib_rows(proto, path):
    """(pole id, kind, condition, outcome, auroc, label, coverage) per non-placebo result of one pass."""
    from bonsai_interp.graph.protocols import v2_stats
    cov = {}
    cp = Path(str(path).replace(".results.", ".coverage."))
    if proto in GRAMMAR_KIND and cp.exists():  # the grammar engine's coverage: share of the pole's 32 strongest activations
        with gzip.open(cp, "rt", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                cov[(c["neuron"], c["condition"])] = (c.get("coverage") or {}).get("share")
    c3 = _compose3_cov(path) if _cover(proto) else {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("placebo"):
                continue
            if proto == "sweeplib@1":
                yield r["neuron"], r["kind"], r["condition"], r["outcome"], (v2_stats(r) or (None,))[0], r.get("label"), None
            elif proto == "sweeplib@2":
                yield r["neuron"], "twin", r["claim"], r["outcome"], r.get("auroc"), r.get("label"), r.get("coverage")
            elif _cover(proto):  # coverage: the hypothesis's share of the pole's sampled strong activations
                cond = r["condition"].split("|")[0]
                yield r["neuron"], "cover", cond, r["outcome"], (v2_stats(r) or (None,))[0], r.get("label"),                     c3.get((r["neuron"], r["condition"]))
            elif proto in GRAMMAR_KIND:
                yield (r["neuron"], GRAMMAR_KIND[proto], r["condition"], r["outcome"], (v2_stats(r) or (None,))[0], r.get("label"),
                       cov.get((r["neuron"], r["condition"])))
            else:
                yield r["target"], "compose", r["kind"], r["outcome"], (v2_stats(r) or (None,))[0], r.get("label"), r.get("coverage_A")


def sweeplib(sources=None):
    """{(layer, "N123+"): (kind, condition, outcome, auroc, [held kind:condition], label, coverage)} over every
    deterministic pass in LIB_SOURCES: the pole's best result. A held test comes first, then a higher coverage, then a
    higher AUROC. Labels are masked; placebo tests are left out. {} when there are no results."""
    best, held = {}, {}
    for proto, path in (sources or LIB_SOURCES).items():
        if not Path(path).exists():
            continue
        for nid, kind, cond, outcome, auroc, label, cov in _lib_rows(proto, path):
            L, key = int(nid.split("/L")[1].split("/")[0]), nid.split("/")[-1]
            cand = (kind, cond, outcome, auroc, mask(label), cov)
            if outcome == "held":
                held.setdefault((L, key), []).append(f"{kind}:{cond}")
            rank = lambda c: (c[2] == "held", c[5] if c[5] is not None else -1, c[3] or 0)
            old = best.get((L, key))
            if old is None or rank(cand) > rank(old):
                best[(L, key)] = cand
    return {k: (*v[:4], held.get(k, []), v[4], v[5]) for k, v in best.items()}


def census_flags(path=ROOT / "data" / "census" / "bonsai1.census.poles.npz"):
    """(36, 2, N) census flag bits (predict 1, frequency 2, entropy 4, steer 8), or None without the census."""
    import numpy as np
    return np.load(path)["flags"] if Path(path).exists() else None


def build_layer(hits_path, screen_path=None, out_dir=ATLAS, lib=None):
    import pyarrow as pa
    L = _layer_of(hits_path)
    lib = lib or {}
    screen = {}
    if screen_path and screen_path.exists():
        with gzip.open(screen_path, "rt", encoding="utf-8") as f:
            screen = json.load(f)["poles"]
    cols = {k: [] for k in _schema().names}
    with gzip.open(hits_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = r["neuron"].split("/")[-1]                     # "N123+"
            n, sign = int(key[1:-1]), key[-1]
            trig = {t["token"] for t in r.get("triggers", [])}
            w = (r.get("words") or [])[:TOP]
            s = screen.get(key, {})
            cols["pole"].append(2 * n + (sign == "-"))
            cols["layer"].append(L)
            cols["neuron"].append(n)
            cols["sign"].append(sign)
            cols["class"].append(r["class"])
            cols["strong"].append(r.get("strong"))
            cols["hit_tok"].append([int(x[0]) for x in w])
            cols["hit_n"].append([int(x[1]) for x in w])
            cols["hit_precision"].append([float(x[2]) for x in w])
            cols["hit_lift"].append([float(x[3]) for x in w])
            cols["hit_coverage"].append([float(x[4]) for x in w])
            cols["hit_trigger"].append([int(x[0]) in trig for x in w])
            cols["bin"].append(s.get("bin"))
            cols["context"].append(mask(s.get("context")))  # sensitive words never reach the public atlas
            cols["test"].append(s.get("test"))
            cols["auroc"].append(s.get("auroc"))
            lb = lib.get((L, key), (None, None, None, None, [], None, None))
            cols["lib_kind"].append(lb[0])
            cols["lib_condition"].append(lb[1])
            cols["lib_test"].append(lb[2])
            cols["lib_auroc"].append(None if lb[3] is None else float(lb[3]))
            cols["lib_held"].append(lb[4])
            cols["lib_label"].append(lb[5])
            cols["lib_coverage"].append(None if lb[6] is None else float(lb[6]))
            cols["census_flags"].append(None)
    schema = _schema()
    arrays = []
    for field in schema:
        a = pa.array(cols[field.name], type=field.type.value_type if pa.types.is_dictionary(field.type) else field.type)
        arrays.append(a.dictionary_encode() if pa.types.is_dictionary(field.type) else a)
    t = pa.Table.from_arrays(arrays, schema=schema).sort_by("pole")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bonsai1_L{L}.arrow"
    _write(t, out)
    return out, t


def build_vocab(ids, out_dir=ATLAS):
    """id -> text for every token the atlas mentions (readers need no tokenizer)."""
    import pyarrow as pa
    from bonsai_interp import explore
    from bonsai_interp.auto.context_tier import _bytes_to_unicode
    tok = explore.tokenizer()
    b2u = {u: b for b, u in _bytes_to_unicode().items()}

    def text(i):  # a piece of a multi-byte character decodes to U+FFFD; show its bytes instead (<0xE5>)
        s = tok.decode([i])
        if "�" not in s:
            return s
        piece = tok.convert_ids_to_tokens(i) or ""
        try:
            raw = bytes(b2u[c] for c in piece)
        except KeyError:
            return s
        out, k = [], 0
        while k < len(raw):
            for j in range(min(len(raw), k + 4), k, -1):
                try:
                    out.append(raw[k:j].decode("utf-8"))
                    k = j
                    break
                except UnicodeDecodeError:
                    continue
            else:
                out.append(f"<0x{raw[k]:02X}>")
                k += 1
        return "".join(out)

    ids = sorted(ids)
    t = pa.table({"tok": pa.array(ids, pa.int32()), "text": pa.array([mask_token(text(i)) for i in ids], pa.string())})
    out = out_dir / "vocab.arrow"
    _write(t, out)
    return out


def load(L, root=ROOT):
    """One layer's atlas as a pyarrow Table, or None when data/atlas has no file for it."""
    p = root / "data" / "atlas" / f"bonsai1_L{L}.arrow"
    return _read(p) if p.exists() else None


def vocab(root=ROOT):
    p = root / "data" / "atlas" / "vocab.arrow"
    if not p.exists():
        return {}
    t = _read(p)
    return dict(zip(t["tok"].to_pylist(), t["text"].to_pylist()))


def add_lib(src=ATLAS, out_dir=ATLAS):
    """Join the sweep library's columns onto existing atlas tables (no stage 1 inputs needed): every pole keeps its
    other columns, and lib_* are replaced by the current results."""
    import pyarrow as pa
    lib, cf = sweeplib(), census_flags()
    if not lib:
        sys.exit("no deterministic results in data/ (atlas.LIB_SOURCES)")
    schema = _schema()
    for path in sorted(Path(src).glob("bonsai1_L*.arrow"), key=_layer_of):
        t = _read(path)
        L = _layer_of(path)
        keys = [f"N{n}{s}" for n, s in zip(t["neuron"].to_pylist(), t["sign"].to_pylist())]
        lb = [lib.get((L, k), (None, None, None, None, [], None, None)) for k in keys]
        signs = t["sign"].to_pylist()
        neur = t["neuron"].to_pylist()
        cols = {"lib_kind": [x[0] for x in lb], "lib_condition": [x[1] for x in lb], "lib_test": [x[2] for x in lb],
                "lib_auroc": [None if x[3] is None else float(x[3]) for x in lb], "lib_held": [x[4] for x in lb],
                "lib_label": [x[5] for x in lb], "lib_coverage": [None if x[6] is None else float(x[6]) for x in lb],
                "census_flags": [None if cf is None else int(cf[L, 0 if sg == "+" else 1, n]) for n, sg in zip(neur, signs)]}
        arrays = []
        for field in schema:
            if field.name in cols:
                a = pa.array(cols[field.name], type=field.type.value_type if pa.types.is_dictionary(field.type) else field.type)
                arrays.append(a.dictionary_encode() if pa.types.is_dictionary(field.type) else a)
            else:
                arrays.append(t[field.name].combine_chunks())
        out = Path(out_dir) / path.name
        _write(pa.Table.from_arrays(arrays, schema=schema), out)
        n = sum(x[2] == "held" for x in lb)
        print(f"L{L}: {sum(x[0] is not None for x in lb)} poles with a library test, {n} held -> {out.name}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add-lib", help="join the sweep library's results onto the existing atlas tables in data/atlas")
    b = sub.add_parser("build", help="build the atlas from the auto-token@2 stage 1 tables and screen exports")
    b.add_argument("--auto2", default=str(ROOT / "data" / "auto2"), help="where bonsai1_L*.hits.jsonl.gz live")
    b.add_argument("--auto", default=str(ROOT / "data" / "auto"), help="where bonsai1_L*.screen.json.gz live")
    a = ap.parse_args()
    if a.cmd == "add-lib":
        return add_lib()
    files = sorted(Path(a.auto2).glob("bonsai1_L*.hits.jsonl.gz"), key=_layer_of)
    if not files:
        sys.exit(f"no stage 1 tables in {a.auto2}")
    ids, lib = set(), sweeplib()
    for f in files:
        L = _layer_of(f)
        out, t = build_layer(f, Path(a.auto) / f"bonsai1_L{L}.screen.json.gz", lib=lib)
        for row in t["hit_tok"].to_pylist():
            ids.update(row)
        screened = t["bin"].drop_null()
        print(f"L{L}: {t.num_rows} poles, {sum(1 for x in t['hit_tok'].to_pylist() if x)} with word hits, "
              f"{len(screened)} screened -> {out.relative_to(ROOT)} ({out.stat().st_size // 1024} KB)")
    v = build_vocab(ids)
    print(f"vocab: {len(ids)} tokens -> {v.relative_to(ROOT)} ({v.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
