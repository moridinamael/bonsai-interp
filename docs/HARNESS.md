# Toward a deterministic characterization harness

The goal is a harness that takes a new model and runs until every MLP pole has either a tested label or a recorded
reason why none fits, with no language model in the loop. Search and steering then run on the result. This document
covers three things:
- what the passes share today;
- how each pass should seed the next;
- the kinds to add, including ideas absorbed from similar projects and from the Twitch loop.

## What passes share today (2026-10-25)

| from → to | shared? | how |
|---|---|---|
| scan → word screen (auto-token@2) | yes | the word screen's stage 1 reads the scan's pole max and its dense, positional and weak flags |
| word screen stage 1 → stage 2 → tests | yes | word hits seed the context screen, and the fitted rule seeds the test |
| word screen → sweep library (sweeplib@1) | **no** | the library screened every pole from scratch. It only uses the word screen at display time (a library result colours a pole the screen left unexplained) |
| scan + weights → predict | yes | top examples and the direct logit effect select the candidates |
| sweep library → word screen (a second round) | **no** | nothing has been re-run with what the library learned |
| survey labels (people, Twitch) → screens | **by hand only** | the copy battery was rewritten after reading the zoo's copy labels, and the validator checks recall against held labels |
| screens → Twitch agent | partly | `play card` showed the word screen. With #964 it shows the library and adds `play search`. The stream's own loop does not read either yet |

So the passes are mostly side by side, not stacked.

## The loop: each pass seeds the next

```
  weights census ──► scan ──► word screen ──► twin batteries ──► kind batteries ──► composition ──► residual mining ─┐
     (no text)       (top-k,   (word hits,     (seeded by each     (pole-agnostic:    (labels flow     (features shared │
                     density)   context rule)   word hit: case,    copy, lang,        along weight     by the top        │
                                                split, slot,       number, syntax,    links: B reads   examples of the   │
                                                sense, tense)      genre, predict)    what A writes)   still-unexplained)│
     ▲                                                                                                                  │
     └──────────── round 2: new conditions and kinds, re-registered as a new protocol version ◄────────────────────────┘
```

Rules that keep it deterministic and honest:
- **Pre-registration per round.** Each round is a protocol version. Its selection rule and tests are committed before
  anything is read on the test split.
- **Placebos everywhere.** Every condition has a quiet-vs-quiet placebo, and every pole's test has a placebo test. In
  sweeplib@1 placebos selected 0 poles and held 0 of 333 tests, so the rules can afford to be loose at selection.
- **A pole leaves the residual** when a test of it holds. Later rounds spend their GPU time on what is left, but a held
  pole can still be retested by a sharper kind.
- **Coverage is the stopping rule.** Each round reports the share of non-weak poles with a held label, by layer. The
  harness stops when a round adds less than a set amount of coverage.

## The kinds to add, in order of expected yield

1. **Twin batteries seeded by word hits.**
   - What it is: the Twitch playbook's most-used method (the "one-swap twin", 69 uses) made deterministic. For each held
     or high-precision word hit, generate the same word across fixed axes, then let the test decide which axis the
     pole cares about:
     - case and leading space;
     - whole token vs split piece;
     - sentence-initial vs mid-sentence;
     - code slot (string, comment, docstring, identifier, prose);
     - inflection family;
     - sense (when a parser can tell senses apart).
   - Why first: it leverages pass 1 directly. In a random 50 of the Twitch loop's 339 held labels, about 20 were "the word X
     in any case and sense" or "only where the letters form their own piece", and these axes decide exactly that.
2. **A weights-only census** (no corpus), run first on a new model:
   - direct logit effect (predict and suppression sets, as sweeplib@1's predict);
   - entropy neurons: a large output norm in the unembedding's effective null space;
   - token-frequency neurons: output aligned with the unigram log-frequency direction;
   - a steer lens: push each neuron ± and record the next-token Δlog-prob and KL on a few contexts;
   - a weight connection matrix between layers, P[B, A] = (W_gate_B·v_A)(W_up_B·v_A), giving each pole's strongest
     direct readers and writers.

   This is minutes of GPU time per layer. It gives every pole an output-side hypothesis before any text is read.
3. **A syntax kind**, from a deterministic parser (spaCy part of speech and dependency labels). It covers conditions
   such as "the same word in a different grammatical role": `so` as purpose vs degree, the particle of a separated
   phrasal verb, the predicate after a dummy "it". In a random 50 of the stream's held labels, 8 were of this shape.
4. **Genre and boilerplate kinds**, from regex document classifiers: licence headers, citations (scripture, legal,
   academic), chemical formulas, standards codes, and code languages. The stream found several such poles by hand.
5. **Lookahead by class.** Extend predict from token sets to classes: "the next token is a digit, a year, a unit, a
   closing bracket". Build the tests from the model's own next-token class probability, as predict does.
6. **Composition.** For a still-unexplained pole, take the labels of its strongest weight-linked upstream poles.
   Propose the conjunction or difference of their conditions (for example "German" and "number" give "numbers in
   German text"), and test it with the batteries that already exist. This is how a second full cycle learns what the
   first could not.
7. **Residual mining.** For poles still unexplained, run the feature extractor (word class, script, language, line
   kind, code state, parser labels, upstream labels) over their top and middle examples against matched controls. Turn
   the best-separating conjunction into a candidate condition for the next round. auto-token@2's context screen does
   this for one word at a time; residual mining does it for the pole.

## What to absorb from similar work

- **lyra's neuron-explorer** ([github.com/lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
  Qwen3.5-4B). It characterizes the *output* side: every neuron is pushed in steered rollouts and ranked against
  random-direction controls. It also has a steer lens covering all layers, at 111 s per layer. Lessons we can take:
  - Doses in units of the residual norm, calibrated per layer (their first layers broke at a dose that was gentle
    elsewhere).
  - A random-direction null, the counterpart of our placebo.
  - Weight-only connection links, precomputed once.
  - A depth story to compare against ours: topics early, discourse in the middle, lexical families late.
  - Steering "recipes" (for example `L19N7839−0.7 and L17N9209+0.5`) as the user-facing way to influence behaviour.
    That is what search over a characterized atlas should feed.
- **Universal neurons** (Gurnee et al., [arXiv 2401.12181](https://arxiv.org/pdf/2401.12181)) and **confidence
  regulation neurons** (Stolfo et al., [arXiv 2406.16254](https://arxiv.org/pdf/2406.16254)). These give a taxonomy of
  neuron families with weight-based signatures: prediction, suppression, entropy, token frequency, attention
  deactivation, position, previous token. Each family is a census kind.
- **LLM-written explanations** (Bills et al.; [openai/automated-interpretability](https://github.com/openai/automated-interpretability),
  archived 2026; [Neuronpedia](https://www.neuronpedia.org/)). These are the non-deterministic baseline the harness
  replaces. Their simulate-and-score idea maps onto our held-out 8 + 8.

## From atlas to search and steering

Once most poles have a held condition, the atlas is an index in both directions:
- **condition → poles.** `play search --kind lang --condition lang/de --held` lists German-text gates.
- **pole → conditions.** `play card` shows a pole's held conditions.

Steering a behaviour then means choosing the poles whose held conditions name it and pushing them with calibrated
doses. A steer-lens check confirms what the push writes before the push is used.

## For a new model

The harness needs four adapters:
1. an MLP hook that records and can replace activations;
2. the down-projection columns, the final norm and the unembedding;
3. a tokenizer with byte-level decoding;
4. a corpus with held-out splits.

Everything else, from the census through the scan, screens, batteries, placebos, judging and atlas, is
model-agnostic. Costs scale with depth by the measured model: ms per scored sequence = 0.309 + 0.713·(L+1). The
reasons the 8B sweep was slow are recorded in docs/SWEEP_LEARNINGS.md, so a re-sweep can avoid them.

## Composition and attention (pilot, 2026-10-26; FINDINGS same date)

Composition works on **tokens at offsets**, not on neurons. Zeroing a pole's top 10 attributed neurons removes only about
half of a peak, but replacing the read token removes almost all of it and the previous token most of the rest. So the
compositional pass runs in four steps:
1. **Occlusion map** (`compose/feeders.occlude`). Which offsets a pole needs, attention included, without modelling heads.
2. **Class at each offset** (`compose/match.cover`). The union of characterized upstream word units covering the tokens
   seen there. An uncovered class (no "making verb" unit) is a gap that says which kind to add next.
3. **Factorial test.** Both classes present, one only, or neither, at the measured offsets. Then a **distance sweep**
   moves the earlier class to 1, 2, 4, 8, 16 or 32 back. A response at 1 only means a previous-token or bigram path;
   a decaying response means a phrase; a flat response means copying or topic.
4. **Heads**, only for confirmed links: knock out the source→target attention edge head by head to find the carrier,
   and type the head (previous-token, induction, duplicate-token, sink).

The composed label is then "A here, given B one token back" or "A anywhere after B". It is tested like any other label:
placebos, a held-out 8 + 8, and the 50% coverage rule.

## Status after round 1 (2026-10-26)

| pass | what it explains | result |
|---|---|---|
| census (weights only) | the output side of every pole | 12% prediction-type, 10% strong steerers, 550 frequency, 0 entropy |
| word screen (auto-token@2) | word-level triggers | 35 layers; 68,295 poles with a held test, most covering under half of the pole |
| sweeplib@1 | copy, number, language/script, predict | 225 held |
| sweeplib@2 twins (seeded by the word screen) | which part of a word is the trigger | 20,164 held on 10,715 poles; 1,378 partial word poles reach assignment |
| sweeplib@3 grammar | generic roles at the last token | 30 held; roles mark sub-triggers, not assignments |
| compose@1 (seeded by characterized units) | layers 6–11 as compositions at offsets 0 and 1 | 506 held, 334 assignment-grade |
| compose@2 | layers 13–17, the same engine | 22 held, 8 assignment-grade: local word composition stalls past ~L11 |
| sweeplib@4 part of speech, morphology (docs/POS.md) | word classes and inflections at the last token | 26 held, 11 assignment-grade; morphology (-est, -er, -ing, -s), early layers; no pure noun or verb poles |
| sweeplib@5 code syntax (docs/CODESYN.md) | Python syntax roles at the last token | 65 held on 54 poles, 20 assignment-grade; binding sites (class, def, for-loop and keyword names), every depth |
| compose@3 (docs/COMPOSE.md) | each pole against the whole condition library on a sample of its strong activations | 285 held, all assignment-grade; word poles for one sense or slot of their word (bear as a noun, as in an import) |
| compose@4 (docs/COMPOSE.md) | the same cover with format conditions and the conditions one and two tokens back | 320 held, all assignment-grade; words in a local frame (after a number, after a conjunction, `(self` after a def name) |
| compose@5 (docs/COMPOSE.md) | the same cover with far-context conditions (a condition in the last 8 tokens, or dense in the last 32) | 22 held of 407; far context adds little; pairs of conditions and WordNet supersenses never reach precision |
| compose@6 (docs/COMPOSE.md) | the same cover with anticipation conditions (a condition at the next token or the one after) | 37 held of 650; phrasal-verb heads that expect their particle, f-string prefixes; mostly deep |
| compose@7 (docs/COMPOSE.md) | union covers: one part, or two parts with distinct triggers, each part tested | 3,663 of 5,518 poles assigned; 3,616 are one-part word sets, mostly L0–5 token detectors no pass had tested |
| compose@8 (docs/COMPOSE.md) | compose@7's covers, one-part only, on the next 8,000 poles of the queue | 6,353 of 7,215 assigned; word sets in L0–5 and word-initial pieces in L21–29 |
| compose@9 (docs/COMPOSE.md) | compose@8 on every remaining one-part cover | 8,958 of 29,044 assigned; the hold rate tracks selection precision (11% at 0.1–0.2, 70% at ≥ 0.5) |
| compose@10 (docs/COMPOSE.md) | pair covers: (previous token, token) pairs against the same token after another token and the same previous token before another | 7,023 of 9,588 assigned (73%); multi-token words, L6–11 strongest after L0–5; 92 in L12–17 |
| compose@11 (docs/COMPOSE.md) | triple covers: (token two back, previous token, token) | 640 of 1,609 assigned (40%); fixed phrases ("insofar as", "above all"); the n-gram covers end here |
| compose@12 (docs/COMPOSE.md) | wider covers: 9–16 read tokens or pairs, each seen twice | 672 of 1,195 assigned (56%); rhymes, fields, first names; the end of the cover line |
| compose@13 (docs/COMPOSE.md) | label precision: each assigned word cover against its nearest embedding tokens, with a fresh-pairs control | 19,149 poles: 14,829 sharp, 1,096 broad (categories), 3,224 unreplicated; replication falls with selection precision (4.9% fail at ≥ 0.5, 61% under 0.2) |
| compose@14 (docs/COMPOSE.md) | replication of the pair and triple covers on fresh pairs | 6,941 of 7,818 replicated (89%); pairs 9.8% fail, triples 27%; with compose@13, 4,101 of 26,967 automatic assignments failed a fresh test |
| compose@15 (docs/COMPOSE.md) | replication of the auto-token@2 screen assignments on fresh draws | 6,704 of 7,552 replicated (89%); lexical labels 1.3% fail, rule labels 18–39%; with compose@13–14, 4,949 of 34,519 automatic assignments failed a fresh test |
| compose@16 (docs/COMPOSE.md) | covers by the next or previous token, and by input-embedding class | 785 of 4,028 assigned (19%): classes 30%, next 9%, prev 10%; 0 of 4,026 placebo tests held |

All of it is in the atlas: lib_* columns (the best test per pole across passes, its coverage, every condition held) and
census_flags. It shows on `play card`, `play search --kind …` and the map.

Next:
- **Layers 12–17:** every method conditioned on the read token or the one or two before it stalls here (compose@3 5
  holds, compose@4 10). Next: far context from each pole's occlusion map, and composition over the mid-layer units
  themselves.
- **The attention distance sweep** on held pairs.
- **The residual count** per round as the stopping rule.
