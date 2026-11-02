# Composition: explaining a pole by the units that feed it

This is the recursive step of the deterministic harness (docs/HARNESS.md): a pole is characterized by
already-characterized units at the token offsets that drive its peaks. A round that succeeds adds new characterized
units, and the next round, one depth band up, can compose from them.

Code: `bonsai_interp/compose/`
- `feeders.py`: position-resolved attribution, ablation and token occlusion
- `pilot.py`, `pilot2.py`
- `match.py`: upstream units covering a token class
- `round.py`: the registered round

Protocol: compose@1 (graph/auto/bonsai1.compose1.*). Data: the `compose-data` release.

## Why tokens at offsets (pilot, FINDINGS 2026-10-26)

Zeroing a pole's 10 most attributed upstream neurons removes about half of a peak. Replacing a single token removes far
more:
- the read token: about 99%;
- the previous token: 47–87%, depending on depth;
- tokens 10–31 back: about 1%.

So a pole's peak is mostly a local function of its read token and the one before it. Each pole's top 32 peaks span
22–24 distinct read tokens, which makes them classes, not fixed bigrams. Attention enters as "which earlier positions
the pole needs", which occlusion measures directly.

## Round 1 (compose@1, layers 6–11)

**Hypotheses** (CPU, every residual eligible pole of layers 6–11):
- For offsets 0 and 1, take the tokens at the pole's 32 peaks and cover them with a greedy union of up to 3 upstream
  units. A unit qualifies when its focus is at least 0.3. Each layer's sparse token tables are matched against every
  target in one product.
- Units must already be **characterized**: a held survey label, or a held word-screen or sweep-library test. This is
  the sudoku rule.
- **pair:** both offsets covered on ≥ 60% of the peaks, and both together on ≥ 50%.
- **class:** the read token covered on ≥ 70% of the peaks, with at least 3 distinct tokens.

This gave 29,105 hypotheses (15,145 pair, 13,960 class); the best 6,000 were kept.

**Tests** (held-out windows 16,400 on, 8 + 8, quick-8+8@2 at the last token over 3 preambles):
- **pair:** fire when class A follows class B. The quiet half is A without B (the same read word); the other half is B
  without A (a read word of the same word class).
- **class:** fire on the class's words; quiet on the same word class outside it.
- **Classes:** A is the upstream units' trigger words that the target itself fires on (selection windows). B is the
  previous-offset units' trigger words.
- **Placebo:** every 10th hypothesis also gets an 8 + 8 test of quiet against quiet (199 had enough held-out items).

| | |
|---|---|
| Protocol | 6c3cc6c0 |
| Tests pre-registered | d03a6f39 |
| Tests built | 3,940 (2,060 hypotheses had too few held-out items) |

**Results:**

| | tests | held | held with coverage ≥ 50% |
|---|---:|---:|---:|
| pair | 1,690 | 405 (24%) | 270 |
| class | 2,250 | 101 (4.5%) | 64 |
| **all** | 3,940 | **506 poles** | **334** |
| placebo | 199 | **0** | |

- **By layer** (held): L6 263, L7 113, L8 90, L9 18, L10 13, L11 9.
- **Coverage** of held compositions (quartiles): 0.45 / 0.61 / 0.81. Coverage is the share of the target's strong
  activations on its class-A tokens. It is exact for class labels and an upper bound for pairs.

Examples (pole, coverage):

| pole | coverage | composition |
|---|---:|---|
| L8/N11199− | 0.72 | ` (` right after ` =` or ` return` (a call or tuple being built) |
| L6/N6274+ | 0.61 | ` day`, ` days`, `日` right after a digit |
| L6/N12084+ | 0.60 | ` le`, ` me`, ` les` right after Spanish ` se` |
| L6/N2951+ | 0.83 | ` of` right after ` ahead` |
| L7/N7774− | 0.58 | ` mean` right after `I` |
| L6/N10015+ | 0.72 | ` in` right after fill / filled / filling |
| L11/N6932− | 0.66 | ` raise` right after a code indent |
| L8/N1934+ | 0.53 | Arabic relative pronouns (التي, الذي) |

## Reading

- **The snowball turns, but slowly with depth.** Composing from known word units explains a quarter of the pair
  hypotheses it tests. 334 new assignment-grade poles are concentrated in layers 6–8, where the word units it builds
  on are densest.
- **Deeper layers need deeper units.** Hold rates fall from L9, where the inputs are less often word units. A second
  round, composing layers 12 and up from round 1's own holds, is the recursive test.
- **Class unions mostly fail** (4.5%): a union of a few word units rarely is the whole story of a pole.
- **The gaps are informative.** A hypothesis whose second offset has no covering unit (the "making verb" example in
  FINDINGS) names a unit kind to add.

Next: round 2 (layers 12–17) with round 1's holds as units, the attention distance sweep on held pairs, and per-round
coverage gain as the stopping rule.

## Round 2 (compose@2, layers 13–17): the snowball stalls

compose@2 (protocol 7eb795ff, tests 53f8b67c) ran compose@1's engine on layers 13–17 with two changes:
- **Units:** every pole characterized after round 1, from the survey, word screen, twins, grammar and compose@1.
- **Matching:** the upstream tables were filtered to those units before matching, which made selection 10–30× faster.

Layer 12 was held out of the word screen, so it has no token table and could be neither target nor source.

It produced 33,480 hypotheses; the best 6,000 gave 4,495 tests and 366 placebos.

| | tests | held | assignment-grade |
|---|---:|---:|---:|
| pair | 957 | 18 (1.9%) | |
| class | 3,538 | 4 (0.1%) | |
| **all** | 4,495 | **22 poles** | **8** |
| placebo | 366 | **0** | |

By layer: L13 8, L14 4, L15 4, L16 5, L17 1. The assignment-grade holds are mostly format and code compositions:
- `http` after punctuation;
- `%` after a quote or indent;
- `#` after a newline or indent;
- year / month after a number;
- `!` in any script;
- the impress / impression family across languages.

**Why it stalls.** In round 1 the composed condition fired its target on fresh text at a median 0.03 of max, and the
held quartile at 0.29. In round 2 the median is 0.001: the condition barely fires the pole at all. The near misses
stay quiet in both rounds.

So past layer ~11, the classes of a pole's read and previous tokens at its peaks are not what drives it. Those tokens
co-occur at the peaks without causing them. The occlusion pilot points the same way:
- the previous token's share falls (87% in layers 6–11, 78% in layers 18–23);
- tokens far back matter more (40% of layer 18–23 peaks have one, 10–31 tokens back, that removes at least a quarter
  of the peak).

**What the next round needs**:
1. **Offsets from each pole's own occlusion map**, not a fixed 0 and 1. That includes far context, with a distance
   sweep.
2. **Context units, not only word units:** language and script gates, topic, code slot, and the census's steering
   units.
3. **Composition over mid-layer units themselves**, layers 12 and up, read through attention, once those units have
   labels.

Round 2 is a registered negative result about the method, not about the model: local word-class composition reaches
about layer 11.

## Round 3 (compose@3, all layers): cover poles with conditions

Round 2 matched upstream *word* units at fixed offsets. Round 3 instead matches each pole against *conditions*: the
whole library of deterministic passes, evaluated on a uniform sample of the pole's own strong activations. A
hypothesis is selected only if it already covers at least half of the pole, so a hold is assignment-grade by
construction (the 50% rule).

Code:
- `bonsai_interp/events/sample.py`: the sample;
- `bonsai_interp/compose/conds.py`: the condition library;
- `bonsai_interp/compose/round3.py`: the round.

Protocol: compose@3 (graph/auto/bonsai1.compose3.*), with the sample pinned by hash. Data: the `compose3-data` release.

**The sample** (one GPU pass, 118 min, all 36 layers):
- **Windows:** the 13,593 selection-split windows that the grammar pass parsed (11,166 English) and sweeplib@5 lexed
  (2,427 code).
- **Per pole:** a bottom-k sample (k = 64, by a hash of window and position) of its activations at 0.3 of max or
  more, plus the count of all of them.
- **Coverage:** nearly every non-weak pole has 64 samples, so coverage can be measured over the selection split as a
  whole, not only the top 32 examples.

**The conditions (71):**
- the domain gates (English, code);
- 12 parts of speech and 8 morphological features (sweeplib@4);
- 20 grammatical roles (sweeplib@3), 6 dependency groups, and "in a question";
- 22 code syntax roles (sweeplib@5).

**Selection** (selection split; per eligible pole not already assignment-grade):
- **The pole needs** at least 32 sampled activations spread over at least 8 windows.
- **The condition c is chosen** among those with coverage ≥ 0.5 and lift ≥ 2 over c's base rate. It is the one with
  the highest **precision**: the share of c's positions where the pole is strong.
- **Tokens:** T is the pole's commonest tokens where c holds, at most 8, covering 80% of those activations. If c and T
  together still cover half the pole, the hypothesis is the conjunction "a token of T where c holds".
- **The conjunction must earn its place:** c must be at least 1.5× commoner at the pole's activations on T than on T
  overall. Otherwise it is just the word pole again.
- **Precision floor:** 0.1.

The precision floor and the within-token rule came from a pilot on the selection split. At 0.1 or more, 15 of 55 pilot
tests held; below it, 1 of 103.

**Tests** (held-out split):
- **Conjunction:** the same token with c on, against the same token with c off.
- **Class:** c whatever the token, against c's natural complement in the same frequency band. Examples: present
  against past verbs; the same word in another grammatical role; code tokens outside the code role.

Every test has a placebo (two c-off snippets) on the same pole.

### Results (2026-10-27)

| | |
|---|---|
| Protocol | 67c3f6e7 |
| Tests pre-registered | 7286b1f7 |
| Results | f71f5464 |

Selection gave 3,536 hypotheses. 877 item sets were untestable (too few c-off uses of the token in the test split),
which left 2,601 tests.

| | tests | held |
|---|---:|---:|
| conjunction (token of T where c) | 2,399 | 270 |
| class (c, any token) | 202 | 15 |
| **all** | **2,601** | **285 poles, all assignment-grade** |
| placebo | 2,518 | **0** |

- **Coverage and firing:** the median held coverage is 0.59. The median held fire is 0.34 of max, against 0.001 in
  round 2: the conditions drive these poles.
- **By family:** part of speech 106, code syntax 85, grammatical role 76, morphology 10, dependency 6, question 2.
- **By depth:** L0–5 133, L6–11 58, L12–17 5, L18–23 40, L24–29 35, L30–35 14.

**What holds:** word poles that fire for one sense or one syntactic use of their word. These are the "word detectors
that are more subtle" the survey suspected.
- **Homographs by part of speech:**
  - " bear", " warrant" and " license / permit" as nouns, not verbs;
  - " file", " deal" (L22) and " result" (L31) as verbs, not nouns;
  - " sign" as a verb (L22).
- **Grammatical role:**
  - " help" when negated ("can't help");
  - " wonder" after a determiner ("a wonder");
  - " care" as an object ("take care").
- **Code:**
  - " as" in an `import … as` (L0 to L23);
  - " is" and " or" in an if-condition;
  - ":" in a def signature (L24);
  - " for" in a comprehension (L6, L21, L22);
  - "from" in an import (L24).
- **Questions:** " you", " are" and " do" in a question (L0, L31).

### Reading

The word screen calls many poles "word detectors" whose word covers under half of them (partial). Round 3 shows what
part of the rest is: the same word in another sense or slot. A pole that fires on " bear" the animal but not on
" bear" the verb is a sense unit, and its coverage passes 50% once the sense is in the label.

These sense units exist at every depth, and most often early (L0–11). That fits disambiguation from the local
context, read through attention one or two tokens back.

Layers 12–17 again yield little (5 holds). That band stays the map's hardest.

## Round 4 (compose@4): format conditions and the tokens before

Round 3 left layers 12–17 almost empty. A diagnostic on its event sample showed why. At L14 and L16 only 7–14% of
poles have a read-token condition covering half of them, against 18–27% at other depths. Their strong activations
also spread over more read tokens: the top 8 cover 33%, against 41–47% elsewhere.

Round 4 keeps round 3's engine and event sample (pinned by the same hash) and widens the library to 222 conditions
(`bonsai_interp/compose/conds2.py`):
- **13 text-format conditions:**
  - line start, sentence start;
  - digit, after a digit;
  - capitalized, all capitals;
  - punctuation, after punctuation;
  - inside brackets or a quotation opened on the same line;
  - in a URL;
  - on a list line, on a `#`-led line;
- **every round 3 condition shifted** to the previous token and the one before it (`prev1:`, `prev2:`).

Poles that round 3 tested are left out, as is every assignment-grade pole. Re-testing them could repeat round 3's
pairs exactly.

| | |
|---|---|
| Protocol | 7cfebb38 |
| Tests pre-registered | c2d27487 |
| Results | 0f7e61dd |

Selection gave 2,499 hypotheses. 672 item sets were untestable, which left 1,755 tests.

| | tests | held |
|---|---:|---:|
| shifted one token (`prev1:`) | 886 | 180 |
| format (`fmt:`) | 738 | 126 |
| shifted two tokens (`prev2:`) | 131 | 14 |
| **all** | **1,755** | **320 poles, all assignment-grade** |
| placebo | 1,735 | **0** |

- **Coverage and firing:** median held coverage 0.63, median fire 0.32 of max.
- **By depth:** L0–5 166, L6–11 73, L12–17 10, L18–23 33, L24–29 23, L30–35 15.
- **What holds:**
  - words after a number ("of" in "3 of", L24), after a conjunction (" other" in "and other", L12) or after a
    possessive;
  - `(self` right after a def name (L16, L24);
  - list and dict continuations (`,\n`) inside brackets (L14, L15, L24);
  - single capital letters at line start (L35);
  - `#`-led lines. In code windows these are Python comments, mostly license headers: " governing", " language" and
    " details" on a `#` line.

### Reading

Format and the token before carry about as much as round 3's read-token roles did. The deterministic passes now cover
three kinds of pole: the word, the word in a sense or slot, and the word in a local frame. Layers 12–17 still yield
little (10). Every method that conditions on the read token or the one or two before it reaches about layer 11, then
picks up again past layer 17. Whatever the mid layers compute is not visible in the local text window at this
resolution.

## Round 5 (compose@5): far context

The same cover, with conditions from the context before the read token (`bonsai_interp/compose/conds3.py`, 386
conditions):
- **`win8:C`:** C held at one of the 8 tokens before;
- **`dense32:C`:** C held unusually often in the 32 tokens before, at or above the 90th-percentile count.

C is any of compose@4's read-token and format conditions. A diagnostic on the event sample motivated it. At layers
14–16 the conditions that best covered poles were of this kind (a pronoun in the last 8 tokens, past tense in the last
32), though none reached the precision floor on its own.

| | |
|---|---|
| Protocol | 538f7506 |
| Tests pre-registered | d64b1c3e |

Selection gave 1,050 hypotheses. 580 item sets were untestable: the same token rarely appears with the context both on
and off. That left 407 tests.

- **Held:** 22 of 407 (5%), against **0 of 405** placebo tests.
- **By family:** near context (`win8`) 12, dense context (`dense32`) 10.
- **By depth:** L0–11 14, L12–17 2, L18–35 6.
- **Examples:**
  - "..." within 8 tokens of a URL (L11, L13);
  - `def` right after a decorator (L8, L10);
  - `else`/`elif` within 8 tokens of a `return` (L7);
  - closing brackets in a passage dense with line starts (L0);
  - `except` among dense line starts (L19).

### Three negative results (selection split diagnostics, 2026-10-28)

1. **Two conditions together, any token.** Pairs of library conditions with no token set never reached the precision
   floor, at any depth (0 of about 2,000 sampled poles).
2. **Semantic classes.** WordNet supersenses at each word's first sense (noun.person, verb.motion, …) also never
   reached it (0 of about 4,000). A supersense occurs at 0.5–2% of positions, while a typical mid-layer pole is strong
   at about 0.01%. For a precision of 0.1, the condition would have to be about that rare.
3. **Far context:** as above, 2 holds in L12–17.

So a mid-layer pole is sparser than any class this library can name: grammatical, format, contextual or semantic.
Where the rounds succeed, a token set supplies the precision. In layers 12–17 the strong activations spread over too
many tokens for that.

## Round 6 (compose@6): anticipation

The same cover, with conditions on what comes **next** (`bonsai_interp/compose/conds4.py`, 360 conditions):
- **`next1:C`:** compose@3's condition C holds at the token right after the read token;
- **`next2:C`:** C holds at the token after that.

The test snippet ends at the read token, so the model never sees the next token. A pole that holds here anticipates
C from the context. The census found prediction-type poles rising with depth (34% in L30–35).

| | |
|---|---|
| Protocol | e676ff2d |
| Tests pre-registered | 0b9184a9 |
| Results | 8ea14011 |

Selection gave 1,391 hypotheses (755 anticipation, the rest compose@4 conditions on poles it had not tested). 671
item sets were untestable, which left 650 tests.

- **Held:** 37 of 650, against **0 of 646** placebo tests.
- **By family:** the next token 35, two tokens ahead 2.
- **By depth:** L0–11 11, L12–17 1, L18–35 25.
- **Phrasal-verb heads that expect their particle:** " carry", " make", " break", " play", " find", " figure" and
  " rule", before "out", "up", "down" or "off" (L10, L23, L28, L31, L32). These are the clearest group.
- **Other anticipations:**
  - " so" before an intensified adjective (L1, L31);
  - an `f` or `u` prefix right before a string literal (f-strings, L10, L13, L28);
  - " West" and " Western" before a proper noun (L25);
  - " for" before a loop variable (L21, L25);
  - " is" or " was" before a passive participle (L22).
- **Class-level anticipation:** the next word is a past-tense verb (L34/N589+) or a third-person singular verb
  (L35/N7247+).

### Reading

Anticipation poles are real and mostly deep: two thirds of the holds are in L18–35, where the census places
prediction-type poles. The phrasal-verb heads show the pattern well. A verb that takes a particle has a pole at
several depths that fires before the particle arrives. Layers 12–17 again yield 1 hold.

## Round 7 (compose@7): union covers

A label can now have two parts: "X where c holds, or Y" (`bonsai_interp/compose/round7.py`, conditions from
`conds4.py`).
- **A part** is a word set alone (the pole fires on these tokens, not on other tokens of the same frequency) or a
  compose@3 conjunction (these tokens where c holds, not the same tokens elsewhere).
- **A cover** is one part covering at least half of the pole's sampled strong activations, or two parts that do so
  together. The two parts have disjoint tokens and at most one is a word set; a single part is preferred when one
  reaches half.
- **Assignment:** each part is tested on its own (quick-8+8@2), and the pole is assignment-grade when every part held.
  Its coverage is the union's.

| | |
|---|---|
| Protocol | 79d94d22 |
| Pre-registration | 3dae0209 |
| Summary | e0133fad |

The queue held 43,874 poles that no earlier round had assigned or compose@3–6 had tested. 6,000 were selected (4,000
one-part, 2,000 two-part); 5,518 had every part testable.

- **Assigned: 3,663 of 5,518 poles**, against **0 of 7,267** placebo tests.
- **One-part covers: 3,629 of 3,740 assigned.** Word sets held 3,616 of 3,694; conjunctions 13 of 46.
- **Two-part covers: 34 of 1,778 assigned.** Their parts held 640 of 3,556 (word parts 477 of 1,297, conjunction
  parts 163 of 2,259).
- **By depth:** L0–5 3,284 (L0 alone 1,482), L6–11 127, L12–17 2, L18–35 250.
- **3,600 of the 3,663 had no earlier result from any deterministic pass.** They are token detectors that the
  condition-based rounds never reached, because a word set alone was not in their library.
- **Deep word poles** include 61 on the document boundary token, and poles on " United", " Santa", " grand", "://"
  and " absolutely".
- **Two-part examples:**
  - a spelling with and without its leading space (" ic" alone, and "ic" after punctuation);
  - " line" as a verb, or " lining";
  - " last" as a verb, or " duration";
  - " else" after a conjunction, or " otherwise" after a determiner.

### Reading

The union format itself adds little: two-part covers rarely hold in both parts (2%), so most poles that need two
parts to reach half their activations are not described by those two parts. The round's yield comes from the
one-part word test, which found 3,616 early-layer token detectors that no earlier pass had tested. Layers 12–17 again
yield almost nothing (2 assigned).

## Round 8 (compose@8): one-part covers, next slice

compose@7's engine, unchanged (`bonsai_interp/compose/round8.py`), testing only one-part covers, since two-part covers
held in both parts for 2% of poles. The next 8,000 one-part covers by the same ranking were selected, leaving out every
pole compose@3–7 tested; 7,215 were testable.

| | |
|---|---|
| Protocol | de573f8a |
| Pre-registration | 370f9844 |
| Summary | 9393067a |

- **Assigned: 6,353 of 7,215 poles**, against **0 of 7,207** placebo tests.
- **By part:** word sets held 6,298 of 7,002; conjunctions 55 of 213.
- **By depth:** L0–5 4,096, L6–11 585, L12–17 12, L18–23 638, L24–29 924, L30–35 98.
- **Deep word poles (L21–29, 1,710 assigned)** are mostly word-initial pieces with their capitalized forms (" dem"/" Dem",
  " rem"/" Rem", " exp"/" Exp", " sil"/" silicon") and whole words with an inflection (" enjoy"/" enjoyed",
  " consistent"/" consistently", " chronic"/" acute").
- **Conjunctions** are mostly a word after a word class: " article" or " basis" after a determiner, " like" after a
  noun, " again" after an adverb.

### Reading

The one-part word test keeps finding assignment-grade token detectors well down the ranking (88% held here, against
98% for compose@7's top slice). Layers 12–17 again yield almost nothing.

## Round 9 (compose@9): the rest of the one-part queue

compose@8's engine, unchanged (`bonsai_interp/compose/round9.py`), on every one-part cover left in the queue: 30,661
selected, 29,044 testable. Poles compose@3–8 tested were left out.

| | |
|---|---|
| Protocol | 8340340b |
| Pre-registration | 35884f15 |
| Summary | ae3e6de2 |

- **Assigned: 8,958 of 29,044 poles**, against **1 of 29,031** placebo tests (a next-token adjective condition on
  L5/N3898+).
- **By part:** word sets held 8,804 of 26,958; conjunctions 154 of 2,086.
- **By depth:** L0–5 3,668, L6–11 1,217, L12–17 61, L18–23 1,202, L24–29 2,368, L30–35 442.
- **The hold rate falls down the ranking:** 67% in the first 5,000, then 52%, 35%, 21%, 13% and 7% in the last 5,000.
- **Precision predicts it:**

| selection precision | held |
|---|---:|
| 0.1–0.2 | 11% (1,384 of 12,636) |
| 0.2–0.3 | 29% (1,790 of 6,216) |
| 0.3–0.5 | 51% (3,567 of 7,026) |
| ≥ 0.5 | 70% (2,217 of 3,166) |

### Reading

compose@7–9 assigned 18,974 poles, 18,718 of them by a single word set. The ranking worked: 98% held at the top of the queue,
88% in compose@8's slice, 31% here. A cover's precision on the selection split (how often the trigger fires the pole
when it occurs) is the best single predictor of whether its held-out 8 + 8 holds. The 0.1 floor admits many covers that
fail; the queue is now exhausted, so later rounds need new hypotheses, not a lower bar.

## Round 10 (compose@10): pair covers

A cover can now name the previous token as well as the read token (`bonsai_interp/compose/round10.py`): the pole fires
on these (previous token, token) pairs, at most 8 of them covering half of its sampled strong activations.
- **Selection:** precision of at least 0.3 (compose@9: holds track precision), and at least 1.5× the precision of the
  same read tokens alone, so the previous token matters.
- **Near misses** come from two strata: the same read token after another previous token, and the same previous token
  before another token.
- **Eligibility:** poles earlier rounds tested but did not assign were eligible, since a pair cover is a new hypothesis.

| | |
|---|---|
| Protocol | 0a978d89 |
| Pre-registration | 2010e6e3 |
| Summary | 083914fd |

13,128 poles had a pair cover and 12,000 were selected. 9,588 were testable.

- **Assigned: 7,023 of 9,588 poles (73%)**, against **0 of 9,586** placebo tests.
- **By depth:** L0–5 3,885, L6–11 2,679, L12–17 92, L18–23 154, L24–29 159, L30–35 54.
- **Precision predicts the hold again:** 46% at 0.3–0.5, 76% at 0.5–1, 88% above 1.
- **Rescues:** 2,505 of the assigned poles had failed an earlier cover round, and 4,518 had not been tested by one
  before.
- **Words split across tokens:** where every read token is a word-internal piece, 1,844 of 2,226 held (" sne|aky",
  " re|printed", " mon|ologue").
- **Other examples:**
  - " the| said", " Mr|.", " Ms|." (L12);
  - " York| Times", " Washington| Post" after a comma (L13);
  - " db|.Column", ".models|.fields" (L17);
  - " Args|:", " Raises|:" in docstrings (L22);
  - " a| set", " a| team", " group| of" (L23);
  - " the| very" (L33).

### Reading

The previous token turns many low-precision word covers into assignments. Layers 6–11, where the word test held less
often, gave 2,679 poles here. Layers 12–17 gave 92, more than any earlier round, but still under 1% of the band's
eligible poles. Many held covers are lists of pairs rather than one distinction: they are assignment-grade by coverage,
and a sharper label is a matter for the survey.

## Round 11 (compose@11): triple covers

compose@10's pair cover with one more token (`bonsai_interp/compose/round11.py`): the pole fires on these (token two
back, previous token, token) triples, at most 8 covering half of its sampled strong activations.
- **Selection:** precision of at least 0.3, and at least 1.5× the precision of the same pairs alone.
- **Near misses:** the same pair after another token two back, and the same two tokens before another token.

| | |
|---|---|
| Protocol | 31e8d492 |
| Pre-registration | 03dac4d5 |
| Summary | 6a42e537 |

1,837 poles had a triple cover; 1,609 were testable.

- **Assigned: 640 of 1,609 poles (40%)**, against **0 of 1,599** placebo tests.
- **By depth:** L0–5 163, L6–11 361, L12–17 26, L18–23 36, L24–29 37, L30–35 17.
- **By precision:** 26% held at 0.3–0.5, 45% at 0.5–1, 66% above 1.
- **Number and time patterns** were selected often and held less: 175 of 507.
- **Phrases held well:**
  - "insofar as" and "to the extent that" (L7);
  - "brought about" and "comes around" (L6);
  - "above all" and "first of all" (L4);
  - "in and out" (L6);
  - "even if" after a clause (L22);
  - license headers on `#` lines (L22, L23).

### Reading

The token two back adds little: 1,837 candidate poles against compose@10's 13,128, and a lower hold rate. The n-gram
covers have reached their limit at three tokens. Most of what they find in layers 6–11 are fixed phrases.

## Round 12 (compose@12): wider covers

Covers of 9–16 items (`bonsai_interp/compose/round12.py`), for poles that have no cover of 8 or fewer: read tokens
first (compose@7's word test), else (previous token, token) pairs (compose@10's pair test).
- **Overfitting guard:** every item must be seen at least twice in the pole's sample.
- **Selection:** precision of at least 0.3.

| | |
|---|---|
| Protocol | 18309234 |
| Pre-registration | 0ad280a6 |
| Summary | ebc000db |

- **Assigned: 672 of 1,195 poles (56%)**, against **0 of 1,195** placebo tests. Token sets held 469 of 844, pair
  sets 203 of 351.
- **By depth:** L0–11 556, L12–17 4, L18–35 112.
- **Some lists have a plain common thread:**
  - words ending in "ake" (" lake", " stake", " cake", " intake", L0);
  - economics terms (" capital", " demand", " supply", " exports", L0);
  - a string literal opening on a symbol (" '/", " '<", L22);
  - male first names (" Frank", " Stephen", " Leonard", L22);
  - "a" before an adjective (" a| good", " a| real", " a| strong", L0).
- **Others are lists only.**

### Reading

A wider cover finds a few hundred more poles, mostly in layers 0–11. Several lists name a distinction the survey can
sharpen (a rhyme, a field, first names). With compose@11 this ends the n-gram cover line. compose@7–12 assigned 27,309
poles, almost none of them in layers 12–17.

## Round 13 (compose@13): label precision

compose@7–12 tested a word cover ("fires on these tokens") against other tokens of the same frequency. That null is easy:
a pole that fires on a whole category passes it with a few of the category's commonest members. compose@13
(`bonsai_interp/compose/round13.py`) re-tests every pole assigned by a one-part word cover of compose@7, 8, 9 or 12.
- **Neighbour test:** the cover's tokens against their nearest tokens in the model's input embedding (8 per cover token,
  by cosine). Inflections and glued variants of a cover token are left out (the twin pass tested those). No activation
  is read to choose the neighbours.
- **Control:** the same 8 fire snippets against compose@7's frequency-matched null, on fresh pairs, so a neighbour
  failure is not resampling noise.
- **Verdicts:**
  - sharp: both tests held;
  - broad: the control held and the neighbour test failed;
  - unreplicated: the control failed.
- **Assignment does not change:** the verdict qualifies an assigned cover.

| | |
|---|---|
| Protocol | ae632819 |
| Pre-registration | f6c83738 |
| Summary | b87aa859 |

- **19,149 poles tested; 2 of 19,116 placebo tests held.**
- **Sharp 14,829, broad 1,096, unreplicated 3,224.** Of the 15,925 covers that replicated, 93% are sharp.
- **The broad rate rises as the neighbours get closer:**

  | Mean neighbour cosine | < 0.10 | 0.10–0.13 | 0.13–0.16 | ≥ 0.16 |
  |---|---|---|---|---|
  | Broad, of replicated covers | 3.9% | 8.6% | 30% | 40% |

  Most covers have only distant neighbours (cosine under 0.10), so "sharp" is a strong claim only where the
  neighbours are close.
- **Broad covers are categories the list sampled:**
  - " I", " we" also fire on " you" and " they" (L24);
  - " likely", " probably" also fire on " usually" and " certainly" (L6);
  - " nervous", " nerve", " neural" also fire on " brain" and " muscle" (L22);
  - " bison", " Montana", " elk", " Yellowstone" also fire on other US states (L6);
  - "-five", "-four", "teen" also fire on other number words (L1).
- **By depth,** broad of replicated: L0–5 6.3%, L6–11 5.5%, L18–23 10%, L24–29 6.9%, L30–35 13%.
- **Replication tracks selection precision:**

  | Selection precision | ≥ 0.5 | 0.3–0.5 | 0.2–0.3 | < 0.2 |
  |---|---|---|---|---|
  | Unreplicated | 4.9% | 27% | 46% | 61% |

  By source: compose@7 0.7%, compose@8 4.1%, compose@9 32%, compose@12 27%. In 2,646 of the 3,224 failures the cover
  tokens' median fire is under 0.20: at low precision most occurrences of the tokens leave the pole quiet, and a single
  8 + 8 test holds when its draw happens to catch the ones that fire.

### Reading

Where a word cover replicates, it names the tokens and not the category around them in 93% of poles. That is the clean,
one-job reading at the token level, with a caveat: the model's 1-bit embedding puts most cover tokens far from their
nearest neighbours. The broad covers are where a category label ("first- and second-person pronouns", "hedges",
"body and nervous system") should replace the list; the survey can write those. The larger finding is replication:
an assignment from one 8 + 8 test at selection precision under 0.5 is fragile. About a third of compose@9's
assignments did not replicate on fresh pairs. The automatic tier's assignment-grade count should carry that, and the
pair and triple covers need the same check.

## Round 14 (compose@14): replication of the pair and triple covers

compose@13's fresh-pairs control, for every pole assigned by a pair or triple cover (compose@10, compose@11, and
compose@12's pair sets; `bonsai_interp/compose/round14.py`). Each cover is tested again with its own round's test and
strata, on pairs drawn with a new seed, with a placebo.
- **Verdicts:** replicated (held) or unreplicated (failed).
- **Assignment does not change.**

| | |
|---|---|
| Protocol | 6baa3211 |
| Pre-registration | 2fcd8eb1 |
| Summary | ed1e4661 |

- **6,941 of 7,818 covers replicated (89%);** 0 of 7,810 placebo tests held.
- **By kind:** pair covers 9.8% unreplicated, triple covers 27%.
- **By selection precision:** 7.6% unreplicated at ≥ 0.7, 15% at 0.5–0.7, 25% at 0.3–0.5.
- **By depth:** L0–5 8.5%, L6–11 12%, L12–35 24–31%.
- **How they fail:** in 617 of the 877 failures the pole still fires (median fire ≥ 0.20). A near miss fires too: the
  same token after another previous token, or the same previous token before another. The cover names the
  pole's commonest contexts but not a boundary the pole keeps.

### Reading

The pair covers mostly replicate, better than compose@9's word covers did, since a pair cover's selection precision is
higher (≥ 0.3 by rule, ≥ 0.7 for most). Together compose@13 and @14 re-tested 26,967 automatic assignments; 4,101
(15%) failed a fresh test. Low precision predicts it in both rounds, and so does depth.

## Round 15 (compose@15): replication of the screen assignments

compose@13 and @14's replication, for the largest set of automatic assignments they left: the 7,960 poles assigned by
the auto-token@2 screen (atlas test held, a trigger covering ≥ 0.5 of the strong activations) and by nothing compose@13
or @14 re-tested (`bonsai_interp/compose/round15.py`).
- **Test:** each pole's held auto test, drawn again by the same construction on the same testing windows, with a new seed.
  Every window the original test used is left out; the original draw was rebuilt from its seed and matched the original
  tests file for all 7,960 poles.
- **Judge:** the auto test's own (quick-8+8@2, last token, three preambles), with a placebo of fire snippets against fire
  snippets.
- **Verdicts:** replicated (held) or unreplicated (failed). 408 poles had too few unused snippets (untestable).
- **Assignment does not change.**

| | |
|---|---|
| Protocol | 37bbacc7 |
| Pre-registration | 293ff983 |
| Summary | 5454ab6b |

- **6,704 of 7,552 replicated (89%);** 2 of 6,881 placebo tests held.
- **By bin, the label's kind decides it:**

  | Bin | lexical | context | copy | predictive | format | collocation |
  |---|---|---|---|---|---|---|
  | Poles | 4,269 | 1,970 | 52 | 750 | 240 | 271 |
  | Unreplicated | 1.3% | 18% | 25% | 31% | 35% | 39% |

- **By trigger precision:** 2.3% unreplicated at ≥ 0.7, 19% at 0.5–0.7, 35% at 0.3–0.5, 17% under 0.3.
- **By depth:** L0–5 7.1%; every later band 19–24%.
- **How they fail:** in 755 of the 848 failures the pole still fires on the trigger (median fire ≥ 0.20). The near
  misses fire too: the trigger where the rule fails, or a swapped word. The trigger is right; the context rule is
  narrower than the pole.

### Reading

A lexical screen label ("the token X") replicates almost always (98.7%). A rule label ("X after Y", "X before a
subword", "X at a line start") fails a fresh draw 18–39% of the time, and mostly by being too narrow: the pole fires on X
outside the rule as well. With compose@13 and @14, 34,519 automatic assignments have now been re-tested, and 4,949 (14%)
failed a fresh test. About 3,100 assignments are still untested this way (the twin pass, compose@3–6's condition covers,
compose@1–2, and the 408 untestable here).

## Round 16 (compose@16): covers by the neighbouring token and by embedding class

compose@12 ended the n-gram cover line, and the poles it left fire on many different read tokens (the commonest covers
about 15% of a pole's sample). This round lets the read token vary and fixes something else
(`bonsai_interp/compose/round16.py`):
- **next:** the token after the read token is in a set of at most 8. The snippet ends before that token, so the pole
  has to expect it.
- **prev:** the token before the read token is in a set of at most 8 (any token after it).
- **class:** the read token is in a class of 9–256 tokens that the model's input embedding groups together (the tokens
  nearest the centroid of the pole's commonest read tokens), a category rather than a list.

Each eligible pole not yet assignment-grade (753,574) kept its best hypothesis with coverage ≥ 0.5 and precision ≥ 0.2.
- **next and prev tests:** the same read token, with the neighbour in the set against not.
- **class tests:** compose@7's word test (a class token against other tokens of the same frequency and domain).

| | |
|---|---|
| Protocol | a71dc2f3 |
| Pre-registration | 3b7fc105 |
| Summary | 1c444cc7 |

- **4,390 poles had a hypothesis**, 0.6% of the eligible poles. 4,028 were testable.
- **Assigned: 785 of 4,028 (19%);** 0 of 4,026 placebo tests held.
- **By kind:**

  | Kind | class | next | prev |
  |---|---|---|---|
  | Tested | 1,945 | 1,576 | 507 |
  | Held | 593 (30%) | 139 (9%) | 53 (10%) |

- **By precision, classes:** 19% held at 0.2–0.3, 34% at 0.3–0.5, 51% at 0.5–0.7, 65% at ≥ 0.7. Next covers: 3%, 6%,
  20%, 40%.
- **By depth:** L0–5 309, L6–11 60, L12–17 8, L18–23 111, L24–29 158, L30–35 139.
- **Held classes name categories:**
  - web formats (" URL", " HTML", " API", " HTTP", " JSON", L0);
  - Russia and its neighbours (" Russian", " Soviet", " Ukraine", L0);
  - titles (" director", " professor", " Minister", L0);
  - keys and values in code (" value", " key", " table", " index", L3).
- **Held next covers are mostly late-layer code poles** that expect an indent, a closing brace or `.append`. Held prev
  covers include "any token after ' Sir'" (L7).
- **How they fail:** in most failures the pole is quiet on the fire snippets (median fire under 0.20 in 1,123 of 1,352
  class, 1,327 of 1,437 next and 416 of 454 prev failures). The neighbouring token goes with the pole's strong
  activations but does not bring them on.

### Reading

The assignment-grade total rises by 785, to 38,414 (4.34% of poles). Embedding classes are the only one of the three
kinds that works often, and they work when precise. The neighbouring token mostly co-occurs with the pole without
causing it. What remains (about 830,000 poles, most of them firing on many tokens in many contexts) is beyond what a
fixed token or condition library can describe at precision 0.2.
