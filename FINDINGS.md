> Note (2026-10-18): the package was renamed from `mono` to `bonsai_interp`. Commands and paths below use the new name; commit messages and pre-registration files keep the old one.

# Findings log

## 2026-10-17: Neuron identity survives quantization-aware training (mid layers)

**Setup** (`python -m bonsai_interp.identity --windows 12`)
- 12,288 tokens from the corpus tail, run through all three models.
- Neuron = SwiGLU product channel.
- Metric 1: same-index Pearson r.
- Metric 2: share of neurons whose best-correlated neuron in the other model is the same index.

| pair | L1 | L5 | L9 | L13 | L17 | L21 | L25 | L29 | L33 | L35 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B ~ Ternary, best match = same idx | 43% | 99.1% | 99.9% | 99.8% | 99.4% | 99.6% | 99.5% | 98.7% | 97.6% | 78.5% |
| Qwen3-8B ~ 1-bit | 40% | 94.6% | 98.7% | 97.2% | 95.3% | 94.7% | 92.0% | 84.2% | 80.7% | 47.2% |
| Ternary ~ 1-bit | 43% | 99.7% | 99.9% | 99.7% | 98.3% | 99.4% | 99.2% | 96.6% | 96.8% | 82.5% |

**Reading the table**
- Bonsai's quantization-aware training (QAT) keeps the base model's neuron basis. In layers 5–33, the same index is the same neuron, so the precision ladder is a paired comparison.
- Same-index Pearson r is only moderate (median 0.25–0.54). It is dominated by the bulk of weak activations, and the tail metric is the one that matters for the essay's audit; it is still to be measured.
- Identity erodes in late layers, most at 1 bit. At L35, 1-bit vs FP gives 47%. This is where the essay's "say vs think" / token-suppression neurons live.
- L1 matches poorly at every precision pair. This may be an artifact: early neurons are token-specific and sparse, and 12k tokens may be too few. Re-check with more tokens and a tail-overlap metric.

**Perplexity** on a 1,024-token essay passage: Qwen3-8B 28.7, Ternary 32.7, 1-bit 36.7.

## 2026-10-17: Solver behaviour on a toy SwiGLU neuron (8 inputs)

The toy is `a = silu(w_g·x)·(w_u·x)`, with x ∈ [-1,1]⁸. All solvers ran against the same `InverseProblem`.

| goal | globalMOO (after 181-pt DOE) | differential_evolution | least_squares + autograd |
|---|---|---|---|
| a ≈ 1.0 ±1% (inside DOE range) | ✔ 5 evals | ✔ 704 | ✗ 0.99, just outside the band (7) |
| a ≈ 3.0 ±1% (beyond DOE max 1.84) | ✗ stalled at 1.84 | ✔ 704 | ✔ 7 |
| a ≈ -0.2 with g ≥ 0.5 (leak regime) | ✔ 1 (already in DOE) | ✔ 704 | ✔ 11 |
| maximize a | 1.92 (60) | 3.53 (704) | n/a |
| a ≈ 1.0 & ‖x‖₁ ≤ 2 | ✗ 0.61 | ✗ 0.84 | ✗ 0.88 (may be infeasible) |

**Reading the table**
- globalMOO is a learned surrogate. It interpolates inverse targets inside its trained envelope in a handful of evals, and once trained, each query is cheap ("train once, query many"). It does not extrapolate past the DOE range.
- So for *maximization* or out-of-envelope targets, use gradient methods or DE, or re-train globalMOO on a zoomed box around the best region.
- For *mapping a neuron's response surface at many target levels*, globalMOO is the right tool.

**Integration bug, fixed:** globalMOO PERCENT bands around an initial value of 0 collapse to exactly 0 and pin that output. Untargeted outputs now use absolute VALUE bands.

## 2026-10-17: Pilot, 12 random Bonsai-8B (1-bit) neurons, read forwards and backwards

**Forward reading** (`runs/pilot_interps.json`)
- Corpus: 8M tokens. Read the top 10 examples per pole for 12 random neurons from layers 8–28, 24 poles in all.
- Result: 19 of 24 poles show one main job, 1 shows two related jobs, and 4 are not yet understood.
- Many jobs hold across languages:
  - "nerve" in English, German, Korean and Arabic;
  - the afterlife ("heaven", "Paradise", الدنيا والآخرة);
  - "best" in competitions.
- The two poles are usually related (afterlife / God's presence; payment terms / debt collection). L10/N5749 is the exception: genre (+) vs refuelling at gas stations (−).
- No echo (−×−) regime appears among strong activations for any of the 12 neurons.
- Caveat: this is a first reading by Claude, not blind-tested.

**Backward screen** (`data/screen_bonsai1.jsonl`, local only, no globalMOO)
- Method: gains on each pole's top 24 upstream neurons by attribution, using its top example, with context capped at 384 tokens. The cap keeps 91–103% of the full-window activation.
- **Round trip.** Whenever the target is reachable, `least_squares` lands within about 0.04 of the requested fraction in ≤25 evaluations. Targets below the zero-all-24 floor are unreachable by construction.
- **Concentration varies widely.** Zeroing the top 24 leaves anywhere from 0.00 (printf pole) to about 1.0 (several poles whose input comes through attention or is spread out).
- **Keep-curves.** Keep only the top k upstream neurons and zero the rest:
  - L9/N3688+ ("nerve"): 0.37 at k=7, jumping to 0.98 at k=8. One unit (L7/N384, N-word subword completion) is pivotal.
  - L13/N10734+: 0.41 at k=1 and 0.78 at k=2.
- **Antecedent chains** (`runs/pilot_antecedents.json`):
  - "genre" = the token "genre" + the subword "gener-" + typology + literary forms;
  - printf = `{}` placeholders + `%s` specifiers + `/%`;
  - the "nerve" pole is fed by an Arabic nerves/neural neuron.
- **Lesson 1: sample each branch.** The top example of the "best" pole is its "all … in common" offshoot, so the backward screen described the offshoot. Run the inverse per forward cluster.
- **Lesson 2: the route-sharing metric is too strict.** It compares raw attribution vectors over 442k units at cos > 0.5, and nearly every pole came out as 8 separate routes. Group into labelled bundles before comparing.

**Infrastructure fixes**
- Autograd kept every dequantized weight alive, which is the whole model in bf16. A custom backward now re-dequantizes instead.
- Screens set a GPU memory cap, so they fail instead of spilling into shared memory.

## 2026-10-17: Neuron probe (forward + inverse + induce) on L9/N3688, Bonsai-8B 1-bit

Tool: `python -m bonsai_interp.probe.run {forward|solo|invert|induce} 9/3688 +`. All artifacts are in `probes/bonsai1_L9_N3688_pos/`.

**Forward**
- Top corpus examples suggested H1: "the morpheme *nerve* across languages".
- A bank of 98 natural sentences, rated by meaning before evaluation, showed that H1 over-claims: anatomical "nerve" is weak.
- H2: *nerve(s) as psychological strain or composure*.

**Position confound (methodology)**
- Bare sentences put the key token at sequence positions 1–10, where this neuron is damped.
- The tool now evaluates every text after neutral preambles (mean of 3), reading only the text's tokens.
- This changed conclusions, so it is now the default.

**Pre-registered held-out test of H2** (24 sentences written before evaluation, with fire/quiet predictions)
- Accuracy 0.79, AUROC 0.92.
- 7 of 12 predicted-fire sentences fired (≥30% of corpus max); 0 of 12 predicted-quiet fired.
- Misses: French "les nerfs", Korean "신경을 쓰다", "nerves of steel", "nervous energy", "nerves were shot". All were weaker than predicted, never false positives.

**Text-level inverse** (3-sentence passages on the bank manifold; inputs are concept and strength coordinates per slot)
- Graded targets: 25 / 50 / 75% met by both solvers. Context can raise a sentence from 0.17 alone to 1.08.
- Leak, i.e. firing with nothing nerve-related: only differential evolution found one, 0.40 on " strong" ("A strong breeze…" after a violent sentence). This is plausibly the fortitude branch.
- Offshoot: anatomical "nerve" fires 0.33–0.60 in non-initial positions (神经, нерва). The Korean name 신경민 fires 0.51, a token-identity offshoot.
- Sense flip: "She finally worked up the nerve" scores 0.72 alone and 0.08 after a pain-receptor sentence, while the opposite pole jumps to 1.56. The neuron resolves the sense of "nerve" from context.

**globalMOO vs SciPy differential_evolution** (one globalMOO model, trial 17552; plan pre-registered in `gmoo_plan.json`)

| | globalMOO | differential_evolution |
|---|---|---|
| Model evaluations for the 8-goal battery | 268 (161 shared design points + queries) | 4,992 |
| Goals met | 6/8 (max 0.84; missed leak and near-miss) | 8/8 (max 0.97) |
| Tight ±0.015 targets from a neutral start | 25% in 17 evals; 50% and 75% within 0.02–0.03 after 41 evals | all three at 624 evals each |

- Caveat: 5 of globalMOO's goals were met by its best design point. The tight-target queries are the fair test of its learned inverse.

**Forward-induce** (add c × corpus max to the neuron at every position)
- c = −2 to −40: gains pain / heat / warmth, then stimulus / receptor / reflex. "I felt" continues "the pain" instead of "nervous".
- c = +2 to +10: gains calmly / inspired / strongly / powerful / proud.
- Above +20: text degrades. Concept tokens never rise, which fits a layer-9 detector.

**Verdict:** one context-resolved axis, psychological strain/composure (+) versus bodily sensation (−), with local offshoots (anatomical sense, the 신경 token in names, "strong"). No unrelated concept was found by the forward, inverse or induce views.

## 2026-10-17: Fan-out from L9/N3688+ ("nerve"): neighbourhood and circuit inverse

`python -m bonsai_interp.probe.neighbors 9/3688 +` uses 12 branch examples (6 psychological, 4 anatomical, 1 name, 1 "strong"), all read after a neutral preamble.

**Upstream (single-unit zero ablation; share of the pole's activation lost)**
- **Narrow fan-in, two dominant inputs:**
  - L3/N6493, 74%: the token "nerve" (nerve, Nerven, ner-ve). Loses 95% on anatomical examples and 77% on psychological ones.
  - L8/N7756, 61%: the nerve/neural concept, with Arabic عصب top examples.
- **Minor inputs:**
  - L7/N1548 (veins), L5/N5628 (enceph-), L3/N6185 and L0/N11629 ("neuro-" prefix, mostly the name branch), L0/N1684 ("serve", the "-erve" spelling), L6/N1460 (numbness/tingling).
- The "strong" branch uses none of these.

**Sense-flip disambiguators** ("worked up the nerve" alone vs after a pain-receptor sentence; nerve goes from +6.5 to −9.4)
- L8/N3858, a pain detector: zeroing it alone restores 25% of the gap.
- L6/N2055, pain across languages (Schmerzen, doler, وجع).
- Zeroing the top 40 candidates together restores 65%. The rest presumably comes through attention.

**Downstream (zero the nerve neuron, change in later neurons relative to their corpus max)**
- **Wide, weak fan-out.**
  - Psychological branch: L10/N6901 (nervous system / neuroanatomy) moves +0.115; L14/N9470 (emotional disappointment) +0.06; the rest are under 0.06.
  - Anatomical branch: different readers, all under 0.05.

**Circuit inverse** (5 outputs: nerve, token, concept, pain, reader; same natural-passage manifold; `circuit.json`)

| goal | DE (624 evals each) | globalMOO (161 design points + ≤41 per goal) |
|---|---|---|
| bypass: nerve ≥ .3, token ≤ .1 | ✔ 0.61 via the concept route (Arabic "nervous system") | ✗ 0.29 |
| gated: token ≥ .5, nerve ≤ .1 | ✔ with pain 1.04 (sense flip) | ✔ in 10 evals, **pain 0.02**: "He lost his nerve" |
| override: nerve ≥ .5, pain ≥ .5 | ✔ 0.61 via the concept route next to "chronic pain" | ✗ 0.28 |
| concept without token | ✔ (same passage as bypass) | ✗ near miss (concept 0.50, nerve 0.29) |
| reader with nerve quiet | ✗ best 0.40 | ✗ |
| clean max (pain ≤ .1) | 1.06 | 0.37 |

**Reading (hypothesis-grade)**
- Two parallel input routes. The token route (Latin script) is sense-gated: pain context closes it, and so does the "lost his nerve" construction (counterexample found by globalMOO).
- The concept route (non-Latin scripts) looks sense-agnostic.
- The pole's psychological specificity is a property of the token route and its gates.
- Solver note: on this 5-output, rare-combination problem DE beat globalMOO on 5 of 6 goals. globalMOO's one success produced the most informative counterexample.

## 2026-10-17: Pre-registered gate/route test for L9/N3688+ (`gate_test.json`, committed in 9613a67 before the run)

Each sentence was read after [preamble] + context, under neutral and pain contexts (2 contexts × 3 preambles each). Nodes were measured on the sentence's own tokens. Result: **9 of 11 predictions pass.**

| prediction | result |
|---|---|
| A open Latin constructions fire (≥ .3) | ✔ 7/8 |
| A pain context halves them (ratio ≤ .5) | ✔ 6/8 (e.g. "Ich brauchte starke Nerven" 0.72 → 0.03) |
| A pain unit ≥ .3 on the sentence's tokens | ✗ 0/8. The pain unit fires on the context sentence, not on the test sentence. The gate is carried forward, presumably by attention. |
| B "lost his nerve" / "had the nerve to": token ≥ .4 | ✔ 6/6 |
| B …while the pole stays < .3 | ✔ 6/6; mean 0.19 vs 0.60 for set A |
| C Arabic concept unit ≥ .3 | ✔ 4/4 (token unit 0.00) |
| C Arabic fires in either sense | ✔ 3/4 |
| C Arabic not gated by pain (ratio ≥ .5) | ✗ **0/4**. Ratios 0.07–0.37: pain closes the concept route too. |
| D Latin anatomical controls: token present, pole weak | ✔ 4/4, ✔ 4/4 |

**Exploratory (unscored)**
- "kept his nerve" fires (0.41); "lost his nerve" does not (0.14).
- "What a nerve he has" fires (0.61); "had the nerve to" does not (0.14–0.25).
- Korean 신경 fires ~0.31–0.37 in both senses. Chinese 神经 does not fire (0.02–0.04).

**Correction to the earlier circuit reading**
- The "override" example (Arabic beside "chronic pain" at 0.61) had the pain sentence *after* the Arabic one. A causal model cannot be gated by later text, so that example never tested the gate.
- Revised story: both input routes are gated by *preceding* physical-sensation context. The token route is additionally construction-sensitive: open for "worked up / kept / mustered the nerve", "nerve-racking", "Nerven blank"; closed for "lost his nerve" and "had the nerve to".

## 2026-10-17: Probe calibration on three neurons, and neighbour mining for L9/N3688

**Protocol**
- Every neuron got the same probe.
- Banks for the two calibration neurons were written by separate agents that saw only corpus examples.
- At most one label refinement is allowed before the held-out test. The L22 refinement was done by an agent that saw the bank results.
- Every held-out set and globalMOO plan was committed before it was run (commits 3977f9d, 103f75a, 973fd76).

| measure | L9/N3688 nerve | L10/N5749 genre | L22/N11024 unclear |
|---|---|---|---|
| held-out fire rate (label-written text fires ≥ .3) | 58% | **75%** | **33%** |
| held-out false positives | 0% | 0% | 0% |
| held-out AUROC | 0.92 | 1.00 | 0.92 (after refinement) |
| opposite pole, on-label fire rate | 82% | 89% | **0%** |
| cross-talk (opposite pole on the pole's concept) | 0.21 (related axis) | 0.005 (unrelated) | 0.02 |
| leak found by the DE inverse | 0.40 ("strong") | 0.29 (none) | 0.23 (none) |
| induction | coherent both ways | coherent both ways (fuel; genre / literary / tradition) | incoherent (relief, glad / on, still), ~6× weaker KL |
| globalMOO 25 / 50 / 75% | ✓✓✓ | ✓✓✓ in 7–10 evals | ✓✗✗ |

**Reading**
- The probe's verdicts differ as intended: one context-gated axis, two clean unrelated poles, and no single coherent job.
- No single number does it. AUROC is too lenient: a refined label can rank sentences without making the neuron fire. Low leakage is not cleanliness: the unclear neuron barely fires on anything.
- The discriminating evidence is the held-out fire rate (the essay's own new-example test), opposite-pole coherence, induction coherence and strength, and cross-talk (related vs unrelated poles).
- The unclear neuron's refined label did find real structure. Hypothetical money outcomes ("would have roughly $200,000") reach 0.4–1.0.

**Neighbour mining, L9/N3688** (`bonsai_interp/probe/mine.py`)
- Method: 115 top corpus sentences of 12 causal neighbours (6 inputs, 2 gates, 4 readers), rated for relevance before evaluation (commit 3977f9d).
- 6 fired the pole off-label:
  - 3 Arabic anatomical sentences, the known concept-route offshoot;
  - "nerve poison";
  - a symptom list peaking on "tingling", a small physical-sensation leak;
  - one unexplained (0.33).
- None of the 40 sentences from downstream readers fired it.
- No unrelated alternative reading was found. The mining confirms the known offshoots.

**New views**
- The iris plot: every probe sentence placed around the neuron by type (angle) and label relevance (radius), coloured by which pole it drives. It is in every dossier and on the calibration page (`reports/calibration.html`).

## 2026-10-17: Quick checks of nerve's four key neighbours (`bonsai_interp/probe/quick.py`; tests committed in d56f984)

Each neighbour got: weights-only direct token drivers; "one layer back" (attribution on its own top corpus examples, then zeroing each candidate feeder); and 8 fire / 8 quiet pre-registered sentences.

| neighbour | label | fire / FP | one layer back |
|---|---|---|---|
| L3/N6493 + (main input) | token "nerve", Latin script | ✔ 8/8, 1 FP ("Nerf" 0.39) | Direct tokens: ` nerve`, ` nerves`, then Serv / erved / reserved. Feeders are layer-0 spelling units (serve, reserved, -erv-). **A spelling detector for nerv/nerf.** |
| L8/N7756 + (second input) | nerve/nervous-system concept, Arabic-heavy | ✔ 8/8, 0 FP (Arabic 0.9–1.2, German 0.67, English 0.37–0.49) | Arabic subword units (بي, ب) and the "veins" unit |
| L8/N3858 + (pain gate) | pain, word and concept | ✔ 7/8, 0 FP ("pain au chocolat" 0.11) | Multilingual pain unit L6/N2055 (Schmerzen, doler), then numbness |
| L10/N6901 + (strongest reader) | nervous system / neuroanatomy | ✗ 4/8 | Fires on motor commands / motor control / speech production; silent on hippocampus, thalamus, dopamine. Feeders: ` motor` detectors at L0 and L2. **Post hoc (untested): a motor ↔ sensory axis.** Its − pole is taste and smell receptors. |

Reading: the circuit's inputs and gate are solid. The token route is spelling-level. The reader was mislabelled, and it does not respond to psychological "nerve" ("my nerves were on edge" 0.02).

## 2026-10-17: The neighbourhood web around L9/N3688 (`bonsai_interp/probe/web.py`, `lateral.py`; tests committed in b273a0c, plan in bfac101)

**Crawl:** 48 neurons.
- 6 readers: kept if at least 2 of 3 methods agree (zero-ablation, a 2×-max boost, weights-only cosine).
- 24 siblings: other main inputs of those readers.
- 10 co-firing on nerve's sentences. Sanity check: the top one is nerve's own concept input, r = 0.69.
- 8 second-hop upstream feeders.

**Testing:**
- Every node got a blind label from 3 Claude agents (top examples only), and two pre-registered 8 + 8 sentence tests from the same label: one written by Claude, one by Bonsai-8B itself.
- A label holds with at least 5 of 8 firing and at most 1 false positive.

**What held**
- Claude-written tests held **38 of 48** labels.
- Upstream feeders and siblings are mostly crisp token or spelling detectors: motor, serve, diarr-, trait, -ate, vic, Albany, p-value digits.
- **All 6 readers failed.** Nerve's downstream is diffuse at this level of analysis.

**Lateral family of psychological "nerve"** (all held their labels)
- suspense L7/N9782; eager anticipation L9/N1061; overwhelmed L12/N3688; "disappointed" L1/N4037.
- Each fires on its own sentences (0.6–0.9) and barely on the others' or on nerve's.
- Zeroing any one moves the others by at most 0.03: parallel jobs, not a chain.
- Their shared reader L14/N9470 reaches 0.30–0.34 without nerve, via anticipation and suspense. This **falsified** the pre-registered prediction; it is a family hub.

**Lateral-cluster inverse** (6 outputs)
- DE met 6 of 6 goals at about 600 evals each.
- globalMOO met 5 of 6 at **2–10 evals** each (one model, trial 17556; a crashed first run was resumed on the same trained model).
- Answers: every relative fires without nerve; nerve fires with all of them quiet (0.60); mixed passages co-activate them.

**Two authors**
- Bonsai's sentences are more predictable to Bonsai (median perplexity 40 vs 57), yet they fire the targets less (66% vs 80%) and misfire more as near misses (27% vs 3%).
- Excluding the 146 sentences with leaked prompt wording: 69% vs 80% fire, 15% vs 3% false positives.
- Bonsai's tests held 18 of 48 labels, failing mostly on false positives. The model can produce a concept but cannot reliably write around it.
- Self-generated text gets no special treatment from these neurons.

**globalMOO usage to date:** 8 models (3 toy, nerve, nerve circuit, genre, unclear, nerve lateral cluster).

## 2026-10-17: "nervous" vs the word "nerve" (exploratory, not pre-registered; `runs/nervous_exploratory.py`)

- English " nervous" is a single token, unlike " nerve". **The nerve neuron barely responds to it.**
  - 0.01–0.03 on "felt nervous", "nervous about", "investors were nervous".
  - "a nervous laugh" 0.24; "nervousness" 0.15.
  - Same meaning, different surface form. "nerve-racking" and "worked up the nerve" give 0.6+.
- Forms tokenized with a nerv- piece do fire it: German "nervös" 0.68, Russian "нервничал" 0.49, French "nerveux" 0.44, Spanish "nervioso" 0.17.
- Control: swapping in "anxious" gives 0.02. The suspense and anticipation units respond about equally to nervous and anxious (0.18–0.26).
- Contrasting "nervous" with "anxious" over layers 1–14 finds only spelling-level units: L2/N8177 "Nor", L4/N7189 "-ve / erv", L1/N9531 "na-", plus the known token and concept inputs. No dedicated semantic "nervous ≠ anxious" unit turned up. Expected, since the two are near-synonyms.

**Reading:** L9/N3688 is not a "nervousness" neuron. It responds to the word *nerve* (and nerv- forms) when used in its psychological sense. The token route makes it surface-bound: the meaning of English "nervous" is carried elsewhere, alongside "anxious". This is a third sharpening of the label, after H1 (any "nerve" morpheme) and H2 (psychological sense).

## 2026-10-17: Trying to disprove the nerve neuron, relabelling the failures, and the lateral family's inputs (`bonsai_interp/probe/disproof.py`, `bands.py`, `h4.py`, `predictive.py`, `lateral_out.py`; plans committed in 77e7426, 192e5b9, 0eeecfb, d9e9e34 before each run)

**Band scan.** Full per-token activations for nerve and 14 web neurons over 6,000 corpus windows (6.1M tokens, `data/bands_web.npz`). This was used to look below the top examples.

**Disproof attempts on L9/N3688+** (`disproof_plan.json`)

D1, corpus strong firings (prediction: ≥ 90% have a 'nerve' word within 3 tokens). **Failed.**
- Only 49 of 379 (12.9%) of strong firings (≥ 0.3 of max) have a 'nerve' word within 3 tokens.
- The rest are mostly ' strong' and words for being hit, hurt or wounded.

D2, blind annotation of 120 contexts in the bands 0.05–0.1, 0.1–0.2, 0.2–0.3 and 0.3–0.5. The annotator could not see the bands.
- About 100% of contexts are "unrelated" in every band.
- No theme reaches a third of any band. The largest is injury/violence, 7/30 in 0.3–0.5.
- Force, injury and strength together make up about 63% of the 0.3–0.5 band.
- The prediction failed, but the disproof condition (one coherent unrelated theme) was not met.

D3, leak search with SciPy DE over 1,199 sentences containing no 'nerve' word.
- The maximum is 0.52, with its peak on ' strong'. This is the strength branch, not a new concept.

Per-token table (`token_table.json`)
- ' strong' fires strongly on 88 of 538 occurrences (16%, mean +0.17).
- ' nerve' fires strongly on 13 of 143 occurrences (9%, mean −0.05). Its anatomical uses push the − pole.
- Next come injury, hit, nervous and wounded.

H4, one coarser "force / strength / strain" job (`h4_test.json`, sentences without any 'nerve' word). **Rejected**: two of the four branches fail.

| Branch | AUROC | Mean activation |
|---|---|---|
| strength | 0.97 | 0.25 |
| impact/injury | 1.00 | 0.13 |
| reaction/sensation | 0.47 | – |
| courage/composure without the word | 0.61 | – |
| psychological 'nerve' (reference) | – | 0.50 |

Anticipation (exploratory, whole-word nerve tokens only)
- Activation vs P(next token is 'nerve') has a Spearman correlation of −0.01 over 556k positions.
- It is not a predictive unit.

**Reading (H5).** The narrow label is falsified on frequency.
- L9/N3688+ is a lexical detector that peaks on the psychological word 'nerve'.
- It has reliable but weaker branches for 'strong' (about half the peak) and for hit/hurt words (about a quarter).
- No unrelated concept was found anywhere in its range.
- The meaning alone does not drive it: courage without the word, sensation, and anticipation of the word all fail.
- It is one family of words, not one concept. Whether that counts as "monosemantic" depends on whether features are defined by meaning or by the vocabulary they serve.
- Selection-bias note: nerve was chosen as the clearest of 12 random pilot neurons.

**Relabelling the 10 web neurons whose labels failed** (two more rounds, using band-sampled examples as well as top examples)
- 4 now hold:
  - L9/N2445: religious identity in Jewish texts, 7/8.
  - L10/N5553−: biology-textbook terms, 5/8.
  - L7/N7149: the Tul·are piece, 8/8.
  - L17/N2518: head of a Python dotted reference, 5/8.
- 3 are dense units, on for 20–72% of tokens above 0.1 of max: L13/N3792, L19/N2291 and L14/N9470, the family hub. An 8 + 8 test cannot hold for these by design.
- L20/N5963 is weakly dense and incoherent.
- 2 remain unresolved after three rounds: L10/N6901 (4/8 each time) and L18/N12193.

**Inputs of the lateral family** (12 new neurons, 3 per family member, by zero-ablation; 10 of 12 labels held)
- can't-wait L9/N1061 is driven by:
  - 'to' after eagerness (L8/N3632, 27%)
  - "can't stand / can't bear" (L7/N10645, 20%)
  - the 'th' of '19th century' (L7/N2810, 6%)
- disappointed L1/N4037 is driven by 'disconnect' (L0/N10546, 35%), a noun after 'the' (L0/N55) and 'app' in code (L0/N10323).
- overwhelmed L12/N3688 is driven by 'answer to / accountable to' (L8/N8229), a reader-need hook in ad copy (L11/N8779) and the second piece of a technical term (L6/N2446). Each contributes only 3–4%, so this one is diffuse.
- suspense L7/N9782 is driven by 'life span' (L1/N1313, 10%). Two inputs failed: 'watch' TV (L0/N947, 3 false positives) and an unclear filler unit (L4/N7913).

**Totals.** The web now has 60 neurons, and 52 labels hold on the latest round.
- Published: web page https://claude.ai/artifact/QfEFnhVJj281rQw5Jpx3Xu and dossier https://claude.ai/artifact/AHrzdNHkF9wYFP9FVWXJRt.
- Figure: `reports/figures/nerve_connectome.png`.

## 2026-10-17: Blind clean-room replication of L9/N3688 (`runs/cleanroom/`; our predictions committed in 2dae2f0 before launch)

**Setup**
- A headless Claude Code agent (Opus 5.5) ran in a separate folder with its own git repo.
- It had a scrubbed copy of the toolchain, the corpus and scan, and the model weights. It had none of our probes, findings or reports. Nerve-tailored steering prompts and opposite tokens were removed.
- Web access and reads of this repo or of Claude memory were denied by permission rules. The full transcript was audited: 51 tool calls, 0 denied, 0 flagged.
- It took 40 minutes, 4 pre-registration rounds and one globalMOO model.

**Its labels** (from `blind/REPORT.md`)

+ pole: "several jobs", confidence 0.7. All three jobs are vetoed by preceding pain or sensation discourse.
1. "nerve(s)" outside neuroanatomy, in several languages. This is the main job by peak.
2. Physical or material "strong". This is the most frequent strong trigger, and it is damped by business register.
3. A weak branch for violent harm, peaking on the body part.

It rejected: "nerve in any sense" (0/8), nervousness or anxiety ("nervous" 0.01–0.06), danger or strain context (Δ≈0), feeling words in other languages, and one unified job (held-out accuracy 0.62, disjoint feeders).

− pole: "somatosensory / nociceptive physiology", one main job with an offshoot, confidence 0.8.
- The blind held-out set scored accuracy 0.96 and AUROC 1.00.
- "Pain in any sense" failed at 0/8, and "medicine in general" failed.
- The offshoot is "pulling a hand or limb back".

**Agreement with our work**
- The nerve feeder is the same neuron: L8/N7756 (0.635 vs our 0.61).
- The pain neuron L8/N3858 acts as the context switch, as in our gate test.
- "nervous" is silent, matching our H3.
- 'strong' and impact are branches and courage-meaning is not, matching our H5 and H4.
- globalMOO behaved the same way: good at graded targets in 5–22 evals, fails at extremes.

**New to us**
- **Disjoint feeders per + branch.** 'strong' is fed by L8/N9260 ("Powerful, potent") and L0/N8811 at 0.29 each, with about 0 drop on nerve. The nerve feeder gives 0.001 on strong. This argues for "several jobs" over our "one word family".
- **The pain-context veto as a signed axis.** One pain sentence drives every + target to about 0 (nerve −0.46, strong −0.35, injury −0.26) while the − pole rises.
- **A − withdrawal offshoot**: "pulled her hand away" 0.63–0.98, with separate feeders. We never examined this.
- **A register confound in our preambles.** The office/invoice preamble damps physical "strong" by about 0.17. This may explain why our branch tests read 'strong' at about half of nerve.
- It did not report our spelling route L3/N6493.

**Our pre-registered predictions** (`scored.json`): 6 of 8 held, 1 was partial and 1 failed.
- P2 was partial: its label is "nerve outside anatomy", broader than the psychological sense.
- P7 failed: it chose "several jobs".

**Disclosure.** The clean room is blind to our results, not to our methods.
- The README's practice list included "look at weak and middling activations, and how often each kind of trigger occurs", which our disproof round taught us.
- The toolchain carries our preambles, thresholds and goal battery.
- The agent is the same model family as the original investigator.

## 2026-10-18: Second blind clean-room run, on a random neuron (bonsai1/L12/N5397; draw committed in neuron-cleanroom 0d513c1, predictions 1035094)

**Setup**
- The neuron was drawn with a fixed seed (20261018), and the draw was committed before anyone looked.
- I made a quick study of my own and committed it first:
  - − pole label held at 5/8 with 0 false positives (b756721 → b35e372).
  - + label (the delimiter closing a dense run of detail) failed at 0/8.
- The room was built by `cleanroom.build_room` from interp commit a4d218d with `--hints none`.
- Run: 43 minutes, 77 tool calls, 2 sub-agents, 0 globalMOO models.
- Audit clean once two auditor false positives were fixed. The agent used its own session scratchpad twice, to hide the answer key of a blind test whose sha256 it committed first.

**Its findings**

Window-start artifact:
- 17 of the + pole's 32 top corpus examples sit at window positions below 64 (about 2 expected).
- After a neutral preamble they collapse, e.g. 0.95 → 0.16.
- Both poles are 3–4× enriched in the first 64 positions.
- This explains why my + label failed: it was built on the artifact.
- **Every scan-based label may be affected.** The scan should exclude or flag early positions.

+ pole: several jobs (confidence 0.55).
- The peak is the line break that ends a row deep inside a long hex/byte dump. It grows with dump length: 3 lines 0.17, 12 lines 0.57, 30 lines 0.88. It exceeds the corpus max on synthetic dumps.
- Separate offshoots, each with disjoint feeders: `', '.join` separators and long "ha ha" runs.
- A diffuse 0.3–0.5 tail remains unexplained. Blind labelling of corpus hits scored at chance (0.45).

− pole: several jobs (confidence 0.65).
- Its top-level reading (H-1, token right after a number or code) held at 8/8, 0 false positives, matching mine.
- It splits into four sub-jobs, each 5–6/6 with 0 false positives and its own feeders:
  - unit after a dimension
  - nested-array row closer
  - inner characters of long IDs
  - delimiters after phone numbers and citation years
- "Any number" fails: version strings 0.07, coordinates 0.25.

**Our predictions:** 7 of 8 held. P5 (− pole "one main job") failed.

**Pattern across both blind runs:** the agents count jobs by circuits, so disjoint feeder sets mean several jobs. This is a defensible operational definition, and it gives a stricter verdict than the essay's "one job plus offshoots".

**New in the graph:** a derived `scrutiny` level per pole, separate from status. The levels are tested, re-tested, checked, investigated and replicated.

## 2026-10-18: Is L9/N3688+ a neuron for words with both a physical and a psychological sense? No. (`probes/bonsai1_L9_N3688_pos/dualsense_test.json`, committed in 4a74b78 before running)

**Setup**
- 13 dual-sense words: strong, weak, stiff, tough, hard, soft, fragile, spine, backbone, grit, crack, shaken, broken.
- Each in one physical and one psychological sentence.
- Controls: 4 psychological-only sentences and 4 physical-only sentences.

**Results**
- Excluding "strong", **0 of 24** sentences fired at ≥ 0.3 (mean 0.046). The physical-only controls average 0.026, and the psychological-only ones 0.018.
- "strong": physical 0.48, psychological 0.07.
- Reference sentences with "nerve": 0.24 and 0.47.

**Against the predictions**
- P1, P2 and P3 held.
- P4 failed: the physical impact words did not fire (crack 0.05, shaken 0.06, broken 0.08), so the hit/hurt branch is narrower than its name.

**Reading.** The neuron's two strongest triggers prefer opposite senses: physical "strong" and psychological "nerve". The dual-sense hypothesis is rejected. What unites the branches is still open: packing, a shared downstream use, or a token-level property. The circuit trace is the planned test.

## 2026-10-18: Blind run 5, L17/N9491 (Claude): a long-context copying unit (neuron-cleanroom `runs/2026-10-18_bonsai1_L17_N9491`, PR #4)

- **+ pole held under quick-8+8@2** (AUROC 0.94, median 0.224): "near-certain continuation in new text", most often the next piece of a rare word, name or identifier already used earlier in the document.
  - It needs long context (256 to 1,024 tokens). Removing the earlier mention kills or halves the peak.
  - A pre-registered held-out entropy check replicated the direction but narrowly missed two of its own thresholds, and the report says so.
- **− pole:** no label survived.
- **Feeders:** 2 of 4 held (arsenic L3/N2114−; analogy vehicle onset L14/N7100+).
- **Our predictions:** 5 of 6 (Brier 0.17). The miss was ours: the top examples looked like idioms, and the blind agent's occlusion and entropy tests showed copying instead.
- **Audit:** clean after one auditor fix (`../x` after `cd` into the agent's own scratchpad).
- **First PR through the colour-in lane:** checks passed, the merge workflow merged it 10 s later and pushed "fold after #4".

## 2026-10-18: The auto tier at layer 12: a pole's top peak token rarely explains it (`graph/auto/bonsai1_L12.*`, pre-registered in e04269c)

**Rule (auto-token@1, fixed before scoring)**
- **Candidate:** a pole is a candidate when one token is the peak of ≥ 16 of its 32 top scan examples.
- **Test:**
  - Fire: 8 held-out snippets (corpus windows the scan never read) that end on that token.
  - Near misses: the same snippets with only that token swapped for a same-shape, similar-frequency word.
- **Judge:** quick-8+8@2.

**Results (layer 12, 24,576 poles)**
- **Classes:**
  - context: 22,743 (no peak token covers half the top examples)
  - token: 644, plus 544 untestable (punctuation and fragments with no same-shape substitute)
  - dense: 558
  - positional: 87
- **Held:** 34 of 644 (5%). The failures:
  - The median fire snippet reached only 0.04 of max, and the median AUROC was 0.59.
  - Function words: 0 of 243 held.
  - Even poles where all top examples share one token: 4 of 54 held.
- **What held is clean:**
  - English: 'according' (×3), 'hopefully', 'maybe', 'pytest', 'classmethod', 'facebook', 'redistribute'
  - Other languages: 'auch', 'として', 'حيث'
- **Cost:** 11 minutes of CPU and 2.4 minutes of GPU per layer.

**Reading**
- At layer 12, sharing a peak token is far from being a token detector. Most of these poles respond to that token only in particular contexts.
- So the top-k view overstates lexical monosemanticity. This bears directly on the essay's claim, at least for "clean" read as "fires on one token".
- The tier cannot colour the map. It labels about 0.1% of poles per layer, and classifies the rest for agents.
- A context-aware rule (e.g. previous token + peak token) would be auto-token@2. It needs a fresh pre-registration on a different layer, because layer 12 was used to look.

## 2026-10-18: Sweep missions, the first five rounds (`runs/playtest/2026-10-18/sweep-*.jsonl`, PRs #5-#9)

Headless agents played the survey page's sweep prompt in an ordinary player checkout (1.2 GB GGUF weights), one sweep at a time, and every PR merged itself.

| round | agent | held | cost | minutes |
|---|---|---|---|---|
| 1 | Claude (opus-5.5) | 10/16 | $0.90 | 6.5 |
| 2 | Codex (default model) | 2/16 | n/a | ~6 |
| 3 | Claude | 12/16 | $1.41 | 7.2 |
| 4 | Codex | 6/16 | n/a | ~8 |
| 5 | Claude | 10/16 | $0.93 | 4.6 |

**Reading**
- 40 of 80 poles held, and the map grew from 100 to about 160 poles, at about 2 to 3 minutes and $0.06 per pole.
- **The agents label differently.**
  - Codex wrote narrow "word X in context Y" labels, with near misses that keep the word and change only the context: 4 of 16 reached AUROC 0.90 in round 2.
  - Claude labelled at the token or topic level the card supports: 13 of 16 reached it.
  - So in a sweep, Codex mostly tests context-gating hypotheses and finds them false, and that is still information.
- **Typical failures:** poles the card flags broad or weak, and labels written from scattered cards.
- **Papercuts fixed between rounds:**
  - `download_models.sh` lost `config.json`
  - the card's logit effect could not read the GGUF
  - `play run` printed counts that read like the pass rule
  - `draw` and `new` ids could drift
  - `uv.lock` was stale
  - `<` in a sentence was taken for a template placeholder
  - one card per process was slow (cards now batch, and `probe --plan` scores many poles at once)
  - branch names collided
  - long output was truncated
  - the PR body file sat inside the repo

## 2026-10-20: auto-token@2, a staged corpus screen for the auto tier (`bonsai_interp/auto/context_tier.py`; layer 20 pre-registered in ee6a89a and be402bc)

auto-token@1 read a pole's 32 top examples and guessed "the token X". At layer 12 that guess held for 34 of 644 poles, because most poles respond to their peak token only in some contexts. auto-token@2 asks the corpus first, and the top examples never.

**Pipeline (no LLM until the dossier)**
- **Three window splits, so no stage reads what a later stage is judged on:**
  - 0..7999 selection (stage 1)
  - 8000..15999 screening (stage 2)
  - 16400.. testing (the auto test only)
- **Stage 1, word hits by frequency.** One streaming pass over the selection windows: threshold on the GPU, bincount the (token, pole) and (previous token, token, pole) pairs. A strong activation reaches 0.3 of the pole's corpus max; positions below 64 are not counted (the window-start artifact, 2026-10-18). Per pair: precision P(strong | token), lift over the pole's base rate, coverage of the pole's strong activations. A hit needs n ≥ 8, coverage ≥ 0.05 and lift ≥ 10; each pole keeps up to 3.
- **Stage 2, a context screen.** Up to 256 occurrences of each trigger in the screening windows, scored in their own context. A rule of one or two (feature = value) conditions is fitted by information gain on "strong". The features are the previous token, its word class (a heuristic tagger), line and sentence position, window position, script and code-like text. Bins:
  - context or format: a rule with precision ≥ 0.5, recall ≥ 0.3 and a margin of 0.3 over the rate outside it
  - positional: the same, on window position
  - lexical: no such rule, but strong on ≥ 50% of occurrences
  - unexplained or rare: the rest
  - Two features are tabulated but never used in a rule. The following token can only be a correlate, because the model is causal. The corpus domain is not carried by a snippet.
- **Minimal pairs check the bin:** swap the trigger (lexical), or swap the previous token out of and into the rule's context. The result is recorded; it does not gate.
- **The auto test.** 48-token snippets from the testing windows that end on the trigger:
  - Fire: 8 where the rule holds.
  - Near misses: 4 of the word without the context (the rule fails), plus 4 of the context without the word (the trigger swapped). Lexical hits get the auto-token@1 swap.
  - Each snippet is read at its last token, after the three preambles, and judged with quick-8+8@2's thresholds.
- **Stage 3, a dossier per pole (≤ 64 KB, nothing from a test window).** It holds the card, both tables, screening examples, and with `--links` the feeders and readers (`play step`, `step --down`). An agent labels the pole from it and then writes a fresh quick-8+8@3 test.
- **The map layer.** `export` writes one record per pole to `data/auto/bonsai1_L<L>.screen.json.gz`: its bin, top hits, context rule and auto-test outcome. That file feeds `site/screens.py` ("colour by Screen"), through the auto-screens release rather than git.

**CPU smoke test (layer 20, fp32, 8 threads; thresholds lowered for a tiny corpus, so the numbers only show the plumbing)**
- **Setup:** each split read 128 windows × 256 tokens, with n ≥ 3, at least 12 occurrences and support ≥ 4.
- **Stage 1** took 5.6 min at about 95 tok/s.
  - 24,330 poles were eligible (178 dense, 68 positional).
  - 2,283 poles had at least one hit, 2,950 hits in all.
  - The commonest triggers are whitespace, punctuation and function words (' ', ',', '.', ' =', '\n', ' and', ' of'). The top-32 view underweights exactly these.
- **Stage 2,** on 400 random hits: 273 unexplained, 117 rare, 9 context, 1 format.
- **Auto tests: 2 of 8 held.**
  - ' I' as the first word of a sentence (format; AUROC 0.91, median 0.24).
  - An indentation run right after ':\n' (context; AUROC 1.00, median 0.45).
  - Typical failure: clean separation but weak firing at the trigger after a preamble ("'0' after '5'": AUROC 0.98, median 0.07).
- **The design catches context poles.** In an earlier run, the near misses for "' def' in code-like text" that keep the context but swap the word fired at 0.36-0.43. That pole marks a position in code, not the word.
- **Dossiers** were 10-16 KB. With `--links`, one takes about 25 min on the CPU.

**Changed before registration, because of the smoke test** (no L20 result exists yet)
- **Snippets 24 → 48 tokens.** Context tests separated well but fell just under the median-fire bar. For example, ' that' after a preposition reached AUROC 0.98 with median 0.19: a 24-token snippet loses the context the rule was fitted in.
- **Read at the trigger, not the max over the snippet.** With longer snippets, near misses fired elsewhere in the snippet: another '. A' in a near miss for "' A' after '.'". The claim is about the trigger position.
- **A splitting rule wins over a high overall rate.** ' self' fired on 65% of occurrences and was binned lexical, though "in source code" split it 68% vs 0%.
- **Window positions below 64 excluded,** following the 2026-10-18 clean-room finding.
- **The smoke test scored snippets from windows 16000..16399,** so the testing split starts at 16400.
- **More features, mined from the 284 held labels on main** (a survey by the coordinating session):
  - where the token sits in its word; the nearest content word 1-3 tokens back; digit position; language of the preceding text; line kind; inside quotes
  - repetition inside the snippet → a **copy** bin
  - the next token → a testable **predictive** bin, since the test selects snippets by their continuation
  - PMI with the previous token → a **collocation** bin
  - the pole's own mean activation over the preceding snippet → a **topic** bin (untested, because a text-only test cannot select it)
  - look-alike and inflection swaps for lexical hits; a look-alike that keeps firing is shown on the map as **spelling**
  - Not added: the model's own token probability and entropy (they need the full forward pass), and the synonym, translation and other-sense controls (they need lexicons).
- **A GPU smoke test at the real thresholds** (200 selection and 300 screening windows, tests from the burned block 16000..16399): 95 of 213 auto tests held (82 poles), and lexical hits held 27 of 28. Stage 1 ran at 6,000 tok/s. Like the CPU smoke test, this checked the plumbing and was not registered.
- **Corpus domain is tabulated but never in a rule.** Rules like "' up' in web text" cannot be tested, because a snippet does not carry which corpus a window came from. Script and code-like text are the locally visible stand-ins. (This change came after the final smoke run, whose numbers above still allowed domain rules.)

**Expected GPU cost per layer (RTX 5080, bf16; estimated before the GPU smoke test, which measured stage 1 at 6,000 tok/s, about 21 min per layer)**
- Stage 1: 8.2M tokens to layer 20. At the 5-7k tok/s the full-window scans have shown (`bands.scan`: about 1,000 windows in 3-4 min), that is about 20-30 min.
- Stage 2: at most 8,000 screening windows, each cut at its last needed occurrence: 15-25 min. The minimal pairs add at most about 5M tokens of 32-token snippets: 5-10 min.
- Scoring: 16 snippets per test × 3 preambles × about 105 tokens, so about 5k tokens per test. For 1,000-5,000 tests that is 5-25M tokens, or 8-40 min (auto-token@1 scored about 1M short-snippet tokens per GPU minute at layer 12, which is shallower).
- **Total about 1-2 h per layer,** depending mostly on how many hits reach a test, with a peak of about 5 GB of GPU memory. The CPU parts (ranking, features, fitting, prereg) take minutes.

**Layer 20 results** (protocol ee6a89a, tests pinned be402bc, scored at be402bc; `graph/auto/bonsai1_L20.auto2.summary.json`)
- **Stage 1:**
  - 11,343 of 24,330 eligible poles have at least one word hit, 20,740 hits in all.
  - 178 poles were excluded as dense and 68 as positional.
- **Stage 2 bins** (of 20,740 hits):

  | bin | hits |
  |---|---|
  | unexplained | 19,315 |
  | context | 520 |
  | predictive | 300 |
  | lexical | 251 |
  | format | 128 |
  | topic | 79 |
  | collocation | 69 |
  | rare | 45 |
  | positional | 27 |
  | copy | 6 |

- **Auto tests: 682 of 1,185 held (58%), covering 562 poles.** auto-token@1 managed 34 of 644 at layer 12. The median AUROC was 0.95.

  | bin | held |
  |---|---|
  | lexical | 222 / 228 (97%) |
  | context | 242 / 491 (49%) |
  | format | 53 / 108 (49%) |
  | predictive | 134 / 285 (47%) |
  | collocation | 27 / 67 (40%) |
  | copy | 4 / 6 |

- **Why the 503 failures failed** (a test can fail on several counts):
  - 445 had AUROC < 0.90
  - 244 had more than one near miss at ≥ 0.30
  - 106 had a median fire below 0.20
- **Examples that held:**
  - lexical: ' provides', ' plug', ' looked'
  - context: ' whose' right after a preposition; ' get' shortly after the word 'def'
  - predictive: ' area' right before a preposition
  - collocation: ' ©' and ' عليه' in fixed pairs (PMI ≥ 5)
  - format: ' return' in code-like text
  - copy: ' django' repeating an earlier pair
- **The pre-registered expectations:**
  1. More held poles than auto-token@1: **yes**, 562 against 34.
  2. Lexical > context > format: **lexical yes** (97%). Context and format **tie** (49.3% vs 49.1%).
  3. Hits whose minimal pairs confirm the bin hold more often: **yes, within context**. Confirmed held 31/54 (57%), partial 73/113 (65%), not confirmed 46/111 (41%), no check possible 92/213 (43%).
  4. "Most eligible poles have a word hit": **no**, only 47% do. "Most hits land in context or unexplained, not lexical": **yes**, 93% unexplained and 1.2% lexical.
- **Measured cost on the RTX 5080:**
  - hits: 23.3 min at 6,000 tok/s, about 9.6 GB uncapped
  - screen: 28.2 min, of which 25.1 forward, 1.7 fit, and the rest minimal pairs; 8.3 GB
  - prereg: about 3 min on the CPU
  - score: 19.2 min, 5.8 GB under BONSAI_GPU_FRACTION=0.35. A first try at batch 64 ran out of memory and wrote nothing; the rerun used batch 16.
  - export: seconds
  - So about 75 GPU minutes per layer. With the cap, use `--batch 2` for hits and screen.

**Reading**
- **A pole's commonest word is rarely its trigger.** 93% of word hits are unexplained: frequent tokens (indentation, ' ', ' =', digits, '\n', ' is', ' for') that the pole fires on in a median 0.4% of occurrences. They pass lift ≥ 10 only because the poles' base rates are tiny. These are sparse context poles whose context the 19 rule features do not capture. They are the dossier lane for agents: labelled for a context, not a word.
- **Where a cheap rule exists, it holds about half the time on fresh text.** A lexical claim holds almost always. auto-token@1's top-k "token" claims held 5% of the time. The two runs are at different layers (12 and 20), so this is not a controlled comparison, but the gap is large. Running auto-token@1's rule at layer 20 would settle it.
- **The map gets about 560 scrutiny-0 labels from this layer,** in bins (word, context, format, collocation, copy), plus a bin for every screened pole.

**Stage 2's per-occurrence activations** are saved from the next layer on (`occ.npz`). A new context feature then becomes a CPU-only `refit`, which reproduced stage 2 exactly on a smoke run. Layer 20 predates the store. If a future filter needs it, that filter's first step re-runs layer 20's stage 2 with the store on (about 28 GPU min). That leaves the registered results untouched, because the stage is deterministic on the same windows.

**Backup:** the layer's data files are in the `auto2-data` release and its map file in `auto-screens`, with sha256 and sizes in the summary's `artifacts`.

**Layer 4 results** (protocol 13e20da, tests pinned e1546fb; scored with speed-only amendments e1b3ee8, a043447 and 897b2fb; `graph/auto/bonsai1_L4.auto2.summary.json`)
- **Stage 1:** 14,421 of 19,880 eligible poles have a word hit, 33,137 hits in all.
  - Excluded: 2,207 dense, 2,269 weak and 220 positional poles.
  - At layer 20, 47% of eligible poles had a hit. At layer 4, 73% do.
- **Stage 2 bins:**

  | bin | hits |
  |---|---|
  | unexplained | 17,852 |
  | context | 4,986 |
  | lexical | 3,352 |
  | rare | 3,026 |
  | predictive | 1,469 |
  | format | 1,042 |
  | collocation | 794 |
  | topic | 301 |
  | positional | 260 |
  | copy | 55 |

- **Auto tests: 6,337 of 10,088 held (63%), covering 4,670 poles.** The median AUROC was 1.00.

  | bin | held |
  |---|---|
  | lexical | 2,946 / 3,076 (96%) |
  | context | 2,311 / 4,133 (56%) |
  | predictive | 540 / 1,242 (43%) |
  | collocation | 256 / 741 (35%) |
  | format | 272 / 851 (32%) |
  | copy | 12 / 45 (27%) |

- **Examples that held:**
  - lexical: 'RT', ' ihr', ' ت'
  - context: ' again' shortly after the word 'once'; ' edge' right after an ordinary word
  - predictive: ' charged' right before ' with'
  - collocation: ' 입' in a fixed pair
- **Reading:**
  - Layer 4 is far more lexical than layer 20: 3,352 lexical hits against 251, and 4,670 held poles against 562.
  - Lexical claims hold at 96% in both layers. Context claims hold at 56% (49% at layer 20).
  - Format and copy hold less often early. Line-position and repetition rules are weaker claims at layer 4.

**Speed log (auto-token@2 run)**

| unit | measured | bottleneck | change | parity |
|---|---|---|---|---|
| L20 | hits 23.3, screen 28.2, score 19.2 min (batch 8/8/16, mostly uncapped) | – | baseline | – |
| L4 hits | 15.2 min to depth 4 at batch 2 under the 0.35 cap; 14.5k tok/s falling to 10.7k when the CPU was shared; GPU busy 17–63% | the sparse counter (a device-to-CPU copy every batch, frequent CPU re-sorts) and the batch-2 cap | the counter merges on the device and copies to the CPU every 16 batches | counts identical to the old counter on random key streams |
| all-layer build | CPU smoke on layers 2, 3 and 5: 474 s for the passes vs 818 s single-layer | one forward per layer | one forward pass per batch reads every layer | identical outputs; stored activations within 1.9e-6 |
| L4 score | stopped after 8 min | a driver reset at 10:20:24, during steady running with the stream's probe loaded, made the coordinator force-kill the score process; a second reset (TDR, video-engine timeout) followed about 3 s after the kill | every GPU loop now checks data/auto2/STOP at each batch boundary, so a stop never needs a kill; the batch runs only while the stream is offline | – |
| scoring | – | a fixed per-forward cost plus about 34 preamble tokens of every ~82-token sequence | each preamble's KV cache is computed once per layer and reused, so sequences shrink by about 41% and a larger batch fits; joined encodings that split differently (0.9%) take the full path | CPU rescoring of the smoke tests: all outcomes and every activation identical |

Not yet measured: batch size against attention backend under the cap. A 1 GiB fp32 allocation at batch 8 suggests SDPA falls back to its math kernel here.

Two scale facts for the plan:
- Layer 4 made 10,088 auto tests (layer 20 made 1,185), about 160k snippets × 3 preambles. Scoring dominates the early layers.
- L4 is registered (13e20da) and its tests are pinned (e1546fb). Its scoring will be rerun with its registered code, from a worktree at 6d2fb42, once the GPU is free.

## 2026-10-20: What a quick-8+8@3 pass is worth, a null baseline (the Twitch stream; lets-play `scripts/null_baseline.py`, `logs/null-baseline.jsonl`)

**Method**
- Held poles were sampled across depths: 13 per arm.
- For each, a fresh blind author that sees only a label wrote 8 + 8 sentences from (a) the pole's own label and (b) another held pole's label from the same depth band.
- Both were scored on the target pole with the resident model, using the same evaluator as `play run`, and judged by the quick-8+8@3 rule.
- Nothing was committed to the map. The display screen withheld 4 of 30 labels.

**Result**
- **Wrong labels passed 0 / 13** (95% CI 0–23%).
- **True labels passed 11 / 13 on re-test** (85%, CI 58–96%).

| depth band | true labels passed | wrong labels passed |
|---|---|---|
| shallow | 3 / 4 | 0 / 3 |
| middle | 4 / 4 | 0 / 5 |
| deep | 4 / 5 | 0 / 5 |

- **The median-fire bar does the work.** Wrong labels had a median fire of 0.00–0.10; true labels 0.20–1.04. AUROC alone would have passed one wrong label (0.95 on L25/N6966-).

**Reading**
- A held quick test reliably means "not an unrelated label" (false-pass rate likely under about 20%).
- It does not yet show that the label is *precise*. A random wrong label is the easy null.
- **Next:** the harder null, distinguishing a label from a nearby, broader or narrower one, or from a sibling pole's label.

## 2026-10-23: Scores barely depend on batch shape; replication failures were device drift

Two PRs failed CI spot-replication with the outcome reproduced (#677, recorded on Apple MPS; #746, recorded on CUDA).
I measured where the differences come from, on CUDA bf16 through the GPU queue.

- **Batch shape is a small effect.** 40 random recent tests (640 sentences), each scored alone and as one mixed
  batch, as `play run` does: no sentence moved more than 0.03; the largest move was 0.009 and the median of each test's
  largest move 0.003. One outcome flipped with every sentence within 0.01, a test on the pass line.
- **The outlier needs both a large batch and padding.** #677's L22/N2943- sentence ("Nearly 450,000 visitors…") scores
  0.192 alone, 0.191 in a batch of 16 copies of itself, 0.192 padded beside the longest sentence, and 0.231 in the
  mixed, padded batch of 16: the bf16 arithmetic changes with the overall (batch, length) shape.
- **The replication failures were mostly device and precision drift.** CI re-scores on a CPU in float32. #677's
  recorded 0.294 came from MPS (CUDA mixed batch: 0.231). #746's dense L35 pole differs between CUDA bf16 and CPU float32 by
  up to 0.053, against 0.023 between batch shapes on CUDA.

**Consequence.** No change to how tests are scored. The replication check now separates device drift from a mismatch:
a test passes as `drift` when its outcome reproduces and at most 2 of its sentences differ by more than 0.03, none by
more than 0.12 (`bonsai_interp/graph/replicate.py`). Invented numbers still fail: they change the outcome or move many
sentences.

## 2026-10-25: Held is not assignment: most word labels cover a minority of their pole (`bonsai_interp/graph/coverage.py`)

**The bug.** A quick-8+8 test shows that a label's trigger reliably fires the pole: it is a sufficiency test. Nothing
checked coverage, the share of the pole's activity the label accounts for. So L4/N11474− held "the word 'cobra'" while
cobra makes up 0.3% of its 48,246 strong activations; ' Cobra' is the peak in only 3 of its top 32 examples. The user's
verdict: "That's information, not assignment."

**The measure.** Coverage is the share of a pole's strong activations (at least 0.3 of max, in auto-token@2's selection
windows) on the label's trigger tokens. It is exact, computed from the word screen's full (token, pole) tables. Older
labels have no declared trigger, so their triggers were read from the clean word-like terms they quote, but only for
word-shaped labels.

Of 348 held labels measurable this way, 105 cover at least half of their pole and 243 do not. Of these, 242 become
`partial` on main; one is `contested`, a status that takes precedence.

**The rule (the user, 2026-10-25).** A held label is the pole's assignment only if it covers at least 50% of the strong
activations. Below that it is `partial`: kept in the record as a true sub-trigger. Coverage comes from three sources, in
order:
1. a declared token trigger, measured exactly;
2. a coverage round (`play coverage`): 16 random strong activations, one per held-out window, seeded by the label's
   contribution id, each marked "described" or not;
3. the quoted-word estimate.

**Open.** 1,516 held labels describe contexts, not token sets, so they stay `held` with coverage unmeasured until a
coverage round measures them. The deepen queue (`play search --deepen`) lists the partial labels first.

**Caveat.** The quoted-word estimate can be wrong. It undercounts a label that lists its word forms in parentheses
instead of quotes, and it counts a quoted context word as part of the trigger. A new contribution that declares the
trigger replaces the estimate.

## 2026-10-26: What feeds a pole's peak: a token or two, not a few neurons (`bonsai_interp/compose/`)

**Question.** The compositional pass would characterize a neuron by the already-characterized neurons that feed it,
then recurse upward like a sudoku. Is a pole's activation driven by a few upstream units, and how does attention enter?

**Method.** For 60 random eligible poles in each of three depth bands, I took 4 of each pole's strongest corpus examples
(about 238 examples per band) and measured three things:
1. Position-resolved gradient × activation attribution to every earlier (layer, neuron, position).
2. Ablation: zero the top 1, 3 or 10 attributed terms at their positions and measure the activation lost.
3. Token occlusion: replace the token k back with " the", for k = 0..31.

The context was 256 tokens for layers 6–23 and 128 tokens for layers 30–35, because of GPU memory.

**Neuron feeders are diffuse.** A single example's attribution splits into thousands of terms. Their absolute sizes add
up to 35–150 times the activation, and most of it cancels.

| layers | activation lost, top 1 zeroed | top 3 | top 10 |
|---|---:|---:|---:|
| 6–11 | 16% | 37% | 61% |
| 18–23 | 11% | 24% | 46% |
| 30–35 | 14% | 28% | 55% |

(medians)

**Tokens are sharp.** Median share of the activation lost when the token k back is replaced:

| layers | k = 0 | 1 | 2 | 3 | 4 | 8 | 16 | far (10–31) | examples with a far token ≥ 0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6–11 | 0.99 | 0.87 | 0.18 | 0.10 | 0.05 | 0.02 | 0.01 | 0.004 | 17% |
| 18–23 | 0.97 | 0.78 | 0.40 | 0.21 | 0.11 | 0.04 | 0.02 | 0.012 | 40% |
| 30–35 | 0.99 | 0.47 | 0.21 | 0.08 | 0.06 | 0.02 | 0.01 | 0.008 | 21% |

A single token's replacement removes half the activation in 98%, 97% and 88% of examples respectively.

**So at their peaks, poles are mostly local functions:** the read token, the one before it, and (from mid depth) a
little more. That is the clean basis for composition: units at small offsets.

**Attention is handled as positions, not heads.** Occlusion asks which earlier tokens the pole needs, wherever attention
carried them from. Long-range dependence peaks mid-network: 40% of layer 18–23 examples have a token 10–31 back that
matters.

**Peaks are classes, not fixed bigrams.** A pole's top 32 peaks span a median of 22–24 distinct read tokens and 24–28
distinct previous tokens. Example: L10/N5404− fires on "designed | especially", "constructed | specifically",
"tailored | especially", "created | specifically", "específ|icamente".

**The matcher** (`compose/match.py`) names the class at each offset by the upstream poles whose strong activations
cover it, as a greedy union of word units from the word screen's token tables:
- L10/N5404−'s read token is covered 81% by the ' especially' unit (L3/N4610+) plus a ' for'/' to' unit. Its previous
  token, a making verb (designed, created, tailored), has no covering unit: **a gap the loop found by itself**.
- L10/N6727−'s read token is covered by the ' sinus' (L8), ' cheek' (L3) and ' bone' (L1) units: facial anatomy,
  composed from word units.

**Caveat.** These are peaks (the scan's top examples), which favour sharp local triggers. Coverage of the full
strong-activation pool still has to be measured per label.

## 2026-10-26: The first deterministic round, census, twins, grammar, composition, float16 (docs/CENSUS.md, SWEEPLIB2.md, GRAMMAR.md, COMPOSE.md, FLOAT16.md)

Four passes ran over Bonsai-8B without a language model. All were pre-registered (protocol, tests and results each in a
commit of its own) and judged by quick-8+8@2 with placebo tests, beside the live stream.

**Weights-only census** (all 884,736 poles):
- **12% are prediction-type.** Their direct logit effect stands above a sign-random null, and pushing them writes the
  same tokens. The share rises from 0% in layers 0–11 to 34% in layers 30–35.
- **10% steer the output strongly.**
- **550 are frequency-type,** almost all in the last six layers.
- **There are no entropy neurons.** The unembedding has only two near-null directions.
- **Validation:** the predict flag finds the sweep library's held prediction tests at 3.2× the base rate, and survey
  prediction labels at 2.3×.

**Twin batteries** (sweeplib@2), pass 2 seeded by pass 1:
- For 300 seed words, 40,273 tests each varied one axis of a word hit: case, leading space, own token vs piece,
  inflection, sentence position, code slot.
- **20,164 held on 10,715 poles,** against 20 placebo holds (0.05%).
- Most word poles are case- and spacing-blind and fire on whole words.
- **9,825 poles with a partial word result gained a sharper held claim,** and 1,378 now cover at least half of their
  pole.

**Grammar first pass** (sweeplib@3):
- 20 spaCy roles, each tested as the same word in another role. **30 poles held,** 22 of them on the possessive 's,
  against 0 of 326 placebo tests.
- The core argument roles (subject, object, preposition object, determiner, possessor) select no pole at the last
  token.
- Held conditions cover a median 3% of their poles: sub-triggers, not assignments.

**Composition round 1** (compose@1, layers 6–11), the recursive step:
- Poles were explained by already-characterized units at token offsets 0 and 1. The basis comes from the
  occlusion pilot: the read token carries about 99% of a peak, the previous token 47–87%.
- **506 poles held, 334 of them covering at least 50%,** against 0 of 199 placebo tests.
- Pair compositions held 24% of the time. Examples: "ahead of", "I mean", "fill in", "se le", `(` after `return`, "day"
  after a digit.
- Holds fall from 263 at layer 6 to 9 at layer 11. Deeper poles need deeper units, which is round 2's test.

**float16 parity** (for T4s):
- 150 re-scored tests agree on every outcome. Readings are within 0.049 of a pole's max, against bfloat16's own
  batch-shape noise of 0.025.
- No value overflows. float16 is allowed for missions; scans and screens stay in bfloat16.

**Engineering lessons:**
- **One batch model copy on the card at a time.** `gpu_turn.hold_batch_slot` lets several jobs queue beside the stream.
- **Stopping a background agent hard-kills its GPU child.** Doing that mid-work caused event 153 at 22:12 on Oct 25.
  An idle process stopped on Oct 26 caused none.
- **Coverage must be computed layer by layer.** In test order it reloaded a 17M-entry table per held claim, turning
  minutes into hours (sweeplib@2 amendment 1).

## 2026-10-26: Composition round 2 stalls past layer 11 (compose@2, docs/COMPOSE.md)

compose@1's engine ran on layers 13–17, composing from everything characterized after round 1 (twins, grammar,
compose@1).
- **22 poles held of 4,495 tests,** 8 of them assignment-grade, against 0 of 366 placebo tests.
- **Round 1 by comparison:** 506 poles in layers 6–11.
- **The composed condition barely fires the targets:** a median of 0.001 of max on fresh text, against 0.03 in
  round 1. The near misses stay quiet. Past layer ~11, the token classes at a pole's peaks co-occur with it rather
  than drive it.
- **What the next round needs:**
  - offsets taken from each pole's own occlusion map, including far context;
  - context units (language, topic, code slot, steering units);
  - composition over mid-layer units.

The local word-class basis reaches about layer 11.

## 2026-10-27: Part of speech is morphology, early (sweeplib@4, docs/POS.md)

sweeplib@4 screened all 884,736 poles for part of speech and morphology:
- 12 word classes, each against other classes of the same word frequency;
- the same word used in two classes;
- 8 morphological features.

- **313 poles selected, 26 held,** against 0 of 313 placebo tests. 11 of the holds are assignment-grade.
- **The holds are morphological and early:**
  - superlative 10, comparative 3, gerund 3, proper noun 3, third person singular 2;
  - 20 of the 26 are in layers 0–5;
  - superlative poles also appear at L18–24 (L18/N11306+ covers 62% of its top activations, against a 0.3% base rate).
- **No pure noun, adjective, adverb or pronoun pole held** once the frequency band was matched.
- **With the word held fixed, the class it is used in selected one pole** (of 884,736), which failed.

Word class, as such, is not a single-pole feature in Bonsai-8B. Classes marked in the letters are, early. The rest
is either spread over many partial poles or bound to particular words; composition round 3 tests the second reading.

## 2026-10-27: Code syntax poles mark binding sites (sweeplib@5, docs/CODESYN.md)

sweeplib@5 screened all 884,736 poles for 22 Python syntax roles (39 conditions) on a Pygments lex of the corpus's
code windows. The conditions compare the same token in and out of a role, and any token in a role against tokens of the
same type and frequency.

- **442 poles selected; 65 tests held on 54 poles,** against 0 of 442 placebo tests. 20 poles are assignment-grade.
- **The holds are binding sites:**
  - a for loop's variable (31 held tests);
  - the name in a class (18) or def (8) statement;
  - keyword-argument names (4).
  Examples: L26/N1135− (class name, 94% of its top activations), L32/N8525− (loop variable, 89%).
- **Positional roles held nothing:** inside a call, a list, a condition, a string or a comment, and nesting depth.
- **Unlike part of speech, they run deep:** 21 of the holds are in layers 30–35, 18 in layers 0–5, and none in 6–11.

The model dedicates poles to where a program introduces a name, not to where a token merely sits. In prose, the
grammar pass's possessive and relative-pronoun holds point the same way.

## 2026-10-27: Word poles are sense poles (compose@3, docs/COMPOSE.md)

compose@3 sampled every pole's strong activations over 13,593 parsed and lexed windows (64 per pole, uniform). It
then matched each pole against 71 conditions from the deterministic passes: domain, part of speech, morphology,
grammatical role, dependency, question, and code syntax.

A hypothesis was tested only if it already covered at least half of its pole, with a precision of at least 0.1.
Conjunctions ("this word, when it is a noun") also had to show that the condition matters beyond the word.

- **285 of 2,601 tests held on 285 poles, all assignment-grade** (median coverage 0.59), against **0 of 2,518**
  placebo tests.
- **The poles fire hard:** a median 0.34 of max on held-out text, against 0.001 for round 2's word compositions.
- **Most are sense or slot units of a word:**
  - " bear", " warrant" and " license" as nouns; " file", " deal" and " result" as verbs;
  - " help" when negated;
  - " as" in a Python import; " is" in an if-condition; ":" in a def signature; " for" in a comprehension;
  - " you" in a question.
- **By depth:** 133 holds in L0–5, 58 in L6–11, 5 in L12–17, 89 in L18–35.

Many poles the word screen labels "partial" (the word covers under half of the pole) are the word in one sense or
syntactic slot. Put the sense in the label and the coverage passes half. Layers 12–17 remain the hardest band for
every deterministic method so far.

## 2026-10-27: Format and the previous token cover another 320 poles; layers 12–17 stay dark (compose@4)

compose@4 ran compose@3's cover on the same event sample with 222 conditions. The new ones are 13 text-format
conditions, and every compose@3 condition shifted one or two tokens back. Poles compose@3 tested were left out.

- **320 of 1,755 tests held on 320 poles, all assignment-grade** (median coverage 0.63, median fire 0.32), against
  **0 of 1,735** placebo tests.
- **By kind:** conditions one token back 180, format 126, two tokens back 14.
- **Examples:**
  - " other" after a conjunction (L12);
  - "of" after a number (L24);
  - `(self` after a def name (L16, L24);
  - list continuations inside brackets (L14–15);
  - license-header words on `#` lines.
- **By depth:** 239 holds in L0–11, 10 in L12–17, 71 in L18–35.

With compose@3 the cover rounds add 605 assignment-grade poles. Both stall in layers 12–17 (15 holds together). There
the strong activations spread over more read tokens, and fewer poles have any local condition covering half of them.

## 2026-10-28: Mid-layer poles are sparser than any describable class (compose@5 and three diagnostics)

compose@5 added far-context conditions to the cover: a condition held in the last 8 tokens, or unusually often in the
last 32. **22 of 407 tests held** (0 of 405 placebo), 2 of them in layers 12–17. Examples: "..." near a URL; `def`
right after a decorator.

Three diagnostics on the selection split's event sample point the same way:
- **Pairs of conditions with no token set:** never reach a precision of 0.1, at any depth.
- **WordNet supersenses** (noun.person, verb.motion, …): never reach it either.
- **The arithmetic:** a typical mid-layer pole is strong at about 0.01% of positions. A word class, a semantic class or
  a register occurs at 0.5–2%. A pole that fires on only 1 in 100 occurrences of a class fails a held-out 8 + 8,
  however much of the pole the class covers.

Where the deterministic rounds succeed, a set of tokens supplies the precision. In layers 12–17 the strong activations
spread over too many tokens for that. Next: anticipation (what the next token will be), and composition over the
mid-layer units themselves.

## 2026-10-28: Anticipation poles (compose@6)

compose@6 added conditions on the next token and the one after, which the model has not seen. **37 of 650 tests held**
(0 of 646 placebo), 25 of them in layers 18–35.

- **The clearest group: phrasal-verb heads that anticipate their particle.** " carry", " make", " break", " play",
  " find", " figure" and " rule" have poles at L10, L23, L28, L31 and L32 that fire before "out", "up", "down" or "off"
  arrives, and stay quiet on the same verb without a particle.
- **Other anticipations:**
  - " so" before an intensified adjective;
  - an `f` prefix before a string literal;
  - " for" before a loop variable;
  - " is" or " was" before a passive participle.
- **Two late class poles:** the next word is a past-tense verb (L34), or a third-person singular verb (L35).

## 2026-10-28: Untested token detectors, and union covers (compose@7)

compose@7 allowed a two-part label ("X where c holds, or Y") with each part tested on its own. **3,663 of 5,518 poles
were assigned** (0 of 7,267 placebo tests held), which brings the assignment-grade total from 10,379 to 14,053.

- **Most of the yield is one-part word sets: 3,616 of 3,694 held.** 3,600 of the assigned poles had no earlier result
  from any deterministic pass: early-layer token detectors (3,284 in L0–5) that the condition-based rounds never
  reached.
- **Two-part covers rarely hold:** 34 of 1,778 poles had both parts hold. Examples: a spelling with and without its
  leading space, " line" as a verb or " lining", " else" after a conjunction or " otherwise".
- **Deep word poles:** 61 fire on the document boundary token; others on " United", " Santa" and "://".
- **Layers 12–17 again yield almost nothing** (2 assigned).

## 2026-10-28: Token detectors further down the queue (compose@8)

compose@8 tested the next 8,000 one-part covers of compose@7's queue. **6,353 of 7,215 poles were assigned** (0 of
7,207 placebo tests held), which brings the assignment-grade total to about 20,400.

- **Word sets held 6,298 of 7,002**, in two bands:
  - L0–5 (4,096 assigned);
  - L21–29 (1,710 assigned), mostly word-initial pieces with their capitalized forms (" dem"/" Dem", " rem"/" Rem")
    and words with an inflection (" enjoy"/" enjoyed").
- **Layers 12–17 again yield almost nothing** (12 assigned).

## 2026-10-28: The end of the one-part queue, and precision predicts a hold (compose@9)

compose@9 tested every one-part cover left in the queue. **8,958 of 29,044 poles were assigned** (1 of 29,031 placebo
tests held), which brings the assignment-grade total to 29,334 (3.3% of 884,736 poles).

- **The hold rate falls down the ranking,** from 67% in the first 5,000 to 7% in the last 5,000.
- **Selection precision predicts it:** 11% held at a precision of 0.1–0.2, 29% at 0.2–0.3, 51% at 0.3–0.5 and 70% at
  0.5 or more.
- **compose@7–9 assigned 18,974 poles, 18,718 of them by a single word set,** most of which no earlier pass had tested.
- **Layers 12–17:** 61 assigned of 490 tested. They remain the dark band.

## 2026-10-28: The previous token completes the cover (compose@10)

compose@10 tested pair covers: the pole fires on a few (previous token, token) pairs, against the same token after
another token and the same previous token before another. **7,023 of 9,588 poles were assigned** (0 of 9,586 placebo
tests held), which brings the assignment-grade total to about 36,360 (4.1% of 884,736 poles).

- **2,505 of them had failed an earlier cover round:** adding the previous token rescued them.
- **Words split across tokens make up much of the gain.** Where the read token is a word-internal piece, 1,844 of
  2,226 held (" sne|aky", " mon|ologue").
- **Layers 6–11 gave 2,679 poles** and layers 12–17 gave 92, the most any round has found there.
- **Precision predicts the hold again:** 46% at 0.3–0.5, 88% above 1.

## 2026-10-28: Three tokens add little (compose@11)

compose@11 added the token two back to compose@10's pair covers. Only 1,837 poles had such a cover, and **640 of
1,609 tested poles were assigned** (40%; 0 of 1,599 placebo tests held). The assignment-grade total is now about
36,960.

- **What held were mostly fixed phrases:** "insofar as", "above all", "brought about" and "in and out" (L4–7), and
  "even if" after a clause (L22).
- **Number and time patterns** were selected often and held less (175 of 507).
- **The n-gram covers end here.** The read token gave about 19,000 poles (compose@7–9), the previous token another
  7,000 (compose@10), and the token two back 640.

## 2026-10-28: Wider covers, and the end of the cover line (compose@12)

compose@12 allowed covers of 9–16 tokens or pairs, for poles with no smaller cover. **672 of 1,195 tested poles were
assigned** (0 of 1,195 placebo tests held). The assignment-grade total is now about 37,630 (4.3% of 884,736 poles).

- **Some lists name a distinction:** words ending in "ake" and economics terms (L0); male first names, and string
  literals opening on a symbol (L22).
- **The cover rounds compose@7–12 assigned 27,309 poles,** almost all in layers 0–11 and 18–29. Layers 12–17 gave
  under 200 across them. A pole there has no small set of tokens, pairs or triples that covers half of it with
  precision.

## 2026-10-29: Label precision; most word covers are sharp, and low-precision ones don't replicate (compose@13)

compose@13 re-tested the 19,149 poles assigned by a word cover. Each cover was tested against the tokens nearest to it
in the model's input embedding (inflections left out), and again against the frequency-matched null on fresh pairs. 2 of
19,116 placebo tests held.

- **Sharp: 14,829. Broad: 1,096. Unreplicated: 3,224.** Of the 15,925 covers that replicated, 93% fire on their tokens and
  not on the nearest ones.
  - The broad share rises with how close the neighbours are: 3.9% at cosine under 0.10, 40% at 0.16 or more.
  - Broad covers sample a category: pronouns (L24), hedges such as " likely", " probably", " usually" (L6), the nervous
    system and body (L22), US states (L6), number words (L1).
- **Replication falls with selection precision:** 4.9% of covers fail it at precision ≥ 0.5, 27% at 0.3–0.5, 46% at
  0.2–0.3 and 61% under 0.2.
  - compose@9 (the low-precision tail of the queue) did not replicate for 32% of its assignments; compose@7 for 0.7%.
  - One held 8 + 8 test is not enough at low precision: most occurrences of the tokens leave the pole quiet, and the
    test holds when its draw catches the ones that fire.
- **Consequence:** about 3,200 of the 37,629 assignment-grade poles rest on a cover that failed a fresh test. The pair and
  triple covers (compose@10–12) need the same replication before the count is quoted without that caveat.

## 2026-10-29: Pair and triple covers mostly replicate (compose@14)

compose@14 re-tested the 7,818 poles assigned by a pair or triple cover, with their own tests on fresh pairs. **6,941
replicated (89%);** 0 of 7,810 placebo tests held.

- **Pairs fail 9.8% of the time, triples 27%.** Failures rise as selection precision falls (7.6% at ≥ 0.7, 25% at
  0.3–0.5) and with depth (8.5% in L0–5, about 25–30% past L12).
- **Most failures are near misses that fire,** not a silent cover: the same token in another context also fires.
- **With compose@13:** 4,101 of the 26,967 automatic assignments that were re-tested (15%) failed a fresh test. If an
  assignment had to replicate, the assignment-grade total would be about 33,500 (3.8% of poles) instead of 37,629.

## 2026-10-29: Screen labels replicate when they name a word, less when they name a context (compose@15)

compose@15 drew the auto test of the 7,960 screen-assigned poles again, with a new seed and none of the original
windows. 7,552 were testable; **6,704 replicated (89%);** 2 of 6,881 placebo tests held.

- **Lexical labels replicate (1.3% fail, 4,269 poles).** Rule labels fail more: context 18%, predictive 31%, format
  35%, collocation 39%.
- **The failures are rules that are too narrow.** In 755 of 848 the pole still fires on its trigger, and the near
  misses (the trigger where the rule fails) fire too. For these poles "the token X" is the better label.
- **Replication falls with trigger precision** (2.3% fail at ≥ 0.7, 35% at 0.3–0.5) **and past layer 5** (7.1% in L0–5,
  19–24% later).
- **With compose@13 and @14:** 4,949 of 34,519 re-tested automatic assignments (14%) failed a fresh test. If an
  assignment had to replicate, the assignment-grade total would be about 32,700 (3.7% of poles) instead of 37,629.

## 2026-11-02: Embedding classes cover a few hundred more poles; neighbouring tokens mostly don't (compose@16)

The assignment-grade count had been flat at 37,629 since compose@12: compose@13–15 only re-tested assignments, and the
n-gram covers were used up. compose@16 tried three new kinds of cover on every eligible unassigned pole: the next token,
the previous token, and a class of tokens the input embedding groups together.

- **Only 4,390 of 753,574 eligible poles (0.6%) had any such cover** at coverage ≥ 0.5 and precision ≥ 0.2.
- **785 of 4,028 tested held (19%);** 0 of 4,026 placebo tests held. The total is now 38,414 (4.34%).
- **Classes held 30%** (65% at precision ≥ 0.7), for example web formats, Russia and its neighbours, and job titles.
- **Next and previous tokens held about 10%.** In most failures the pole stays quiet on the fire snippets: the
  neighbouring token goes with the pole's activity without bringing it on.
- **The remaining poles fire on many tokens in many contexts.** A fixed library of tokens and conditions doesn't
  describe them at useful precision.
