# Sweep learnings: what would make re-sweeping a new network faster

A running log from the auto-token@2 sweep of Bonsai-8B (from 2026-10-20). It runs on one RTX 5080 that also drives the
display and a live Twitch stream. After every run unit (a layer, an all-layer pass or a scoring chunk), we add:
- what it took
- the bottleneck
- the one change made before the next unit
- the lesson that transfers to a new model (for example Bonsai-27B)

After the sweep this becomes the source for a write-up on re-sweeping a network fast.

## The docket: principles for re-sweeping a network

Distilled from the run log below. Ranked by how much each changed the cost of a sweep, and revised at every
abstraction pass. Each principle names its evidence and whether we have measured the gain.

| # | principle | evidence | gain |
|---|---|---|---|
| 1 | **Share work across layers.** Read every layer in one forward pass. Only per-layer test scoring is inherently separate. | stage 1 one layer at a time: 15–23 GPU min each × 33; all layers in one pass: about 85 GPU min | 6–9× on stage 1 (measured) |
| 2 | **Compute only what is read.** Skip the unembedding and layers past the deepest one read. Reuse shared prefixes: preambles, and near misses that differ in their last token. | lm_head off: 3.05 → 1.8 GB weights; preamble cache: 57 → 80 snippets/s at L20 | 1.4× scoring (measured); prefix sharing pending |
| 3 | **Find the real bottleneck before optimising.** Check utilisation, host vs device, and which kernel runs. | L4 stage 1 at 17–63% GPU (CPU-bound counter); SDPA math fallback for GQA on Windows; 321 vs 207 snippets/s from host overhead; a CPU test cut a GPU pass by 30% | 1.25× attention (measured); others pending |
| 4 | **Budget cost by depth.** Early layers are word-driven and test-heavy; deep layers need context and human effort. | lexical poles L4 2,418 vs L20 171; tests 10,088 vs 1,185; labels hold 86–88% at L0–9 vs 24–50% at L20+ | planning |
| 5 | **Enforce limits at the choke point.** The model loader applies memory caps; every GPU client (helper servers too) takes compute turns, and ONE process computes at a time: foregrounds exclude each other too (fg.lock), holding per model call. | 20 uncapped modules; a 9 GB band scan; 5 resets with concurrent compute clients | 0 resets since the fixes (measured) |
| 6 | **Leave real headroom on the GPU and on the host:** at least 3 GB of GPU memory free, and system commit charge well under its limit. Never build all-pairs index arrays on the host. | reset at 15.5 / 16.3 GB with turns on; stable at 12.4–12.9 GB. The 15:39 reset came with the GPU at 10.6 GB but stage 2 allocating 11–23 GB of host arrays (191M pairs) on a machine already at 59 / 88 GB committed | stability |
| 7 | **Design for stops.** Checkpoint every long stage, stop only at batch boundaries, report progress every minute. Key each checkpoint by what it holds (which layers, which windows), never by the run's settings, so any later run that needs those values can use it. | 29 GPU min lost to an un-checkpointed stop; a force-kill followed by a reset; stage 2's checkpoints hashed the layer set, so a restart after L0 and L1 were written redid the whole 70-min pass; the scorer's checkpoint required the same batch size, so an out-of-memory retry at half batch lost 40 min of L28. Better still, recover inside the process: split only the batch that failed | minutes lost per stop instead of hours |
| 8 | **Load once, serve many.** One resident model per service, commands in-process. | `play why` 28 s → 3.3 s; every reload was a reset risk | 8× latency (measured) |
| 9 | **One pre-registration for many units.** Pin the rule and the test sets before scoring; merge commits keep the hashes valid. | 1 protocol for 33 layers instead of 33 | fewer waits |
| 10 | **Plan wall time by the foreground's duty cycle,** not by GPU minutes. | the stream held 28–70% of the GPU during scoring | planning |
| 11 | **Pick the output format for the full model from the start:** per-layer columnar tables keyed by pole, words as token ids plus one vocab, lazy per-layer views. Check what the target machine allows (application control blocked Parquet here; Arrow IPC worked). | word hits for all 35 layers in about 65 MB (`atlas` release); one JSON file per pole would be gigabytes | scale (docs/SCALING.md) |
| 12 | **Keep the GPU fed: overlap CPU stages with GPU stages.** Run one layer's CPU work (the stage-2 fit, the test pre-registration) while the GPU works on another layer, and spread single-threaded CPU stages across cores within the RAM budget. | stage 2 spends about 3 min per layer on a CPU fit with the batch's GPU share idle (about 1.6 h over 33 layers); prereg-all runs on one core for about 1 h (1–3 min per layer, scaling with tests) before any scoring starts | up to 2.5 h per sweep (estimated) |
| 13 | **Screen kinds with pole-agnostic batteries, not per-pole tests.** A battery of minimal pairs (same last token, condition on vs off) read once at all layers scores all 884,736 poles; add a placebo (quiet vs quiet) to every condition and calibrate the rule on it before registering. | sweeplib@1: 54 conditions in 25 min of selection beside the stream; placebo selections 0 everywhere, placebo tests 0/333 held; a pilot showed the first copy battery asked the wrong question (zoo copy units at AUROC 0.5) before anything was registered | a kind costs ~20 s of GPU, not a sweep |

**One queue, not many locks** (2026-10-20, after 8 resets and 4 partial fixes)
- Each partial mechanism closed one gap and left another:
  - batch vs stream turns
  - the stream's private lock
  - the screen server outside both
  - the yield-before-subprocess rule
- What holds is a single machine-wide queue (`gpu_turn`):
  - tickets ordered by (priority, time)
  - an OS file lock, so exactly one holder, freed by the OS if it dies
  - one model call per hold
  - every GPU client in it, helper servers included (wrapped at the client)
  - an auditor that compares actual per-process GPU compute with the named holder
- **Transfers:** design GPU sharing as one arbiter from day one, with an audit of reality rather than trust in
  callers.
- **Check that the arbiter's files are really shared** (the root cause, found 2026-10-20 by the stream session):
  - Processes started inside an MSIX-packaged app (the Claude desktop app) have their writes to AppData
    redirected to a private copy, while reads fall through to the real files.
  - So the batch, started from a Claude session, saw the stream's tickets and yielded. But the stream, started by a
    supervisor outside the package, never saw the batch's tickets or lock: exclusion only ever worked one way.
  - The fix: put the queue under the user profile root, which isn't redirected. Tested: an in-app holder's lock
    blocks an outside process.
  - **Transfers:** test cross-process locks across every launch path that will use them (app, terminal,
    supervisor), not only between two processes from the same shell.
- **Make the queue the only way in:** two side tools built their own screen client without a ticket. The fix: one
  factory function that always attaches the turn, plus a test that fails if anything constructs a client directly
  (lets-play bdbd9a8).
- **Audit against intervals, not snapshots:** per-process engine counters average over about 1 s, while a screen call
  lasts about 0.2 s. A snapshot of the holder therefore flagged legitimate calls, and periodic calls even made
  "3 in a row". gpu_turn now logs every turn's start and end to `turns.log`, and the auditor accepts any turn
  overlapping the counter window. One stream session reconciled llama-server's request log against the turn log
  to prove it (122 of 130 in bridge turns; the other 8 were a ticketed but unlogged script).
  - A second audit bug: the timestamp was taken *after* the 1–2 s counter sample, so bursts that ended just
    before it fell outside the window. Stamp before sampling, and allow about 5 s. After the fix, 1,365 of 1,365
    screen requests reconciled inside turns. Reconcile against the server's own request log before believing an
    auditor.
  - A third: turns are logged when they *end*, so a turn in progress at the sample isn't in the log yet. The rule
    that works: read the holder before AND after the counter sample, and accept any logged turn overlapping
    [t_before − 1 s, t_after]. Handovers between holders take 2–16 ms.

**Reset record** (nvlddmkm 153, 2026-10-19/20): 8 resets with heavy compute on either side alone, during model loads, and with the probe and the llama.cpp screen overlapping. No bugcheck and no unexpected shutdown in 14 days, including 1,593 driver events on 10-10, so recovery has been reliable on this machine. The crash risk to guard is host-commit exhaustion; resets cost a process.

**Open experiments** (the next speed and effectiveness questions):
- label precision: a held quick test rejects unrelated labels (0 / 13 wrong passed; FINDINGS 2026-10-20 null baseline). Does it also reject a nearby, broader or narrower label? That decides how much a held label says.
- stage 2's cost scales with hits × occurrences (847k × ≤ 256). Would capping occurrences per frequent trigger (e.g. 128) change any bins? It changes results, so it's the user's call; measure on one layer's occ store first
- prefix sharing and batch 32 for scoring (GPU benchmark after stage 2)
- ~~scores depend on batch shape on the GPU~~ resolved 10-23 (FINDINGS): batch shape moves no sentence more than 0.03 in 640 tested (max 0.009); the PR failures were device and precision drift (MPS, CPU float32 vs CUDA bf16). Spot-replication now passes a test as `drift` when its outcome reproduces with at most 2 sentences between 0.03 and 0.12
- the all-layer passes run at forward batch 1 (the driver's hits batch, chosen for memory under the 0.22 cap). Benchmark batch 2–4 for stage 2's 8,000 windows: likely the biggest remaining lever on the 70-minute pass
- the scoring tokens per held pole by depth (a paper metric; L4 about 24.5k, L20 about 172k layer-tokens)
- feeders-first versus cold deep targets for human and agent labels (stream A/B)
- float16 on T4-class GPUs (a parity study task)
- caps on tests per pole versus evidence of multiple jobs per pole (the user's decision; uncapped for now)

**Speed ledger** (one row per unit; the same work should get cheaper each time)

| unit | wall | GPU min | rate | key change since the previous unit |
|---|---|---|---|---|
| L20, per layer (uncapped, alone) | about 75 min | about 75 | stage 1: 6,000 tok/s | (baseline) |
| L4 stages 1–2, per layer | 31 min | about 31 | stage 1: 9,300 tok/s (shallower) | GPU-side counter merge |
| L4 scoring (0.22 cap, beside the stream) | 38 min | about 27 | 207 snippets/s | preamble KV cache, lm_head off, resumable |
| all-layer stage 1 (33 layers, batch 1, beside the stream) | 115.7 min (31 yielded) | about 85 (vs 500–760 one layer at a time) | about 1,300 tok/s × 33 layers | one pass for all layers; memory-efficient attention; +15 min CPU table write |
| all-layer stage 2, attempt 1 | stopped at window 1 of 8,000 (4 min) | – | planned 393 min for 847,211 hits | built 191M (hit, occurrence) pairs on the host at once (11–23 GB of RAM) on a machine at 59 / 88 GB committed; driver reset; rebuilding lean and chunked |
| all-layer stage 2, attempt 2 (500-window chunks) | stopped by the guard at 96% commit after chunk 1 | – | 6.0 min per chunk | llama-server's 8 GiB prompt cache plus the batch; then `--cache-ram 0` and 250-window chunks |
| all-layer stage 2, attempt 3 (250-window chunks) | 21 of 32 chunks in 57 min (18 min yielded) | about 39 | about 2.7 min per chunk | stopped itself at < 2 GB free RAM; resumes from checkpoints under a supervisor |
| all-layer stage 2, attempt 3 resumed | pass done at about 03:41 (chunks 22–32); L0 written in 11.9 min, L1 in 10.0 min | – | fits and minimal pairs about 11 min per layer at template batch 32 | a guard stop at 90% commit (03:18) and one low-RAM relaunch on the way |
| all-layer stage 2, attempt 4 (template batch 64) | pass redone for the 31 layers left: 94.2 min (35 yielded); then L2 4.8, L3 9.9, L5 9.4 min per layer, falling with depth (L6–L11: 6, 7, 9, 5, 7, 4 min) | about 59 for the pass | about 2.1 min per chunk unyielded, at forward batch 1; per layer about the same as at template batch 32 (L0 11.9, L1 10.0), so the minimal-pair batch was not the bottleneck. The log doesn't split the CPU fit from the GPU minimal pairs; the guard's GPU log does (L27, 08:25–08:39): about 3 min of CPU fit at the stream's 12% baseline, then about 11 min of minimal pairs at 91–97%. So the minimal pairs are the bottleneck, and in deep layers they take 12–15 min per layer (L21–L27) against 2–7 min in L13–L19 | restarted at a layer boundary to double the minimal-pair batch; the checkpoint signature hashed the layer set, so the finished pass didn't count (about 70 min lost). Since 5243a0a (speed amendment 1335006) checkpoints count per layer, so a later stop costs at most one chunk |
| all-layer stage 2, attempt 5 (template batch 32 again) | L0–L11 written by attempt 4, which hit a CUDA out-of-memory error in L13's minimal pairs at template batch 64 under the 3.5 GB cap (06:46) | – | relaunched in 1 min: "127,400,593 pairs already read from checkpoints, 0 screening windows" | the per-layer checkpoints saved the 94-min pass on their first use. The driver's out-of-memory retry halves the forward batch, which was already 1; it should halve the stage's inner batch (here the minimal-pair batch) instead. Also: `tee` overwrites each stage's log on a relaunch, so attempt 4's per-layer lines survive only as file times; append instead |
| all-layer stage 2, attempt 5 finished | L13–L35 (22 layers) in 186 min (06:47–09:53) with no stop | – | about 8.5 min per layer: L13–L19 2–4 min, L21–L28 10–15 min, L29–L35 about 8–14 min; 52 min yielded to the stream across the run | stage 2 is done for every layer. From the takeover relaunch (about 23:40 on 10-20) it took about 10 h of wall time, including the 70-min pass redo, a guard stop and a low-RAM stop; before that, attempt 1 (15:39) and an 8-h frozen session |
| prereg-all (33 layers) | 98 min on one CPU core (09:57–11:35), committed alone as 078ce12 | 0 | 1–3 min per layer, scaling with tests | 148,256 tests: L0 20,381, falling to about 500 at L15–L18, rising again to 4,000–4,600 at L24–L29 and 6,234 at L35 |
| all-layer scoring, first minutes (L8) | – | – | about 91 snippets/s beside the stream at batch 16 (L4: 207 at depth 4); 42 snippets per test | the scorer stops the forward at the scored layer, so cost grows with depth: an estimated 30 h for all 33 layers beside the stream (first chunk L8/16/24/28/32 about 5 h; L35 alone about 3 h). Deep layers with many tests dominate |
| scoring L8, L16 | L8 57.2 min (25 yielded), L16 7.3 min (2 yielded) | L8 about 32 | L8: 96,539 snippets × 3 preambles, about 150 snippets/s unyielded; L24 starts at 19.6/s with half its time yielded | held: L8 3,700 / 6,141 (60%, 2,997 poles), L16 237 / 468 (51%, 215 poles). The estimate held for L8; the stream's duty cycle (25–50%) is the biggest swing factor |
| scoring L24, L28 | L24 90.5 min (37 yielded); L28 ran out of GPU memory at batch 16 after 40 min (14:47) and restarted at batch 8 from scratch | – | L24 3,790 tests; L28 resumed at 15 snippets/s overall | held: L24 2,121 / 3,790 (56%, 1,603 poles). Prefix groups run in order of length, so a deep layer's longest batches come last and peak memory there. Since cc56dab (speed amendment c236aab) the scorer splits a batch that runs out of memory and carries on at batch 16 |
| L28 at batch 8 (measured 16:51) | 10,230 of 22,845 batches in 125 min, 55 min yielded (44%) | about 70 | about 19 snippets/s of its own GPU time, against about 56/s for L24 at batch 16: halving the batch cost about 2.5×, far more than the depth difference | the driver keeps the halved batch for the rest of the chunk, so the coordinator stops at L32's first batch boundary and relaunches at batch 16 with the splitting scorer (about 1 GPU h saved on L32, estimated) |
| L28 done; L32 at batch 16 after the switch | L28 197.3 min (85 yielded); switch at 18:02 cost 13 s; L32 2,894 / 5,556 batches by 18:51 | – | L32 about 30 snippets/s of its own GPU time; the scorer split 6 out-of-memory batches in halves and kept going, where the old driver would have restarted the layer at batch 8 | held: L28 2,293 / 3,868 (59%, 1,784 poles). The in-process split works in production: 6 of 6 recovered, 0 restarts |
| scoring chunk 1 (L8, 16, 24, 28, 32) | 11:35–19:13, 7.6 h wall; merged as e443d6c, atlas and map updated by 19:28 | about 4.3 h (3.3 h yielded) | L32 67.6 min (30 yielded) | held 9,492 / 16,152 tests (59%), 7,603 poles. Map refresh after a chunk: atlas build 26 s on the CPU, release upload, Pages redeploy about 1 min, then up to 10 min of CDN cache |
| scoring chunk 2 (L2, 6, 10, 14, 18) | 19:13–21:12, 2.0 h wall; merged, atlas and map updated by 21:20 | about 1.0 h (about 0.9 h yielded) | L2 about 450 snippets/s of its own GPU time, falling with depth | held: L2 4,606 / 6,426 (72%, 3,504 poles), L6 4,240 / 6,135 (69%, 3,523), L10 1,338 / 2,471 (54%, 1,202), L14 377 / 729 (52%, 347), L18 246 / 473 (52%, 211); chunk 10,807 / 16,234 (67%), 8,787 poles. Shallow layers are cheap per test and hold more often |
| driver bug (found 21:13) | chunk 2 skipped L35; after chunk 2 the run repeated it and exited | – | – | `push_branch` and `merge_main` looped with a global `i`, which the chunk loop also used, so every successful push reset the loop to index 1. Fixed in 74a885d (local counters; commit_alone is a no-op when nothing changed) and relaunched in 1 min. Transfer: in bash drivers, declare every helper's variables `local`, and dry-run the chunk plan before a long run |
| guard stop at 21:43 (chunk 3, L22) | the guard wrote STOP at under 2 GB of free GPU memory (14.4 of 16.3 GB used); the batch left at its next turn; relaunched at 22:04 and resumed L22 from its checkpoint (preamble 2 of 3) | – | about 20 min idle | cause: the stream's probe kept its allocator cache at its 5.85 GB peak between about 40 back-to-back solver evaluations (normally 4.3 GB) and released it only at the end of the request. Fix on the stream side (lets-play b2e746f): empty_cache before every turn is released. Transfer: in a shared-GPU queue, every client returns cached blocks when it releases its turn, not when its own job ends; the free-memory guard caught what the turn queue cannot see |
| scoring chunk 3 so far (L22, L26) | L22 47.9 min after the resume (19 yielded), plus about 27 min before the stop; L26 116.5 min (53 yielded) | about 1.5 h | L26 about 51 snippets/s of its own GPU time at batch 16, no out-of-memory splits | held: L22 2,092 / 3,478 (60%, 1,491 poles), L26 2,473 / 4,164 (59%, 1,901 poles). Deep layers hold about as often as mid layers (59–61%); the stream's share stayed at about 45% |
| scoring L30, L34 (chunk 3) | L30 91.9 min (40 yielded); L34 1,801 / 6,690 batches by 02:51 | L30 about 52 | L30 about 43 snippets/s of its own GPU time; L34 has split 16 out-of-memory batches in halves in its first 29 min and kept going at batch 16 | held: L30 1,672 / 2,842 (59%, 1,404 poles). The in-process split is now routine at depth 34 and costs nothing visible; without it each would have been a restart at batch 8 |
| scoring L34 done, L35 under way (chunk 3) | L34 91.1 min (42 yielded); L35 2,704 / 9,798 batches by 04:51 (30 of 58 min yielded) | L34 about 49 | L34 about 37 snippets/s of its own GPU time. Out-of-memory splits: 49 in L34; 109 in L35's first 2,704 batches (4%), 4 of them split twice (8 → 4). At about 4% of batches the failed attempts cost a few percent, against a restart that would halve the batch for the whole layer | held: L34 1,268 / 2,311 (55%, 1,139 poles), the lowest deep-layer rate so far. Next sweep: start the deepest layers at a batch that fits their longest prefix groups, or sort prefix groups so memory peaks early and the split rate is measured first |
| scoring chunk 3 (L35, 22, 26, 30, 34) | 21:14–06:21 wall, 494 min of scoring plus the 21 min guard pause; merged, atlas and map updated by 06:56 | about 4.3 h (about 4 h yielded) | L35 146.5 min | held: L35 1,910 / 4,042 (47%, 1,655 poles), the lowest of any layer; chunk 9,415 / 16,837 (56%), 7,590 poles. Hold rates by depth so far: L1–L2 72–77%, L6–L8 60–69%, L10–L18 51–54%, L22–L32 56–61%, L34–L35 47–55% |
| scoring chunk 4 (L1, 3, 5, 7, 9) | 06:21–09:37, 193 min of scoring | about 1.7 h | about 690 (L1), 350 (L3) and 220 (L5) snippets/s of their own GPU time; no out-of-memory splits | held: L1 6,740 / 8,705 (77%, 4,401 poles), L3 6,800 / 9,716 (70%, 4,732), L5 6,983 / 10,668 (65%, 5,428), L7 2,995 / 4,447 (67%, 2,525), L9 1,778 / 3,173 (56%, 1,568); chunk 25,296 / 36,709 (69%), 18,654 poles: the most productive unit of the sweep. Shallow layers hold more poles per GPU minute than any other depth |
| scoring chunk 5 (L11, 13, 15, 17, 19) | 09:37–10:56, 76 min of scoring; the map for chunks 4 and 5 updated together by 11:05 | about 0.7 h | few tests per layer (436–1,817) | held: L11 937 / 1,817 (52%, 846 poles), L13 583 / 1,129 (52%, 529), L15 293 / 536 (55%, 266), L17 240 / 436 (55%, 219), L19 413 / 718 (58%, 337); chunk 2,466 / 4,636 (53%), 2,197 poles. Mid layers yield few auto tests: most of their poles screen as unexplained, so they are the place for agents and humans |
| L12 is held out, by design (corrected 20:15) | the stream session flagged L12 as missing from every auto-token@2 run, and the coordinator planned a single-layer run; but L12 was the layer used to design auto-token@2 (FINDINGS 2026-10-18: 'it needs a fresh pre-registration on a different layer, because layer 12 was used to look'), and context_tier.py refuses to register it | – | – | L12 keeps its auto-token@1 result only. Transfer: record each held-out layer and why in the sweep plan, so a coverage check reads 'held out', not 'missing'; and check the code's own guards before proposing a run |
| merge workflow fixed (10-23, #783) | GitHub keeps one pending run per concurrency group, so bursts of finished checks silently dropped merges (#371/#373, a 40-PR backlog) | – | – | merge.yml now sweeps every open PR on each run, plus a 30-minute schedule and a dry-run dispatch. Transfer: an event-per-item workflow under a concurrency group loses events; make each run reconcile the whole state |
| scoring chunk 6 (L21, 23, 25, 27, 29) | 437 min of scoring (11:00–18:14); merged and on the map by 18:58. L21 48.0 min (21 yielded), L23 86.5 (38), L25 119.2 (56), L27 92.6 (29), L29 90.2 | about 4.3 h | about 64 (L21), 59 (L23), 53 (L25) and 50 (L27) snippets/s of their own GPU time; no out-of-memory splits through L27 | held: L21 1,260 / 2,134 (59%, 961 poles), L23 2,066 / 3,619 (57%, 1,530), L25 2,407 / 4,256 (57%, 1,842), L27 2,380 / 4,035 (59%, 1,857), L29 2,164 / 3,697 (59%, 1,719); chunk 10,277 / 17,741 (58%), 7,909 poles. Tooling note: stopping the supervisor's background shell left its script running, so two supervisors watched the driver from 08:51 to 12:04 (neither acted). Replace a watcher by giving it a stop file or a pid file, and check the process list afterwards |
| scoring chunk 7 (L31, 33, 0) | L0 15.4 min (5 yielded), L31 49.3, L33 47.1; merged 20:08 and on the map by 20:14 | about 1.3 h | about 1,400 snippets/s of its own GPU time at depth 0 | held: L31 1,293 / 2,109 (61%, 1,120 poles), L33 1,015 / 1,802 (56%, 907), L0 15,712 / 18,563 (85%, 8,659 poles): the most poles and the highest hold rate of any layer, for 10 GPU minutes. The ledger's clearest transfer: score the shallowest layers first on a new network; they are nearly free and fill most of the map |
| **the all-layer sweep, complete** | stage 2 from the 10-20 takeover; scoring 10-21 11:35 to 10-22 20:08 (about 32.5 h wall beside the stream) | – | – | **35 of 36 layers** under auto-token@2 (L12 held out as the design layer): 92,792 of 142,056 auto tests held (65%), 68,658 poles held (8.0% of those layers' poles), every layer on the map and in the atlas release. Hold rate by depth: L0–L3 70–85%, L5–L8 60–69%, L9–L19 51–58%, L21–L33 56–61%, L34–L35 47–55% |

Entry format:

```
### <date> <unit>
- measured: minutes per stage, tokens/s, GPU utilisation, peak memory, tests/snippets
- bottleneck: …
- change before the next unit: … (parity check: …)
- gain of the previous change: … (measured)
- transfers: …
```

## Lessons so far (before the all-layer passes)

**Compute plan**
- **Share passes across layers.** One forward pass can record every layer at once, so per-layer corpus passes
  repeat the same work. auto-token@2 ran stage 1 and stage 2 once per layer: 34 passes × about 50 GPU min.
  All-layer passes cut that to about 2–3 h. Scoring cannot be shared (each layer's snippets differ), so it becomes
  the long pole.
- **Test counts scale with how lexical a layer is.** L20 produced 1,185 auto tests; L4 produced 10,088
  (158,857 snippets × 3 preambles, about 2.5 h of scoring). Budget scoring per layer from stage 2's bin counts.
  Decide a cap on tests per pole before registering, not after.
- **Measure utilisation, not just time.** L4's word-hit stage ran at 17–63% GPU utilisation. It was CPU-bound on
  a sparse counter; moving the merge onto the GPU fixed it. A cap that forces batch 2 on 1,024-token windows also
  points to the attention kernel: check which SDPA backend actually runs.

**One display GPU, a live stream and a batch**
- **Memory caps are necessary but not sufficient.**
  - What works: `BONSAI_GPU_FRACTION` per checkout makes an over-budget process fail with an OOM instead of
    spilling.
  - Size a cap from the load peak, not the resident size: a Bonsai-8B load peaks at about 5.2 GB (the vocabulary
    matrix lands on top of 4.0 GB), while it holds 4.1 GB loaded.
  - What failed: five NVIDIA driver resets (nvlddmkm 153) in two days.
    - Four came as a second process started or loaded the model while another was computing.
    - One (10:20:24) came with NO stream model on the card. Only the batch was computing, beside the
      stream's llama-server chat screen, the NVENC video encoder and a captured browser page.
    - Two resets on the first night had no batch at all, just the screen, NVENC and a stream model job.
    - So the common factor is heavy torch work beside *any* other GPU compute client, helper servers
      included, not two copies of the model.
- **What makes sharing safe: turn-taking on compute, not just budgets on memory.**
  `bonsai_interp/gpu_turn.py`:
  - Both models stay loaded.
  - The foreground (the stream) takes the GPU, and the batch yields at its next batch boundary.
  - Only one process runs kernels at a time.
  - Every compute client belongs in it, including helper servers such as a llama.cpp screen: wrap their
    calls from the client side if the server itself can't take part.
- **Load once.** Every subprocess that reloads the model is a reset risk and costs about 25 s. Serving commands
  from one resident model per session removed both. The stream's `play why` dropped from about 28 s to 3.3 s.
- **Never force-kill a GPU process mid-kernel.** A Stop-Process during scoring was followed 3 s later by another
  reset and a TDR. Stop at batch boundaries (a STOP file).
- **Verify a "release" from the allocator, not from the caller.** A probe reported that it had yielded while still
  holding 4.4 GB. A stale model reference meant a second copy loaded beside it.
- **A smaller helper model frees room.** The stream's chat screener on Bonsai-8B (1.9 GB) instead of Bonsai-27B
  (4.3 GB) returned 2.4 GB to the budget.

**Caps must be enforced where the model loads, not where callers remember them** (2026-10-20, 12:23 reset)
- About 20 modules loaded the model without calling `limit_memory()`. One was the stream's band scan, which
  grew to 9 GB against a 5.7 GB budget.
- An orphaned scan (left behind by a tool timeout) also kept its turn request alive for about 30 minutes. So
  the batch yielded 70% of its run while the card sat above 15 of 16.3 GB, and the driver reset.
- Fixes:
  - `model.load()` applies the cap itself (861621a).
  - Budget for at least 3 GB of free headroom.
  - Make long scoring resumable, so a stop costs minutes, not hours.
- **Transfers:** put resource limits and turn-taking in the one choke point every job passes through (the
  model loader), and treat any process a timeout leaves behind as a resource leak.

**Don't load what the job never reads** (2026-10-20)
- Bonsai-8B's untied lm_head expands to a dense 1.24 GB bf16 matrix. Activation screening never uses it, yet
  it was 40% of the batch's 3.05 GB of weights.
- `model.load(lm_head=False)` (fe17bd1) leaves it on the meta device: about 1.8 GB resident, with identical
  scores. That let the batch fit a 3.6 GB budget beside a live stream.
- **Transfers:** for activation-only sweeps, skip the unembedding, and also any layers deeper than the
  deepest one being read. On a larger-vocabulary or wider model (e.g. 27B), the saving grows with
  vocab × hidden.

**On Windows, GPU memory also costs host commit** (2026-10-20, measured by the stream)
- Under WDDM, every GiB torch allocates on the GPU adds about a GiB of host commit charge: Windows keeps
  system-memory backing so the allocation can be evicted. Measured at +1.0/+2.0/+3.0 GiB of commit for 1/2/3 GiB
  of device tensors.
- **A torch process's CPU-side baggage:**
  - torch itself: about 1.6 GiB
  - transformers and other imports: about 0.9
  - the CUDA context: about 0.5
  - plus any data it loads, e.g. a 1 GiB corpus scan
  - plus about 2–3 GiB that Python can't see: CUDA libraries such as cuBLAS, loaded lazily on first use, and
    native load-time leftovers. This grows over the first hours of use.
- **A measured resident service:** the stream's probe commits about 11 GiB.
  - Accounted for: the GPU backing 4.2, torch 1.6, imports 0.9, the scan 1.1, the CUDA context 0.5, the corpus 0.2.
  - Unaccounted: about 2.6.
- **Budget:** host commit ≥ the GPU allocations + 4–6 GiB per torch process + its data. A full 16 GB card can
  cost 16 GB of commit on its own.
- *Withdrawn* (an earlier entry was wrong): freed host memory IS returned to Windows; a freed 1 GiB array drops
  commit by 1 GiB. An apparent high-water mark was an `lru_cache` keeping the scan alive.
- **Transfers:** on Windows, watch the commit charge as closely as VRAM, and keep the number of torch processes
  small.

**Helper servers grow silently: check their defaults** (2026-10-20, 23:47)
- llama-server's host-RAM prompt cache defaults to 8 GiB. The stream's 1-bit 8B screen grew from 5.8 to 7.4 GB of
  commit over the day, and with the batch's stage 2 on top, commit reached 88% with a 96% spike. The guard stopped
  the batch cleanly.
- `--cache-ram 0` (the slots still reuse the system prompt's KV) plus one probe restart took baseline commit from
  79% to 65%.
- Stage 2 in 250-window chunks instead of 500 cut the batch's RSS from 5.6 to 4.2 GB, and commit held at about
  80–82% with the batch running.
- **Transfers:** before a long shared run, read every helper server's memory defaults (caches, slots, context),
  and set a per-process commit budget alongside the GPU one.

**Plan by depth: early layers are word-driven, deep layers are not** (2026-10-20, L4 vs L20)
- **Auto tier:**
  - L4 binned 2,418 poles as lexical, against 171 at L20.
  - L4 held auto tests on 4,670 poles, against 562 at L20.
  - L4's lexical tests held 96% of the time.
- **Human and agent labels on the map:** they hold 86–88% at L0–9, against 24–50% at L20 and deeper.
- **Cost:** the auto tier's test count, and so its scoring cost, peaks in early layers (L4: 10,088 tests).
  Deep layers need richer context features (or upstream feeders) instead.
- **GPU share:** a live agent stream held the foreground for about 28% of a scoring run (649 s of 38 min),
  so budget about 1.4× wall time for a background job sharing with it.
- **Transfers:** on a new model, budget scoring time by depth, aim the cheap lexical screen at early layers,
  and spend the human and agent effort on deep layers. The stream is A/B testing "feeders first" against
  cold deep targets.

**Method notes for interpreting deep poles** (the stream's inverse lab, 2026-10-20; one lexical L20 pole so far)
- **MLP feeders explain little:** the 20 most-attributed upstream MLP neurons carry only 27% of the pole's
  activation; attention carries the rest. Gain-on-feeder inverses were inconclusive there. The context (token)
  analysis was the informative one.
- **Ablate embeddings by blending, not scaling:** scaling a token's embedding is mostly undone by RMSNorm, so blend
  it toward the mean embedding.
- **Transfers:** measure how much of a pole an input set can actually move ("authority") before reading any inverse
  built on it. For deep poles, budget attention-path analysis, not only MLP feeders.

**From the stream side** (the Twitch session, 2026-10-20)
- **A load gate checked at process start is too early.** Python imports take 10 s or more, so the gate
  passed before the batch had allocated. Check again immediately before the allocation, or hold a turn for
  the process's whole life.
- **A loader swapped in for an in-process command can pin the model.** It stayed bound in a module that was
  first imported during that command. Restore every loader binding after the run, and check the freed memory
  with the allocator's own count.
- **Keep secrets out of resident services.** The solver that needs an API key (globalMOO) stays a
  subprocess, which holds a GPU turn for its whole life. The long-lived service an agent can reach never
  holds the key.
- **Watchers must survive log rewrites.** A supervisor that rewrites its log on restart breaks a watcher that
  reads only new bytes. Re-read the whole file and take the last marker.
- **Turn-taking in practice:** with a background holding the GPU, a foreground `play why` waited less than
  one background batch, and the background yielded for the whole call (1.9 s).

**Process**
- **A self-stopping job needs a supervisor that restarts it.** Stage 2's RAM guard stopped it cleanly at chunk 21 of
  32 when free RAM fell under 2 GB (00:55). Nothing was watching the driver, so the GPU sat idle for 2 h. Now a
  supervisor relaunches after a low-RAM stop once 6 GB is free again; the run resumes from its chunk checkpoint. It
  wakes the coordinator on finish, on STOP, or on any other failure. Physical RAM (31.7 GB, about 7 GB free beside the
  stream) is the binding limit, not GPU memory or commit.
- **Every wait needs a timeout and a structured marker.** One agent waited 5 h (15:34–20:51) on
  `until grep "screen windows needed"` while the log said "screening windows needed". The session sat blocked behind
  it, with seven coordinator messages queued, and the batch stood idle. Write explicit markers (`=== stage2 done`,
  `=== failed`) and wait on those, with a deadline that reports instead of looping. A coordinator should check that
  each worker is actually progressing (last commit and activity time), not just "running".
- **One registration for many layers is as valid as one per layer,** and much faster: fewer commits and fewer
  waits. What matters is that the rule and the test sets are pinned before scoring.
- **Keep per-occurrence activations from the screen** (`occ.npz`). A new context filter is then a CPU refit
  instead of another GPU pass.
- **Merge with merge commits, never squash or rebase,** so the registration hashes cited in summaries stay valid.

## Run log

### 2026-10-20 L4 score trial (beside the live stream, gpu_turn on)
- measured: 99 min of wall time for 25,054 of about 29,800 batches (84%), at batch 16, BelowNormal priority, the
  0.35 cap and L4's registered (uncached) scorer. 4,187 s (70%) was yielded to the stream's foreground turns, so
  about 29 GPU minutes did 84% of the work: a full L4 score is about 35 GPU minutes.
- bottleneck:
  - The stream held the GPU 70% of the time.
  - A driver event (nvlddmkm 153, 12:23:29 local) made the watcher create STOP. The scorer exited at the next batch
    boundary, 3 s later, with no kill and no follow-on reset.
  - Scoring writes nothing until a layer ends, so 29 GPU minutes were lost.
- change before the next unit: scoring checkpoints its partial sums every few hundred batches and resumes from
  them, so a stop loses minutes, not a layer (parity check: a stopped and resumed score must equal an uninterrupted
  one).
- gain of the previous change: the graceful stop worked as designed (STOP to exit in 3 s, the turn released).
- memory: the card held more than 15 of 16.3 GB for 70 of the 99 minutes. The reset came at about 15.5 GB, with
  turns on. Near-full memory on the display GPU is the leading suspect.
- further changes before the next unit:
  - Return the allocator's cached blocks (torch.cuda.empty_cache) before every yield, so the stream can use what
    the batch is not holding.
  - A progress line every minute (batches done / total, snippets/s, yield seconds). The trial printed nothing for
    99 minutes.
  - The cap gets sized from the benchmark instead of 0.35: the batch stays at or under about 4 GB, and the whole
    card at or under 13 GB.
- transfers:
  - On a shared display GPU, budget wall time by the foreground's duty cycle (here 70%), not by GPU minutes.
  - Budget memory to leave real headroom on the card (at least 3 GB): a turn scheme stops two processes computing at
    once, not the card filling up.
  - Stops are routine, so every long stage must checkpoint and report progress. Design both in from the start, not
    after a lost run.

### 2026-10-20 fit check and L4 score (beside the live stream, cap 0.22)
- measured:
  - The fit check under BONSAI_GPU_FRACTION=0.22 (3.5 GiB): resident weights 2.88 GB with lm_head left off.
  - Scoring peaks were 3.10–3.30 GB in every configuration.
  - Chosen: the preamble-cached scorer at batch 16. Benchmark: 321 snippets/s at layer 4 and 80 at layer 20 (the
    full scorer did 232 and 57). Peak 3.17 GB, about 0.3 GiB under the cap.
  - L4 score: 476,571 sequences (10,088 tests × 16 snippets × 3 preambles) in 38.3 min wall at 207 snippets/s.
    649 s was yielded, about 27 GPU minutes. No reset, no OOM, nothing lost.
- bottleneck: per-batch Python overhead (copying the preamble cache, building tensors). The real run made 207
  snippets/s against the benchmark's 321.
- change before the next unit: none in scoring (it fits and runs). Next, measure the 1,024-token passes under
  0.22 before registering the all-layer run.
- gain of the previous changes:
  - lm_head off the GPU: peak 5.78 → 3.17 GB at batch 16.
  - The preamble cache: 1.38× (layer 4) to 1.4× (layer 20) in the benchmark.
  - Resumable scoring: not exercised (no stop).
  - The memory fixes on both sides: yield fell from 70% of wall time to 28%, and the card stayed under budget.
- transfers:
  - Load only what the job reads. An untied output matrix is the second-largest tensor in a small model, and
    activation-only jobs never touch it.
  - Cache what every sequence shares (the preambles).
  - Size the memory cap from a measured peak (weights plus the batch), not a guess.

### 2026-10-20 benchmark of the 1,024-token passes (cap 0.22)
- measured: at layer 20, batch 1, 3,481 tok/s with a 3.35 GB peak. Batch 2 runs out of memory under 3.5 GiB.
- bottleneck: SDPA fell back to its math kernel, which materialises the 32 × 1,024² attention matrix.
  - Transformers passes grouped-query attention as `enable_gqa=True` with 8 KV heads against 32 query heads.
  - This PyTorch build (Windows, CUDA) refuses its fused kernels for unequal head counts and has no flash attention.
- change before the next unit: repeat the KV heads before SDPA (`use_gqa_in_sdpa` → False, inside the auto tier
  only). The memory-efficient kernel then runs: 4,347 tok/s (+25%), peak 3.25 GB. Batch 2 still does not fit, so the
  all-layer passes run at batch 1.
  (parity: CPU counts identical; on the GPU the kernel change can move bf16 rounding. The run is not registered yet,
  so this is part of its registered code rather than an amendment.)
- transfers:
  - Check which attention kernel actually runs (the warnings name the reason), not the one the config asks for.
  - "GQA support" in a library can silently mean the slowest kernel on a given platform.
  - Under a tight memory cap, kernel choice decides the usable batch size.

### 2026-10-20 note during the all-layer stage 1 pass
- measured: the pass ran at 1,619 tok/s (batch 1, 33 layers counted per forward). A CPU parity test running beside
  it cut that to about 1,100 tok/s, a 30% loss. The pass is host-bound: per-layer thresholding, bincounts and
  merges run on the CPU between kernels.
- change: CPU-side work (tests, refits, prereg) runs only in the gaps between GPU passes, or limited to 2 threads.
- transfers: a host-bound GPU pass loses about 30% to any concurrent CPU work. Schedule CPU jobs between passes, and
  for the next model move more of the per-layer bookkeeping onto the device.
- 2026-10-27: `gh pr checks --watch` returns at once when no checks have registered yet (a fresh PR): wait until `gh pr checks` lists them before watching, or a code PR merges unchecked (#1183).
- 2026-10-28 10:29: 103 driver resets (nvlddmkm 153) in 90 s, then a reboot at 10:39, during the stream's probe-eval GPU turn while a CPU-heavy compose@7 build (360-condition masks) ran beside it. The batch job held no GPU turn; cause unproven. A chain killed mid-build left a tests file without its prereg manifest: rebuild deterministically and pre-register before any scoring.
- 2026-10-30: `graph.validate` takes its root from the imported package, so validating an archive of main with PYTHONPATH pointing at a local checkout validates that checkout instead (the count sat at 5,911 through dozens of merges). Archive the whole tree (`git archive origin/main`, 112 MB) and run with PYTHONPATH set to the archive.
- 2026-10-30: pages.yml can fail at deploy-pages with "Failed to create deployment (status: 500)" (a GitHub-side error; the build is fine). Re-run the failed job (`gh run rerun <id> --failed`); it succeeded on the first retry.
- 2026-11-02: to add a new deterministic pass to the atlas, run `python -m bonsai_interp.atlas add-lib`, not `atlas build`: `build` in the main checkout finds no screen exports (they live in the sweep worktree's data/auto) and writes census_flags as null, so the assigned count fell from 37,629 to 31,971 until the backup was restored. Back up data/atlas first and diff the columns before uploading.
