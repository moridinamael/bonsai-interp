"""Build site/steering.html: the globalMOO steering rounds (steer@3 to steer@10) as one page.

  python site/steering.py [out.html]        (default: site/steering.html; the Pages build writes _site/steering.html)

Every figure is copied from FINDINGS.md and the rounds' committed reports (bonsai_interp/moo/steer*_summary.txt); each round
links to its pull request. No script, no fetch: inline SVG and CSS bars, the map's fonts and colour tokens, light and dark.
"""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/moridinamael/bonsai-interp"


def esc(s):
    return html.escape(str(s), quote=True)


def pr(n):
    return f'<a href="{REPO}/pull/{n}">PR {n}</a>'


# Held-out prompts, globalMOO's designs against each baseline's: (won, lost, tied) over the round's cells or profiles.
ROUNDS = [
    ("steer@3", 1809, "one or two topics · maximize queries", (13, 11, 0), (17, 7, 0)),
    ("steer@3b", 1811, "same cells · value targets", (18, 6, 0), (21, 3, 0)),
    ("steer@4", 1812, "four new topics · a new model", (21, 3, 0), (14, 10, 0)),
    ("steer@5", 1813, "32 training prompts", (21, 3, 0), (18, 6, 0)),
    ("steer@6", 1814, "whole profiles · seven goals", (18, 6, 0), (24, 0, 0)),
    ("steer@7", 1815, "reachable profiles", (11, 4, 1), (12, 4, 0)),
    ("steer@8", 1816, "64 training prompts", (11, 5, 0), (11, 5, 0)),
    ("steer@9", 1817, "steer@8's designs · 64 fresh prompts", (9, 6, 1), (12, 4, 0)),
]

# Whole profiles: mean worst miss on unseen prompts, log10 (lower is better). gmoo, lookup, additive, de.
PROFILES = [
    ("steer@6", "targets set beyond the panel's joint reach", (0.69, 0.70, 0.95, 1.55)),
    ("steer@7", "targets the design shows are reachable", (0.49, 0.63, 0.59, 1.20)),
    ("steer@8", "64 training prompts", (0.445, 0.467, 0.576, 0.541)),
    ("steer@9", "steer@8's designs on 64 fresh prompts", (0.423, 0.480, 0.431, 0.537)),
]
METHODS = ("globalMOO", "best design point", "additive model", "differential evolution")
PMAX = 1.6

# Every registered claim, in order: round, PR, the claim, held?, the result.
CLAIMS = [
    ("steer@3", 1809, "Steering split by depth beats the uniform knob by 2× on 3 of 4 single topics", False, "2 of 4"),
    ("steer@3", 1809, "globalMOO beats the additive model in more held-out cells than it loses", True, "13 to 11"),
    ("steer@3b", 1811, "Value-target queries beat the best design point in at least 8 of 24 training cells", True, "17 of 24"),
    ("steer@4", 1812, "On four new topics, globalMOO beats the additive model in more held-out cells than it loses", True, "21 to 3"),
    ("steer@4", 1812, "Queries beat the best design point in at least 8 of 24 training cells", True, "12 of 24"),
    ("steer@5", 1813, "On unseen prompts, queries beat the best design point in more cells than they lose", True, "18 to 3, 3 tied"),
    ("steer@5", 1813, "globalMOO beats the additive model in more held-out cells than it loses", True, "21 to 3"),
    ("steer@6", 1814, "Whole profiles: a smaller worst miss than the additive model in more held-out cells than not", True, "18 to 6"),
    ("steer@6", 1814, "Every topic within 1.5× of its target in at least 12 of 24 training cells", False,
     "0 of 24, for every method: the targets were out of joint reach"),
    ("steer@7", 1815, "Reachable profiles: every topic within 1.5× in at least 8 of 16 training profiles", True, "13 of 16"),
    ("steer@7", 1815, "A smaller held-out worst miss than the additive model in more profiles than not", True, "11 to 4, 1 tied"),
    ("steer@8", 1816, "With 64 training prompts, the mean held-out worst miss falls below steer@7's 0.49", True, "0.445"),
    ("steer@8", 1816, "A smaller held-out worst miss than the additive model in more profiles than not", True, "11 to 5"),
    ("steer@9", 1817, "On 64 fresh prompts, globalMOO's mean worst miss is below the additive model's, in more profiles than not", True,
     "0.423 against 0.431; 9 to 6, 1 tied"),
    ("steer@9", 1817, "The gap from training to fresh prompts is below the gap to the 16 held-out prompts", True, "0.414 against 0.428, narrowly"),
    ("steer@10", 1818, "Scored per prompt, two halves of the fresh prompts agree better, in at least 3 of 4 topics", True, "4 of 4"),
    ("steer@10", 1818, "Scored per prompt, training and fresh prompts agree better, in at least 3 of 4 topics", True, "4 of 4"),
]


def split_bar(w, l, t):
    parts = [f'<i class="won" style="flex:{w} 1 0" title="won {w}"></i>']
    if t:
        parts.append(f'<i class="tie" style="flex:{t} 1 0" title="tied {t}"></i>')
    if l:
        parts.append(f'<i class="lost" style="flex:{l} 1 0" title="lost {l}"></i>')
    return f'<span class="split" role="img" aria-label="won {w}, lost {l}{f", tied {t}" if t else ""}">{"".join(parts)}</span>'


def scoreboard(col, title, note):
    rows = []
    for name, n, what, *res in ROUNDS:
        w, l, t = res[col]
        tied = f' <span class="tied">· {t} tied</span>' if t else ""
        rows.append(f'<li><span class="rname"><a href="{REPO}/pull/{n}">{esc(name)}</a><small>{esc(what)}</small></span>'
                    f'{split_bar(w, l, t)}<span class="score"><b>{w}</b> – {l}{tied}</span></li>')
    return (f'<div class="board"><h3>{title}</h3><p class="sub">{note}</p><ol class="rows">{"".join(rows)}</ol></div>')


def profile_chart():
    out = []
    for name, what, vals in PROFILES:
        rows = []
        for i, (m, v) in enumerate(zip(METHODS, vals)):
            cls = "gm" if i == 0 else "base"
            rows.append(f'<li class="{cls}"><span class="mname">{esc(m)}</span>'
                        f'<span class="track"><i style="width:{100 * v / PMAX:.1f}%" title="{esc(m)}: {v:.2f} log10, a factor of {10 ** v:.1f}"></i></span>'
                        f'<span class="val">{v:.2f}<small> ×{10 ** v:.1f}</small></span></li>')
        out.append(f'<div class="pround"><h3>{esc(name)}<small>{esc(what)}</small></h3><ul class="prows">{"".join(rows)}</ul></div>')
    return "".join(out)


def ledger():
    rows = []
    for name, n, claim, held, result in CLAIMS:
        tag = '<span class="tag held">held</span>' if held else '<span class="tag failed">not held</span>'
        rows.append(f'<tr><td class="mono"><a href="{REPO}/pull/{n}">{esc(name)}</a></td><td>{esc(claim)}</td><td>{tag}</td><td>{esc(result)}</td></tr>')
    return "".join(rows)


def lesson(num, title, before, after, body):
    def bar(label, frac, text, cls):
        return (f'<li class="{cls}"><span class="mname">{label}</span><span class="track"><i style="width:{100 * frac:.1f}%"></i></span>'
                f'<span class="val">{text}</span></li>')
    return (f'<article class="lesson"><p class="num">{num}</p><h3>{title}</h3><ul class="prows">{bar(*before, "base")}{bar(*after, "gm")}</ul>'
            f'<p>{body}</p></article>')


SETUP_SVG = """<svg viewBox="0 0 920 262" role="img" aria-labelledby="setup-t" class="diagram" style="--w:920px">
<title id="setup-t">Twenty-four knobs steer Bonsai-8B's writing, twenty-seven readings come back, and globalMOO answers a seven-goal request with one setting of all the knobs</title>
<defs><marker id="arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L9,5 L0,9 z" class="head"/></marker>
<marker id="arrm" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,1 L9,5 L0,9 z" class="head m"/></marker></defs>
<g class="box"><rect x="20" y="22" width="250" height="112" rx="8"/>
<text x="38" y="48" class="k">INPUTS</text><text x="38" y="74" class="t">24 knobs</text>
<text x="38" y="96" class="d">4 topics × 6 depth bands</text><text x="38" y="115" class="d">a gain on each bundle of labelled neurons</text></g>
<g class="box"><rect x="335" y="22" width="250" height="112" rx="8"/>
<text x="353" y="48" class="k">THE MODEL</text><text x="353" y="74" class="t">Bonsai-8B writes</text>
<text x="353" y="96" class="d">24 new tokens after each of 64 prompts,</text><text x="353" y="115" class="d">with those neurons pushed up or down</text></g>
<g class="box"><rect x="650" y="22" width="250" height="112" rx="8"/>
<text x="668" y="48" class="k">OUTPUTS</text><text x="668" y="74" class="t">27 readings</text>
<text x="668" y="96" class="d">24 topic rates · distinct tokens ·</text><text x="668" y="115" class="d">coherence · drift of the other topics</text></g>
<line x1="270" y1="78" x2="331" y2="78" class="edge" marker-end="url(#arr)"/>
<line x1="585" y1="78" x2="646" y2="78" class="edge" marker-end="url(#arr)"/>
<g class="box gm"><rect x="335" y="184" width="250" height="62" rx="8"/>
<text x="353" y="209" class="t">globalMOO inverse model</text><text x="353" y="230" class="d">trained once, on 687 knob settings</text></g>
<path d="M775,134 V215 H589" class="edge m" marker-end="url(#arrm)"/>
<path d="M335,215 H145 V138" class="edge m" marker-end="url(#arrm)"/>
<text x="682" y="206" text-anchor="middle" class="lab">ask: 7 goals at once</text>
<text x="682" y="234" text-anchor="middle" class="lab2">4 topic targets + 3 limits</text>
<text x="240" y="206" text-anchor="middle" class="lab">answer: one setting</text>
<text x="240" y="234" text-anchor="middle" class="lab2">of all 24 knobs</text>
</svg>"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Steering with globalMOO</title>
<meta name="description" content="Labelled neurons in Bonsai-8B as 24 steering knobs, solved as one multiobjective problem with globalMOO: nine pre-registered rounds.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Public+Sans:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root { color-scheme: light;
  --paper: #eef2ef; --sheet: #f8faf8; --ink: #17302c; --ink-2: #44605a; --ink-3: #7b938d; --rule: #d3ddd9;
  --mark: #2d3fc6; --held: #1f7a4d; --failed: #b4441f; --base: #9fb1ac; --wash: rgba(45, 63, 198, 0.07);
  --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", sans-serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", "Cascadia Mono", Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --paper: #0f1716; --sheet: #15201e; --ink: #e3ece9; --ink-2: #a8bcb6; --ink-3: #6f8781; --rule: #26342f;
  --mark: #8c9bff; --held: #4fc386; --failed: #ee7a52; --base: #4d635d; --wash: rgba(140, 155, 255, 0.11); } }
:root[data-theme="dark"] { color-scheme: dark; --paper: #0f1716; --sheet: #15201e; --ink: #e3ece9; --ink-2: #a8bcb6; --ink-3: #6f8781;
  --rule: #26342f; --mark: #8c9bff; --held: #4fc386; --failed: #ee7a52; --base: #4d635d; --wash: rgba(140, 155, 255, 0.11); }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.6 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 28px 80px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); display: flex; flex-direction: column; gap: 32px; }
h1, h2, h3 { font-family: var(--display); margin: 0; letter-spacing: -0.015em; text-wrap: balance; }
h1 { font-size: clamp(2.3rem, 6.5vw, 4rem); font-weight: 800; line-height: 0.98; }
h2 { font-size: clamp(1.25rem, 2.8vw, 1.7rem); font-weight: 700; line-height: 1.2; }
h3 { font-size: 1.05rem; font-weight: 700; line-height: 1.25; }
h3 small { display: block; font: 400 0.82rem/1.4 var(--body); color: var(--ink-3); letter-spacing: 0; margin-top: 2px; }
p, ul, ol { margin: 0; }
a { color: var(--mark); }
code, .mono { font: 0.9em var(--mono); }
.back { font: 600 0.8rem var(--mono); text-decoration: none; letter-spacing: 0.02em; }
.back:hover { text-decoration: underline; }
.eyebrow { font: 600 0.74rem/1.2 var(--mono); letter-spacing: 0.09em; text-transform: uppercase; color: var(--ink-3); }
.lede { color: var(--ink-2); max-width: 68ch; font-size: 1.08rem; }
header { display: flex; flex-direction: column; gap: 16px; }
.stats { list-style: none; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 6px; }
.stats li { background: var(--sheet); border: 1px solid var(--rule); border-top: 3px solid var(--mark); border-radius: 8px; padding: 14px 16px;
  display: flex; flex-direction: column; gap: 4px; }
.stats b { font: 800 2.1rem/1 var(--display); letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.stats span { color: var(--ink-2); font-size: 0.9rem; line-height: 1.45; }
section { background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; padding: 24px; display: flex; flex-direction: column;
  gap: 18px; min-width: 0; scroll-margin-top: 16px; }
.sec-head { display: flex; gap: 12px; align-items: baseline; }
.num { font: 600 0.9rem var(--mono); color: var(--mark); }
.story { display: grid; gap: 10px; max-width: 74ch; color: var(--ink-2); }
.story strong, .lesson strong { color: var(--ink); }
.figure { margin: 0; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 8px; background: var(--paper); padding: 6px 0; }
figcaption, .sub { font-size: 0.82rem; color: var(--ink-3); }
svg.diagram { display: block; width: 100%; min-width: calc(var(--w) * 0.8); max-width: var(--w); height: auto; margin: 0 auto; font-family: var(--body); }
svg .box rect { fill: var(--sheet); stroke: var(--rule); stroke-width: 1.2; }
svg .box.gm rect { fill: var(--wash); stroke: var(--mark); stroke-width: 2; }
svg .k { font: 600 9.5px var(--mono); letter-spacing: 0.1em; fill: var(--ink-3); }
svg .t { font: 700 17px var(--display); fill: var(--ink); }
svg .d { font: 400 12px var(--body); fill: var(--ink-2); }
svg .lab { font: 600 12.5px var(--body); fill: var(--ink); }
svg .lab2 { font: 400 11.5px var(--body); fill: var(--ink-2); }
svg .edge { fill: none; stroke: var(--ink-3); stroke-width: 1.6; }
svg .edge.m { stroke: var(--mark); stroke-width: 2.2; }
svg .head { fill: var(--ink-3); }
svg .head.m { fill: var(--mark); }
.depth { display: flex; gap: 2px; }
.depth div { min-width: 0; display: flex; flex-direction: column; gap: 5px; }
.depth i { display: block; height: 16px; border-radius: 3px; background: var(--base); }
.depth .hot i { background: var(--mark); }
.depth span { font: 600 0.68rem/1.2 var(--mono); color: var(--ink-2); white-space: nowrap; }
.depth small { font-size: 0.76rem; line-height: 1.3; color: var(--ink-3); overflow-wrap: anywhere; }
.boards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr)); gap: 28px 36px; }
.board { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.rows, .prows { list-style: none; padding: 0; display: flex; flex-direction: column; }
.rows { gap: 9px; margin-top: 8px; }
.rows li { display: grid; grid-template-columns: minmax(0, 11.5rem) minmax(60px, 1fr) 5.6rem; gap: 12px; align-items: center; }
.rname { display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }
.rname a { font: 600 0.82rem var(--mono); text-decoration: none; }
.rname a:hover { text-decoration: underline; }
.rname small { color: var(--ink-3); font-size: 0.76rem; }
.split { display: flex; gap: 2px; height: 14px; }
.split i { display: block; min-width: 3px; }
.split i:first-child { border-radius: 4px 0 0 4px; }
.split i:last-child { border-radius: 0 4px 4px 0; }
.split i:only-child { border-radius: 4px; }
.split .won { background: var(--mark); }
.split .tie { background: var(--rule); }
.split .lost { background: var(--base); }
.score { font: 400 0.86rem var(--mono); font-variant-numeric: tabular-nums; color: var(--ink-2); white-space: nowrap; }
.score b { color: var(--ink); font-weight: 600; }
.tied { color: var(--ink-3); font-size: 0.74rem; }
.key { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 0.82rem; color: var(--ink-2); }
.key span { display: inline-flex; align-items: center; gap: 7px; }
.key i { display: block; width: 14px; height: 10px; border-radius: 3px; }
.lessons { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 14px; }
.lesson { border: 1px solid var(--rule); border-radius: 8px; background: var(--paper); padding: 16px; display: flex; flex-direction: column; gap: 10px;
  min-width: 0; }
.lesson p { color: var(--ink-2); font-size: 0.92rem; }
.lesson p.num { font-size: 0.8rem; color: var(--mark); }
.prows { gap: 6px; }
.prows li { display: grid; grid-template-columns: minmax(0, 9.6rem) minmax(40px, 1fr) 5.4rem; gap: 10px; align-items: center; }
.lesson .prows li { grid-template-columns: minmax(0, 6.2rem) minmax(40px, 1fr) 4.6rem; }
.mname { font-size: 0.82rem; color: var(--ink-2); line-height: 1.25; }
.gm .mname { color: var(--ink); font-weight: 600; }
.track { display: block; position: relative; height: 12px; }
.track i { display: block; height: 100%; min-width: 3px; border-radius: 0 4px 4px 0; background: var(--base); }
.gm .track i { background: var(--mark); }
.val { font: 600 0.82rem var(--mono); font-variant-numeric: tabular-nums; color: var(--ink); white-space: nowrap; }
.val small { font-weight: 400; color: var(--ink-3); }
.profiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 22px 36px; }
.pround { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.tscroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 8px 12px 8px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font: 600 0.7rem var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); white-space: nowrap; }
td { color: var(--ink-2); }
td.n, th.n { font-variant-numeric: tabular-nums; white-space: nowrap; }
td:first-child { color: var(--ink); white-space: nowrap; }
td.mono a { text-decoration: none; font-weight: 600; }
.tag { font: 600 0.68rem var(--mono); letter-spacing: 0.06em; text-transform: uppercase; white-space: nowrap; padding: 2px 7px; border-radius: 999px;
  border: 1px solid currentColor; }
.tag.held { color: var(--held); }
.tag.failed { color: var(--failed); }
.now { border-left: 3px solid var(--mark); background: var(--wash); border-radius: 6px; padding: 12px 16px; color: var(--ink-2); max-width: 78ch; }
.now b { color: var(--ink); }
.notes { display: grid; gap: 10px; max-width: 78ch; color: var(--ink-2); font-size: 0.95rem; }
.notes strong { color: var(--ink); }
footer { color: var(--ink-3); font-size: 0.84rem; display: grid; gap: 4px; }
@media (max-width: 560px) { section { padding: 16px; }
  .rows li { grid-template-columns: minmax(0, 1fr) 5.6rem; }
  .rows .rname { grid-column: 1 / -1; }
  .prows li, .lesson .prows li { grid-template-columns: minmax(0, 1fr) 5.4rem; }
  .prows .mname { grid-column: 1 / -1; } }
</style>
</head>
<body>
<div class="wrap">
<header>
  <a class="back" href="./">← Bonsai Neuron Survey map</a>
  <p class="eyebrow">Bonsai-8B · labelled neurons as controls · Nov 7–9 2026</p>
  <h1>Steering with globalMOO</h1>
  <p class="lede">The survey says what each neuron does. This page uses those labels. Bundles of labelled neurons become
  24 knobs on the model's writing, and <strong>globalMOO</strong>, a multiobjective inverse solver, is asked for a whole
  behaviour in one request: raise these topics, hold those, keep the text fluent, leave everything else alone. Nine rounds,
  each with its claims committed before anything was scored.</p>
  <ul class="stats">
    <li><b>8 of 8</b><span>rounds where globalMOO's designs beat the additive model's on prompts neither had seen</span></li>
    <li><b>7 goals</b><span>in a single query: four topic targets, two text-quality floors and one drift limit</span></li>
    <li><b>15 of 17</b><span>registered claims held; the two that did not are in the ledger below</span></li>
  </ul>
</header>

<section id="setup">
  <div class="sec-head"><span class="num">01</span><h2>The problem: 24 knobs in, 27 readings out</h2></div>
  <div class="story">
    <p>Each topic (medicine, weather, money, travel, then school, software, war, family) has a bundle of neurons whose
    labels held their tests. Each bundle is split into six depth bands, and every band gets its own gain, so four topics
    make <strong>24 knobs</strong>. For any setting of the knobs, Bonsai-8B continues a set of prompts and 27 things are
    read off the text: how often each of 24 topics comes up, how varied and how coherent the text is, and how far the
    topics nobody asked about have drifted.</p>
    <p>globalMOO learns the map from one sweep of 687 knob settings. After that, each request names values for several
    readings at once, and globalMOO proposes the knob settings to try.</p>
  </div>
  <figure class="figure"><div class="scroll">__SETUP__</div>
  <figcaption>One model per round, trained once. Every later question is a new set of goals put to the same model.</figcaption></figure>
  <figure class="figure">
    <div class="depth" role="img" aria-label="The six depth bands of Bonsai-8B's 36 layers. Layers 12 to 23 give the cheapest steering.">
      <div style="flex:3 1 0"><i></i><span>L0</span></div>
      <div style="flex:3 1 0"><i></i><span>3</span></div>
      <div style="flex:3 1 0"><i></i><span>6</span></div>
      <div style="flex:3 1 0"><i></i><span>9</span></div>
      <div style="flex:12 1 0" class="hot"><i></i><span>L12–23</span><small>the cheap band: medicine ×23 with the text intact</small></div>
      <div style="flex:12 1 0"><i></i><span>L24–35</span></div>
    </div>
    <figcaption>The six bands across the model's 36 layers, each marked with its first layer. Depth changes the price of steering by an order of magnitude:
    gain in L0–2 breaks the text early, L9–11 barely moves anything, and L12–23 does most of the work (steer@3's pilots,
    training prompts).</figcaption>
  </figure>
</section>

<section id="scoreboard">
  <div class="sec-head"><span class="num">02</span><h2>On unseen prompts, globalMOO's designs win every round</h2></div>
  <div class="story">
    <p>Every method picks its best knob settings on the training prompts. Those settings are then scored on prompts no
    method saw. The hardest baseline is the <strong>additive model</strong>: sweep each band alone, add the curves, solve,
    correct and solve again. It often fits the training prompts as well as globalMOO or better. Its designs transfer worse,
    in every round.</p>
  </div>
  <div class="key" aria-hidden="true"><span><i style="background:var(--mark)"></i>globalMOO's design is better</span>
    <span><i style="background:var(--rule)"></i>tied</span><span><i style="background:var(--base)"></i>the baseline's is better</span></div>
  <div class="boards">__BOARDS__</div>
  <p class="sub">Cells are task × text-quality tier (24 per round) through steer@6, and whole profiles (16 per round) from steer@7.
  steer@3b re-asks steer@3's cells with the same model; steer@9 re-reads steer@8's designs on new prompts. Three independently
  trained models before the profile rounds, eight topics in all.</p>
</section>

<section id="lessons">
  <div class="sec-head"><span class="num">03</span><h2>Three things about the setup that made the difference</h2></div>
  <div class="story"><p>Each early shortfall traced to how the problem was posed, and each fix was tested in the next
  pre-registered round.</p></div>
  <div class="lessons">__LESSONS__</div>
</section>

<section id="profiles">
  <div class="sec-head"><span class="num">04</span><h2>Whole profiles: every topic at its own target, in one query</h2></div>
  <div class="story">
    <p>From steer@6 a request fixes all four panel topics at once, some up, some down, some held where they are, under
    the two text-quality floors and the drift limit. The score is the <strong>worst miss</strong>: the largest error over
    the four topics, in log10, so 0.30 is a factor of 2. Lower is better.</p>
    <p>Once the targets were drawn from what the panel can reach (steer@7), globalMOO put every topic within 1.5× of its
    target on the training prompts in 13 of 16 profiles. On unseen prompts its designs had the lowest mean miss of any
    method in all four profile rounds.</p>
  </div>
  <div class="profiles">__PROFILES__</div>
  <p class="sub">Mean worst miss on unseen prompts over the round's profiles (log10, and as a factor). “Best design point” is the
  best of the 687 settings globalMOO was trained on, with no query at all.</p>
</section>

<section id="measure">
  <div class="sec-head"><span class="num">05</span><h2>The ruler was the limit, not the optimizer</h2></div>
  <div class="story">
    <p>No method held a profile within 1.5× on unseen prompts, and steer@9 and steer@10 found why. Every round had scored
    a topic by its average rate over the prompts, which a few prompts dominate. Scored that way, two halves of the same 64
    fresh prompts disagree about the very same designs by a factor of 3. The misses above sit at that floor.</p>
    <p>Scoring each prompt on its own and then averaging gives one answer on any prompt set. The same readings, both ways:</p>
  </div>
  <div class="tscroll"><table>
    <thead><tr><th>the same 52 designs, the same prompts</th><th class="n">average rate</th><th class="n">per prompt</th></tr></thead>
    <tbody>
      <tr><td>Agreement between two halves of the fresh prompts (correlation, by topic)</td><td class="n">0.27 to 0.55</td><td class="n"><b>0.88 to 0.96</b></td></tr>
      <tr><td>Agreement between training and fresh prompts (correlation)</td><td class="n">0.54 to 0.81</td><td class="n"><b>0.94 to 0.97</b></td></tr>
      <tr><td>Slope from training to fresh prompts (1 means the effect carries over whole)</td><td class="n">0.48 to 0.66</td><td class="n"><b>1.04 to 1.08</b></td></tr>
      <tr><td>Typical gap between the two halves (worst topic, log10)</td><td class="n">0.47</td><td class="n"><b>0.12</b></td></tr>
    </tbody>
  </table></div>
  <p class="now"><b>Running now: steer@11.</b> The profile round again, scored per prompt, with a new globalMOO model.
  Registered before scoring: globalMOO lands at least 8 of 16 profiles within 1.5× on 64 unseen prompts, and beats the
  additive model there. The result goes on this page whichever way it falls.</p>
</section>

<section id="ledger">
  <div class="sec-head"><span class="num">06</span><h2>Every registered claim, as it fell</h2></div>
  <div class="story"><p>Each round's protocol was committed on its own before anything was scored, and its results in a
  separate commit after. Nothing here was chosen after the fact.</p></div>
  <div class="tscroll"><table>
    <thead><tr><th>round</th><th>claim, registered first</th><th>result</th><th>figure</th></tr></thead>
    <tbody>__LEDGER__</tbody>
  </table></div>
</section>

<section>
  <h2>What this does not show yet</h2>
  <div class="notes">
    <p><strong>Precision on new text.</strong> Under the old measure no method kept a four-topic profile within 1.5× on
    unseen prompts. steer@11 is the first fair test of that.</p>
    <p><strong>Training fit.</strong> The additive model usually matches the training prompts as closely as globalMOO, or
    more closely. globalMOO's advantage is in what carries over to new prompts, and in asking for many readings at once.</p>
    <p><strong>Scope.</strong> One model (Bonsai-8B), eight topics, 24 generated tokens per prompt. A topic's rate is the
    probability the model gives its tokens while writing, not a human judgment of the text.</p>
    <p><strong>Margins.</strong> On steer@9's fresh prompts globalMOO's lead over the additive model was narrow (0.423
    against 0.431). Money could not be raised much within the drift limit by any method in steer@3.</p>
  </div>
</section>

<footer>
  <p>From the <a href="./">Bonsai Neuron Survey</a> · figures from <a href="__REPO__/blob/main/FINDINGS.md">FINDINGS.md</a> and the
  reports in <a href="__REPO__/tree/main/bonsai_interp/moo">bonsai_interp/moo</a> · see also <a href="circuits.html">circuits from the stream</a>.</p>
  <p>A project by Matt Freeman · <a href="https://x.com/moridinamael" target="_blank" rel="noopener">@moridinamael on X</a> ·
  <a href="https://www.linkedin.com/in/matt-freeman-phd" target="_blank" rel="noopener">LinkedIn</a></p>
</footer>
</div>
</body>
</html>
"""


def page():
    boards = (scoreboard(0, "Against the additive model", "the strongest conventional baseline: one-band sweeps, added up")
              + scoreboard(1, "Against differential evolution", "a direct search on the live model, with 96 to 768 evaluations of its own"))
    lessons = (
        lesson("steer@3 → steer@3b", "Ask for a value, not a maximum",
               ("maximize", 1 / 24, "1 of 24", ), ("value target", 17 / 24, "17 of 24"),
               "Cells where a query improved on the best design point, same model, same cells. Told to maximize, the queries "
               "stayed where they started. Told to reach a value just past the best known point, they climbed, and found "
               "feasible settings for four cells the 687-point sweep never met.")
        + lesson("steer@4 → steer@5", "Give it enough prompts to average over",
                 ("16 prompts", 4 / 24, "4 of 24"), ("32 prompts", 18 / 24, "18 of 24"),
                 "Cells where the query's gain over the best design point survived on unseen prompts. With 16 training "
                 "prompts most gains were noise (4 kept, 8 reversed). With 32 they held (18 kept, 3 reversed).")
        + lesson("steer@9 → steer@10", "Score every prompt alike",
                 ("average rate", 0.40, "r = 0.40"), ("per prompt", 0.94, "r = 0.94"),
                 "Agreement between two halves of the fresh prompts about the same designs (mean correlation over the four "
                 "topics). The per-prompt measure is the one steer@11 optimizes."))
    return (PAGE.replace("__SETUP__", SETUP_SVG).replace("__BOARDS__", boards).replace("__LESSONS__", lessons)
            .replace("__PROFILES__", profile_chart()).replace("__LEDGER__", ledger()).replace("__REPO__", REPO))


def main(out: Path):
    assert sum(c[3] for c in CLAIMS) == 15 and len(CLAIMS) == 17
    assert all(r[3][0] > r[3][1] and r[4][0] > r[4][1] for r in ROUNDS) and len(ROUNDS) == 8
    assert all(v[0] == min(v) for _, _, v in PROFILES)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site" / "steering.html")
