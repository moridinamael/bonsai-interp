"""One spec for every inverse problem, so every solver can run every problem.

An InverseProblem maps a bounded input vector x (the antecedents: gains on
upstream neurons, coefficients on residual directions, slots of a text
generator, ...) to named outputs (neuron activations, KL, naturalness, ...).
Goals say what we want of the outputs. Solvers in bonsai_interp.inverse.solvers read
the same Problem + Goals; each records every evaluation in problem.history so
solvers can be compared on evaluations-to-goal.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Var:
    name: str
    lo: float
    hi: float
    kind: str = "float"  # "float" | "int" | "bool"


@dataclass
class Goal:
    """op: '>=', '<=', '~=' (within tol, relative), '=' (within tol, absolute), 'max', 'min'."""
    output: str
    op: str
    value: float = 0.0
    tol: float = 0.05
    weight: float = 1.0


@dataclass
class InverseProblem:
    name: str
    vars: list
    outputs: list
    evaluate_batch: Callable  # (n, d) ndarray -> (n, m) ndarray
    torch_fn: Optional[Callable] = None  # (d,) tensor -> (m,) tensor, differentiable
    scales: Optional[np.ndarray] = None  # per-output scale for scalarization
    history: list = field(default_factory=list)  # (x, y, solver) per evaluation
    tag: str = ""

    @property
    def lo(self):
        return np.array([v.lo for v in self.vars], float)

    @property
    def hi(self):
        return np.array([v.hi for v in self.vars], float)

    @property
    def integrality(self):
        return np.array([v.kind != "float" for v in self.vars])

    def out_index(self, name):
        return self.outputs.index(name)

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Y = np.atleast_2d(np.asarray(self.evaluate_batch(X), float))
        for x, y in zip(X, Y):
            self.history.append((x.copy(), y.copy(), self.tag))
        return Y

    def reset(self, tag=""):
        self.history, self.tag = [], tag

    def calibrate_scales(self, n=16, seed=0):
        """Output scales from a small random design (used to scalarize goals)."""
        rng = np.random.default_rng(seed)
        X = self.lo + rng.random((n, len(self.vars))) * (self.hi - self.lo)
        Y = self.evaluate(X)
        self.scales = np.maximum(Y.std(0), 1e-3 * np.abs(Y).mean(0) + 1e-6)
        return Y


def goal_terms(goals, outputs, y, scales, xp=np):
    """Per-goal penalty, 0 when satisfied (max/min are linear rewards)."""
    terms = []
    for g in goals:
        i = outputs.index(g.output)
        s = scales[i]
        v = y[..., i]
        if g.op == ">=":
            t = xp.maximum(g.value - v, 0 * v) / s
        elif g.op == "<=":
            t = xp.maximum(v - g.value, 0 * v) / s
        elif g.op == "~=":
            band = abs(g.value) * g.tol
            t = xp.maximum(abs(v - g.value) - band, 0 * v) / s
        elif g.op == "=":
            t = xp.maximum(abs(v - g.value) - g.tol, 0 * v) / s
        elif g.op == "max":
            t = -v / s
        elif g.op == "min":
            t = v / s
        else:
            raise ValueError(g.op)
        terms.append((g, t))
    return terms


def scalarize(goals, outputs, y, scales, xp=np):
    total = 0
    for g, t in goal_terms(goals, outputs, y, scales, xp):
        total = total + g.weight * (t if g.op in ("max", "min") else t * t)
    return total


def satisfied(goals, outputs, y):
    ok = True
    for g in goals:
        v = y[outputs.index(g.output)]
        if g.op == ">=":
            ok &= v >= g.value
        elif g.op == "<=":
            ok &= v <= g.value
        elif g.op == "~=":
            ok &= abs(v - g.value) <= abs(g.value) * g.tol
        elif g.op == "=":
            ok &= abs(v - g.value) <= g.tol
    return bool(ok)


def pareto_mask(F):
    """Non-dominated rows of F (all columns minimized)."""
    F = np.asarray(F)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dominated = np.all(F[i] <= F, 1) & np.any(F[i] < F, 1)
            keep &= ~dominated
    return keep
