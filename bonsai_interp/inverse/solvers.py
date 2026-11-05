"""Solver backends over the shared InverseProblem spec.

Which to use (see RESEARCH_PLAN.md, "Forwards and backwards"):
  lbfgs / least_squares  smooth problem with a torch_fn; one target / scalarized
  differential_evolution black-box global baseline; handles integer vars
  dual_annealing         black-box global baseline, continuous
  GmooInverse            black-box, multi-output, inverse targets; trained once
                         per problem, queried for many goal sets
Linear (L0/L1) inverses live in bonsai_interp.inverse.linear.
"""
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from dotenv import load_dotenv
from scipy import optimize

from bonsai_interp.env import DEVICE
from bonsai_interp.inverse.problem import InverseProblem, goal_terms, satisfied, scalarize


@dataclass
class Result:
    solver: str
    x: np.ndarray
    y: np.ndarray
    satisfied: bool
    n_evals: int
    seconds: float
    extra: dict = field(default_factory=dict)


def _best_from_history(p: InverseProblem, goals, solver, t0, start, extra=None):
    hist = p.history[start:] or p.history
    losses = [float(scalarize(goals, p.outputs, y, p.scales)) for _, y, _ in hist]
    sats = [satisfied(goals, p.outputs, y) for _, y, _ in hist]
    # prefer satisfied points, then lowest scalarized loss
    i = min(range(len(hist)), key=lambda k: (not sats[k], losses[k]))
    x, y, _ = hist[i]
    return Result(solver, x, y, sats[i], len(hist), time.time() - t0, extra or {})


def _torch_loss(p, goals):
    scales = torch.tensor(p.scales, dtype=torch.float32, device=DEVICE)

    def f(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
        y = p.torch_fn(x)
        loss = scalarize(goals, p.outputs, y, scales, xp=torch)
        loss.backward()
        p.history.append((x_np.copy(), y.detach().float().cpu().numpy(), p.tag))
        return loss.item(), x.grad.cpu().double().numpy()

    return f


def lbfgs(p: InverseProblem, goals, x0, maxiter=60):
    """Gradient-based local solve of the scalarized goals (needs p.torch_fn)."""
    p.tag, start, t0 = "lbfgs", len(p.history), time.time()
    optimize.minimize(_torch_loss(p, goals), np.asarray(x0, float), jac=True, method="L-BFGS-B",
                      bounds=list(zip(p.lo, p.hi)), options={"maxiter": maxiter})
    return _best_from_history(p, goals, "lbfgs", t0, start)


def least_squares(p: InverseProblem, goals, x0, max_nfev=60):
    """Trust-region solve of goal residuals with an autograd Jacobian. Best for
    hitting exact target values on several outputs at once."""
    p.tag, start, t0 = "least_squares", len(p.history), time.time()
    scales = torch.tensor(p.scales, dtype=torch.float32, device=DEVICE)

    def resid_t(x):
        return torch.stack([t * (g.weight ** 0.5) for g, t in
                            goal_terms(goals, p.outputs, p.torch_fn(x), scales, xp=torch)])

    def fun(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            y = p.torch_fn(x)
        p.history.append((x_np.copy(), y.float().cpu().numpy(), p.tag))
        return torch.stack([t for _, t in goal_terms(goals, p.outputs, y, scales, xp=torch)]).cpu().double().numpy()

    def jac(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
        return torch.autograd.functional.jacobian(resid_t, x).cpu().double().numpy()

    optimize.least_squares(fun, np.clip(np.asarray(x0, float), p.lo, p.hi), jac=jac,
                           bounds=(p.lo, p.hi), max_nfev=max_nfev, method="trf")
    return _best_from_history(p, goals, "least_squares", t0, start)


def differential_evolution(p: InverseProblem, goals, maxiter=20, popsize=8, seed=0):
    p.tag, start, t0 = "differential_evolution", len(p.history), time.time()

    def f(X):  # vectorized: X is (d, S)
        Y = p.evaluate(X.T)
        return np.array([scalarize(goals, p.outputs, y, p.scales) for y in Y])

    optimize.differential_evolution(f, list(zip(p.lo, p.hi)), maxiter=maxiter, popsize=popsize,
                                    integrality=p.integrality, vectorized=True, seed=seed,
                                    polish=False, updating="deferred", tol=0)
    return _best_from_history(p, goals, "differential_evolution", t0, start)


def dual_annealing(p: InverseProblem, goals, maxfun=300, seed=0):
    p.tag, start, t0 = "dual_annealing", len(p.history), time.time()
    f = lambda x: float(scalarize(goals, p.outputs, p.evaluate(x)[0], p.scales))
    optimize.dual_annealing(f, list(zip(p.lo, p.hi)), maxfun=maxfun, seed=seed, no_local_search=True)
    return _best_from_history(p, goals, "dual_annealing", t0, start)


# --------------------------------------------------------------------------- globalMOO

def _gmoo():
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    from globalmoo.client import Client
    return Client(timeout=600.0)  # training on 100 outputs outlasts the default 30 s


def _benign(exc):
    m = str(exc).lower()
    return "already been loaded" in m or "satisfied" in m or ("iteration" in m and ("limit" in m or "maximum" in m))


class GmooInverse:
    """globalMOO learned inverse: train once on the problem's design of experiments,
    then query any number of goal sets. Each query is its own objective on the
    same trial, so the DOE is paid once per problem."""

    FREE_SPANS = 10.0  # untargeted outputs: absolute band of this many DOE spans
    ASPIRATION_SPANS = 1.0   # min/max aim this many DOE spans past the observed range
    INPUT = {"float": "float", "int": "integer", "bool": "boolean"}

    def __init__(self, p: InverseProblem, name: str):
        from globalmoo.request.create_model import CreateModel
        from globalmoo.request.create_project import CreateProject
        from globalmoo.request.load_output_cases import LoadOutputCases
        self.p, self.client = p, _gmoo()
        t0, start = time.time(), len(p.history)
        p.tag = "gmoo_doe"
        model = self.client.execute_request(CreateModel(name=name, description=f"mono {p.name}"))
        project = self.client.execute_request(CreateProject(
            model_id=model.id, name=f"{name}-project", input_count=len(p.vars),
            minimums=[float(v.lo) if v.kind == "float" else int(v.lo) for v in p.vars],
            maximums=[float(v.hi) if v.kind == "float" else int(v.hi) for v in p.vars],
            input_types=[self.INPUT[v.kind] for v in p.vars], categories=[]))
        X = np.array(project.input_cases, float)
        Y = p.evaluate(X)
        if p.scales is None:
            p.scales = np.maximum(Y.std(0), 1e-3 * np.abs(Y).mean(0) + 1e-6)
        self.doe_x, self.doe_y = X, Y
        self.trial = self.client.execute_request(LoadOutputCases(
            project_id=project.id, output_count=len(p.outputs), output_cases=Y.tolist()))
        self.doe_evals = len(p.history) - start
        self.doe_seconds = time.time() - t0
        self.model_id, self.project_id = model.id, project.id

    @classmethod
    def attach(cls, p: InverseProblem, model_id: int, trial_id: int):
        """Re-use an already trained globalMOO trial (no new model): read its design back
        from the platform and re-evaluate it locally (deterministic, so outputs match)."""
        from globalmoo.request.read_model import ReadModel
        self = cls.__new__(cls)
        self.p, self.client = p, _gmoo()
        model = self.client.execute_request(ReadModel(model_id=model_id))
        project = model.projects[0]
        t0, start = time.time(), len(p.history)
        p.tag = "gmoo_doe"
        self.doe_x = np.array(project.input_cases, float)
        self.doe_y = p.evaluate(self.doe_x)
        self.trial = type("Trial", (), {"id": trial_id})()
        self.doe_evals, self.doe_seconds = len(p.history) - start, time.time() - t0
        self.model_id, self.project_id = model.id, project.id
        return self

    def _objectives(self, goals):
        from globalmoo.enums.objective_type import ObjectiveType as OT
        lo, hi = self.doe_y.min(0), self.doe_y.max(0)
        vals, types, mins, maxs = [], [], [], []
        by_out = {g.output: g for g in goals}
        for i, name in enumerate(self.p.outputs):
            g = by_out.get(name)
            span = max(hi[i] - lo[i], 1e-6)
            if g is None:
                # a PERCENT band around a value near 0 collapses to 0 and pins the
                # output; an absolute band around the DOE midpoint leaves it free
                vals.append(float((lo[i] + hi[i]) / 2)); types.append(OT.VALUE)
                mins.append(-self.FREE_SPANS * span); maxs.append(self.FREE_SPANS * span)
                continue
            if g.op == "max":
                vals.append(float(hi[i] + self.ASPIRATION_SPANS * span)); types.append(OT.MAXIMIZE)
            elif g.op == "min":
                vals.append(float(lo[i] - self.ASPIRATION_SPANS * span)); types.append(OT.MINIMIZE)
            elif g.op == ">=":
                vals.append(float(g.value)); types.append(OT.GREATER_THAN)
            elif g.op == "<=":
                vals.append(float(g.value)); types.append(OT.LESS_THAN)
            elif g.op == "~=":
                vals.append(float(g.value)); types.append(OT.PERCENT)
                mins.append(-100 * g.tol); maxs.append(100 * g.tol)
                continue
            mins.append(0.0); maxs.append(0.0)
        return vals, types, mins, maxs

    def query(self, goals, max_iter=60, stall_limit=3, start="best"):
        from globalmoo.request.load_inversed_output import LoadInversedOutput
        from globalmoo.request.load_objectives import LoadObjectives
        from globalmoo.request.suggest_inverse import SuggestInverse
        p = self.p
        p.tag, start, t0 = "gmoo", len(p.history), time.time()
        vals, types, mins, maxs = self._objectives(goals)
        if start == "best":  # the DOE point that best meets these goals
            k = int(np.argmin([scalarize(goals, p.outputs, y, p.scales) for y in self.doe_y]))
            x0, y0 = self.doe_x[k], self.doe_y[k]
        else:  # neutral centre of the input box: the learned inverse must do the work
            x0 = (p.lo + p.hi) / 2
            y0 = p.evaluate(x0[None])[0]
        self.init_x = [float(v) if var.kind == "float" else int(round(v)) for v, var in zip(x0, p.vars)]
        self.init_y = [float(v) for v in y0]
        obj = self.client.execute_request(LoadObjectives(
            trial_id=self.trial.id, objectives=vals, objective_types=types,
            initial_input=self.init_x, initial_output=self.init_y,
            minimum_bounds=mins, maximum_bounds=maxs, desired_l1_norm=0.0))
        last, repeats, stopped = None, 0, "max_iter"
        for _ in range(max_iter):
            try:
                inv = self.client.execute_request(SuggestInverse(objective_id=obj.id))
            except Exception as e:
                if _benign(e):
                    stopped = "platform_done"; break
                raise
            x = list(inv.input)
            repeats = repeats + 1 if x == last else 0
            if repeats >= stall_limit:
                stopped = "stalled"; break
            last = x
            y = p.evaluate(np.array([x]))[0]
            try:
                r = self.client.execute_request(LoadInversedOutput(inverse_id=inv.id, output=y.tolist()))
            except Exception as e:
                if _benign(e):
                    stopped = "platform_done"; break
                raise
            if r.satisfied_at is not None:
                stopped = "satisfied"; break
            if satisfied(goals, p.outputs, y) and not any(g.op in ("max", "min") for g in goals):
                stopped = "satisfied_local"; break
        res = _best_from_history(p, goals, "gmoo", t0, start, {"stopped": stopped})
        res.extra["doe_evals"] = self.doe_evals
        return res
