"""
pdcp_utils.py  —  shared code for the PDCP project (Proxy-Decoupled Counterfactual Probing)
Version: 0.1.0  (written by N01, 2026-11)

Contents
  1. Graph families            make_graph(family, n, seed)
  2. Exact metrics             ksm, node_uniformity, edge_length_deviation, crossing_ratio, all_metrics
  3. Differentiable surrogates (torch)  used only inside the optimiser
  4. PDSG Stage 1              pdsg_optimise(G, targets, ...)
  5. Rendering                 render_png(G, X, path, ...)
  6. Parsing model output      parse_answer(text)
  7. Statistics helpers        bootstrap_ci(...)

All coordinates X are numpy arrays of shape (n, 2). Node order = sorted(G.nodes()) after relabel to 0..n-1.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

__version__ = "0.1.2"

# ----------------------------------------------------------------------------
# 1. Graph families
# ----------------------------------------------------------------------------
FAMILY_SIZES = {
    "ER": [10, 25, 50, 100],
    "BA": [25, 50, 100],
    "SBM": [25, 50, 100],
    "WS": [25, 50],
    "GRID": [16, 25, 49, 100],
    "TREE": [25, 50, 100],
}


def _connected_or_retry(fn, seed, max_tries=200):
    for t in range(max_tries):
        G = fn(seed + 1000 * t)
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not sample a connected graph")


def make_graph(family: str, n: int, seed: int) -> nx.Graph:
    """Return a connected simple graph with nodes relabelled 0..n-1."""
    family = str(family).upper(); n = int(n); seed = int(seed)
    if family == "ER":
        p = 3.5 / (n - 1)  # expected average degree 3.5
        G = _connected_or_retry(lambda s: nx.gnp_random_graph(n, p, seed=s), seed)
    elif family == "BA":
        G = nx.barabasi_albert_graph(n, 2, seed=seed)
    elif family == "SBM":
        k = {25: 2, 50: 3, 100: 4}.get(n, max(2, n // 25))
        p_in = {25: 0.40, 50: 0.25, 100: 0.15}.get(n, 0.2)
        sizes = [n // k] * k
        sizes[-1] += n - sum(sizes)
        P = [[p_in if i == j else 0.02 for j in range(k)] for i in range(k)]
        G = _connected_or_retry(lambda s: nx.stochastic_block_model(sizes, P, seed=s), seed)
        G = nx.Graph(G)
    elif family == "WS":
        G = nx.connected_watts_strogatz_graph(n, 4, 0.1, seed=seed)
    elif family == "GRID":
        a = int(round(math.sqrt(n)))
        G = nx.grid_2d_graph(a, a)
    elif family == "TREE":
        try:
            G = nx.random_labeled_tree(n, seed=seed)
        except AttributeError:  # older networkx
            G = nx.random_tree(n, seed=seed)
    else:
        raise ValueError(f"unknown family {family}")
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def graph_arrays(G: nx.Graph):
    """Edge index array (m,2), shortest-path matrix D (n,n), and index arrays for i<j pairs."""
    n = G.number_of_nodes()
    E = np.array(sorted((min(u, v), max(u, v)) for u, v in G.edges()), dtype=np.int64)
    D = np.zeros((n, n), dtype=np.float64)
    for s, lengths in nx.all_pairs_shortest_path_length(G):
        for t, d in lengths.items():
            D[s, t] = d
    iu, ju = np.triu_indices(n, k=1)
    return E, D, iu, ju


def nonadjacent_edge_pairs(E: np.ndarray):
    """All pairs of edges that share no endpoint (the pairs that can cross)."""
    m = len(E)
    a, b = np.triu_indices(m, k=1)
    e1, e2 = E[a], E[b]
    share = (e1[:, 0] == e2[:, 0]) | (e1[:, 0] == e2[:, 1]) | (e1[:, 1] == e2[:, 0]) | (e1[:, 1] == e2[:, 1])
    return a[~share], b[~share]


# ----------------------------------------------------------------------------
# 2. Exact metrics
# ----------------------------------------------------------------------------
def normalise(X: np.ndarray) -> np.ndarray:
    """Uniform scale into the unit box, aspect ratio preserved, centred."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.min(0)
    s = X.max()
    if s <= 0:
        return np.full_like(X, 0.5)
    X = X / s
    X = X + (1.0 - X.max(0)) / 2.0
    return X


def _pav(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool-adjacent-violators: isotonic (non-decreasing) fit of y in given order."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt); wts.append(wt); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def ksm(X: np.ndarray, D: np.ndarray, iu=None, ju=None, ties: str = "secondary") -> float:
    """
    Kruskal (non-metric) stress in [0,1]; dhat = isotonic regression of e on d.
    ties='secondary' (default): all pairs with equal graph distance get the same dhat
        -> identical to sklearn.isotonic.IsotonicRegression, the convention used by Mooney et al.
    ties='primary': Kruskal's primary approach (ties may be broken); gives lower values. Kept for reference only.
    """
    n = len(X)
    if iu is None:
        iu, ju = np.triu_indices(n, k=1)
    d = D[iu, ju]
    e = np.linalg.norm(X[iu] - X[ju], axis=1)
    if ties == "primary":
        order = np.lexsort((e, d))
        dhat = np.empty_like(e)
        dhat[order] = _pav(e[order])
    else:
        levels, inv, cnt = np.unique(d, return_inverse=True, return_counts=True)
        means = np.bincount(inv, weights=e) / cnt
        fit = _pav(means, cnt)
        dhat = fit[inv]
    num = np.sum((e - dhat) ** 2)
    den = np.sum(e ** 2)
    return float(np.sqrt(num / den)) if den > 0 else 0.0


def node_uniformity(X: np.ndarray) -> float:
    """0 = nodes spread uniformly over a ceil(sqrt n)^2 grid on the drawing's bounding box, 1 = all in one cell."""
    n = len(X)
    g = int(math.ceil(math.sqrt(n)))
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    U = (X - lo) / span
    idx = np.minimum((U * g).astype(int), g - 1)
    counts = np.zeros((g, g))
    np.add.at(counts, (idx[:, 0], idx[:, 1]), 1)
    cbar = n / g ** 2
    num = np.sqrt(np.sum((counts - cbar) ** 2))
    den = np.sqrt((n - cbar) ** 2 + (g ** 2 - 1) * cbar ** 2)
    return float(num / den)


def edge_length_deviation(X: np.ndarray, E: np.ndarray) -> float:
    """Coefficient of variation of edge lengths (scale invariant)."""
    L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    mu = L.mean()
    return float(L.std() / mu) if mu > 0 else 0.0


def _orient(p, q, r):
    return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])


def crossing_count(X: np.ndarray, E: np.ndarray, pairs=None) -> int:
    if pairs is None:
        pairs = nonadjacent_edge_pairs(E)
    a, b = pairs
    if len(a) == 0:
        return 0
    P1, P2 = X[E[a, 0]], X[E[a, 1]]
    Q1, Q2 = X[E[b, 0]], X[E[b, 1]]
    o1, o2 = _orient(P1, P2, Q1), _orient(P1, P2, Q2)
    o3, o4 = _orient(Q1, Q2, P1), _orient(Q1, Q2, P2)
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)  # proper crossings only; collinear touches ignored
    return int(cross.sum())


def crossing_ratio(X: np.ndarray, E: np.ndarray, pairs=None) -> float:
    if pairs is None:
        pairs = nonadjacent_edge_pairs(E)
    cmax = len(pairs[0])
    return crossing_count(X, E, pairs) / cmax if cmax > 0 else 0.0


@dataclass
class GraphCache:
    """Pre-computed arrays for one graph, reused across many drawings."""
    G: nx.Graph
    E: np.ndarray = field(init=False)
    D: np.ndarray = field(init=False)
    iu: np.ndarray = field(init=False)
    ju: np.ndarray = field(init=False)
    pairs: tuple = field(init=False)

    def __post_init__(self):
        self.E, self.D, self.iu, self.ju = graph_arrays(self.G)
        self.pairs = nonadjacent_edge_pairs(self.E)

    @property
    def n(self):
        return self.G.number_of_nodes()


def realism_metrics(X: np.ndarray) -> dict:
    """Shape diagnostics used to reject degenerate drawings (not optimisation targets).
    aspect   = short side / long side of the bounding box (1 = square)
    nn_max   = largest nearest-neighbour distance * sqrt(n)   (large = isolated outlier node)
    nn_min   = smallest pairwise distance * sqrt(n)            (small = overlapping nodes)
    dense    = fraction of nodes inside the densest cell of a 4x4 grid (large = clump)"""
    X = normalise(X)
    n = len(X)
    span = X.max(0) - X.min(0)
    aspect = float(span.min() / span.max()) if span.max() > 0 else 0.0
    Dm = np.linalg.norm(X[:, None] - X[None], axis=-1) + np.eye(n) * 10
    nn = Dm.min(1)
    lo, hi = X.min(0), X.max(0)
    idx = np.minimum(((X - lo) / np.where(hi - lo > 0, hi - lo, 1) * 4).astype(int), 3)
    counts = np.zeros((4, 4)); np.add.at(counts, (idx[:, 0], idx[:, 1]), 1)
    return {"aspect": aspect, "nn_max": float(nn.max() * math.sqrt(n)),
            "nn_min": float(nn.min() * math.sqrt(n)), "dense": float(counts.max() / n)}


REALISM_LIMITS = {"aspect_min": 0.70, "nn_max_max": 2.6, "nn_min_min": 0.20, "dense_max": 0.35}


def realism_ok(r: dict, lim: dict = REALISM_LIMITS) -> bool:
    return (r["aspect"] >= lim["aspect_min"] and r["nn_max"] <= lim["nn_max_max"]
            and r["nn_min"] >= lim["nn_min_min"] and r["dense"] <= lim["dense_max"])


def all_metrics(X: np.ndarray, gc: GraphCache, with_realism: bool = False) -> dict:
    X = normalise(X)
    out = {
        "ksm": ksm(X, gc.D, gc.iu, gc.ju),
        "nu": node_uniformity(X),
        "eld": edge_length_deviation(X, gc.E),
        "xr": crossing_ratio(X, gc.E, gc.pairs),
    }
    if with_realism:
        rm = realism_metrics(X)
        out.update(rm)
        out["realistic"] = realism_ok(rm)
    return out


# ----------------------------------------------------------------------------
# 3. Differentiable surrogates (torch)
# ----------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def t_normalise(X):
    """Differentiable uniform-scale normalisation to the unit box."""
    X = X - X.min(0).values
    return X / (X.max() + 1e-9)


def sur_stress(X, d, iu, ju):
    """Linear-scale stress: sqrt(sum (e - a d)^2 / sum e^2), a = <e,d>/<d,d>. Upper bound of KSM."""
    torch = _torch()
    e = torch.linalg.norm(X[iu] - X[ju], dim=1)
    a = (e * d).sum() / (d * d).sum()
    return torch.sqrt(((e - a * d) ** 2).sum() / (e * e).sum() + 1e-12)


def sur_nu(X, n):
    """Soft-histogram node uniformity on the per-axis bounding box grid."""
    torch = _torch()
    g = int(math.ceil(math.sqrt(n)))
    lo = X.min(0).values
    span = (X.max(0).values - lo).clamp_min(1e-9)
    U = (X - lo) / span
    centres = (torch.arange(g, dtype=X.dtype) + 0.5) / g
    cx, cy = torch.meshgrid(centres, centres, indexing="ij")
    C = torch.stack([cx.reshape(-1), cy.reshape(-1)], 1)          # (g^2, 2)
    sigma = 0.5 / g
    d2 = ((U[:, None, :] - C[None, :, :]) ** 2).sum(-1)           # (n, g^2)
    W = torch.softmax(-d2 / (2 * sigma ** 2), dim=1)
    counts = W.sum(0)
    cbar = n / g ** 2
    num = torch.sqrt(((counts - cbar) ** 2).sum() + 1e-12)
    den = math.sqrt((n - cbar) ** 2 + (g ** 2 - 1) * cbar ** 2)
    return num / den


def sur_eld(X, E):
    torch = _torch()
    L = torch.linalg.norm(X[E[:, 0]] - X[E[:, 1]], dim=1)
    return L.std(unbiased=False) / (L.mean() + 1e-12)


def sur_xr(X, E, pa, pb, k=50.0):
    """Smooth crossing ratio. Orientation tests are normalised (sine of angle) so k is scale free."""
    torch = _torch()
    if len(pa) == 0:
        return X.sum() * 0.0
    P1, P2 = X[E[pa, 0]], X[E[pa, 1]]
    Q1, Q2 = X[E[pb, 0]], X[E[pb, 1]]

    def o(p, q, r):
        v1, v2 = q - p, r - p
        cr = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return cr / (torch.linalg.norm(v1, dim=1) * torch.linalg.norm(v2, dim=1) + 1e-9)

    s1 = -o(P1, P2, Q1) * o(P1, P2, Q2)
    s2 = -o(Q1, Q2, P1) * o(Q1, Q2, P2)
    soft = torch.sigmoid(k * s1) * torch.sigmoid(k * s2)
    return soft.sum() / len(pa)


def sur_realism(X, r_max, lam_aspect=1.0):
    """Penalise outliers (nearest-neighbour distance > r_max) and elongated drawings (aspect ratio far from 1).
    Without this, the optimiser reaches high stress or a target NU through degenerate shapes:
    a few far-away nodes stretching the bounding box, or a thin strip."""
    torch = _torch()
    n = X.shape[0]
    D = torch.cdist(X, X) + torch.eye(n, dtype=X.dtype) * 10.0
    nn = D.min(1).values
    outlier = (torch.relu(nn - r_max) ** 2).sum() / (r_max ** 2)
    span = X.max(0).values - X.min(0).values
    aspect = (span[0] - span[1]) ** 2 / (span.max() ** 2 + 1e-12)
    return outlier + lam_aspect * aspect * n


def sur_overlap(X, r_min=0.03):
    torch = _torch()
    n = X.shape[0]
    iu, ju = torch.triu_indices(n, n, offset=1)
    dist = torch.linalg.norm(X[iu] - X[ju], dim=1)
    return (torch.relu(r_min - dist) ** 2).sum()


# ----------------------------------------------------------------------------
# 4. PDSG Stage 1 — optimise one drawing toward target metric values
# ----------------------------------------------------------------------------
DEFAULT_TOL = {"ksm": 0.01, "nu": 0.02, "eld": 0.02, "xr": 0.01}


@dataclass
class PDSGResult:
    X: np.ndarray
    metrics: dict
    converged: bool
    iters: int
    restarts: int
    seconds: float
    errors: dict


def initial_layout(G, low_stress: bool, rng: np.random.Generator):
    n = G.number_of_nodes()
    if low_stress:
        pos = nx.kamada_kawai_layout(G)
        X = np.array([pos[i] for i in range(n)])
        X = normalise(X) + rng.normal(0, 0.02, size=(n, 2))
    else:
        X = rng.uniform(0, 1, size=(n, 2))
    th = rng.uniform(0, 2 * math.pi)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return normalise(X @ R.T)


def pdsg_optimise(gc: GraphCache, targets: dict, tol: dict | None = None, X0: np.ndarray | None = None,
                  max_steps: int = 2000, max_restarts: int = 5, lr: float = 0.05, check_every: int = 50,
                  k_xr: float = 50.0, lam_overlap: float = 1.0, spacing_c: float = 0.45, r_min: float | None = None,
                  outlier_c: float = 2.0, lam_realism: float = 1.0,
                  seed: int = 0, verbose: bool = False) -> PDSGResult:
    """
    targets: dict with any of keys 'ksm','nu','eld','xr' -> float target, or None/absent = free.
    Loss = sum_p ((sur_p - t~_p)/tol_p)^2 + lam_overlap * overlap.
    Spacing: overlap penalty with r_min = spacing_c / sqrt(n) (unit box) stops the optimiser reaching
    high stress by collapsing nodes into a clump (a degenerate, trivially detectable drawing).
    Realism: nearest-neighbour distance capped at outlier_c / sqrt(n); aspect ratio pulled toward 1.
    Outer correction: every `check_every` steps the surrogate target t~_p is shifted by (target - exact),
    so the optimiser converges on the EXACT metric even though it differentiates a surrogate.
    """
    torch = _torch()
    tol = {**DEFAULT_TOL, **(tol or {})}
    active = {k: v for k, v in targets.items() if v is not None}
    seed = int(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.time()

    E_t = torch.as_tensor(gc.E, dtype=torch.long)
    d_t = torch.as_tensor(gc.D[gc.iu, gc.ju], dtype=torch.float64)
    iu_t = torch.as_tensor(gc.iu, dtype=torch.long)
    ju_t = torch.as_tensor(gc.ju, dtype=torch.long)
    pa_t = torch.as_tensor(gc.pairs[0], dtype=torch.long)
    pb_t = torch.as_tensor(gc.pairs[1], dtype=torch.long)
    n = gc.n
    if r_min is None:
        r_min = spacing_c / math.sqrt(n)
    r_max = outlier_c / math.sqrt(n)

    def surrogates(Xt):
        Xn = t_normalise(Xt)
        out = {}
        if "ksm" in active:
            out["ksm"] = sur_stress(Xn, d_t, iu_t, ju_t)
        if "nu" in active:
            out["nu"] = sur_nu(Xn, n)
        if "eld" in active:
            out["eld"] = sur_eld(Xn, E_t)
        if "xr" in active:
            out["xr"] = sur_xr(Xn, E_t, pa_t, pb_t, k_xr)
        return out, Xn

    if X0 is None:
        X0 = initial_layout(gc.G, low_stress=(active.get("ksm", 0.5) <= 0.55), rng=rng)
    best = None
    total_iters = 0
    X_start = np.array(X0, dtype=np.float64)

    for restart in range(max_restarts + 1):
        Xt = torch.tensor(X_start, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([Xt], lr=lr, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=lr * 0.1)
        sur_target = dict(active)
        for step in range(1, max_steps + 1):
            opt.zero_grad()
            sur, Xn = surrogates(Xt)
            loss = sum(((sur[k] - sur_target[k]) / tol[k]) ** 2 for k in sur)
            loss = loss + lam_overlap * sur_overlap(Xn, r_min) / (r_min ** 2) + lam_realism * sur_realism(Xn, r_max)
            loss.backward()
            opt.step()
            sched.step()
            total_iters += 1
            if step % check_every == 0 or step == max_steps:
                Xnp = normalise(Xt.detach().numpy())
                m = all_metrics(Xnp, gc)
                err = {k: m[k] - active[k] for k in active}
                score = sum(abs(err[k]) / tol[k] for k in active)
                if best is None or score < best[0]:
                    best = (score, Xnp.copy(), m, err)
                if all(abs(err[k]) < tol[k] for k in active):
                    return PDSGResult(_random_flip(Xnp, rng), m, True, total_iters, restart, time.time() - t0, err)
                for k in active:  # outer correction on surrogate targets
                    sur_target[k] = sur_target[k] - 0.8 * err[k]
                if verbose and step % 500 == 0:
                    print(restart, step, {k: round(v, 4) for k, v in m.items()})
        X_start = best[1] + rng.normal(0, 0.1, size=best[1].shape)

    _, Xb, m, err = best
    return PDSGResult(_random_flip(Xb, rng), m, False, total_iters, max_restarts, time.time() - t0, err)


def _random_flip(X, rng):
    """Random reflection per axis: preserves every metric (incl. NU), unlike arbitrary rotation."""
    X = X.copy()
    if rng.random() < 0.5:
        X[:, 0] = 1 - X[:, 0]
    if rng.random() < 0.5:
        X[:, 1] = 1 - X[:, 1]
    if rng.random() < 0.5:
        X = X[:, ::-1].copy()
    return X


# ----------------------------------------------------------------------------
# 5. Rendering
# ----------------------------------------------------------------------------
RENDER_DEFAULT = dict(res=768, pad=24, node_r=6.0, edge_w=1.5, node_fill="#333333", edge_col="#000000",
                      bg="#FFFFFF", supersample=4, antialias=True)


def render_png(gc: GraphCache, X: np.ndarray, path: str | None = None, **kw):
    """Draw edges then nodes. Returns a PIL image; saves to `path` if given."""
    from PIL import Image, ImageDraw
    o = {**RENDER_DEFAULT, **kw}
    ss = o["supersample"] if o["antialias"] else 1
    R = o["res"] * ss
    pad = o["pad"] * ss
    Xn = normalise(X)
    P = pad + Xn * (R - 2 * pad)
    P[:, 1] = R - P[:, 1]  # y up
    img = Image.new("RGB", (R, R), o["bg"])
    dr = ImageDraw.Draw(img)
    ew = max(1, int(round(o["edge_w"] * ss)))
    for u, v in gc.E:
        dr.line([tuple(P[u]), tuple(P[v])], fill=o["edge_col"], width=ew)
    r = o["node_r"] * ss
    for x, y in P:
        dr.ellipse([x - r, y - r, x + r, y + r], fill=o["node_fill"])
    if ss > 1:
        img = img.resize((o["res"], o["res"]), Image.LANCZOS)
    if path:
        img.save(path, optimize=True)
    return img


# ----------------------------------------------------------------------------
# 6. Parsing model output
# ----------------------------------------------------------------------------
_ANS_RE = re.compile(r'answer"?\s*(?:is)?\s*[:=]?\s*"?\s*(image\s*)?(1|2|same|no clear difference)\b', re.I)


def parse_answer(text: str) -> dict:
    """Return {'answer': '1'|'2'|'same'|None, 'confidence': int|None, 'reason': str|None, 'parse_ok': bool}."""
    out = {"answer": None, "confidence": None, "reason": None, "parse_ok": False}
    if not text:
        return out
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            a = str(obj.get("answer", "")).strip().lower().replace("image", "").strip()
            if a in ("1", "2"):
                out["answer"] = a
            elif a in ("same", "no clear difference"):
                out["answer"] = "same"
            c = obj.get("confidence")
            out["confidence"] = int(c) if c is not None and str(c).strip().isdigit() else None
            out["reason"] = obj.get("reason")
            out["parse_ok"] = out["answer"] is not None
            if out["parse_ok"]:
                return out
        except Exception:
            pass
    m = _ANS_RE.search(text)
    if m:
        a = m.group(2).lower()
        out["answer"] = "same" if a.startswith(("same", "no")) else a
        out["parse_ok"] = True
    return out


# ----------------------------------------------------------------------------
# 7. Statistics helpers
# ----------------------------------------------------------------------------
def bootstrap_ci(values, groups=None, n_boot=10000, ci=95, seed=0, stat=np.mean):
    """Bootstrap CI of `stat`. If `groups` is given, resample groups (e.g., graph ids) with replacement."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if groups is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        boots = np.array([stat(values[i]) for i in idx])
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boots[b] = stat(values[np.concatenate([members[g] for g in pick])])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat(values)), float(lo), float(hi)
