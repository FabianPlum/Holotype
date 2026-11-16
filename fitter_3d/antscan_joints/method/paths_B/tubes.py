"""Junction clusters, pairing of edges through a junction (a tube passing straight through a contact), and tube
extraction from a pruned LimbGraph.

A junction outside the trunk is either a contact (two tubes touching: X when they cross, T when one tube's end
touches the side of another) or a remaining fork. Edges that continue each other through a junction cluster (small
turn, similar radius) are paired; an unpaired edge ends at the cluster (its tube ends at a contact).
"""

import itertools  # noqa: F401

import numpy as np

from limbgraph import arclen


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def clusters(G, c_short=1.5):
    """Group junction nodes joined by short edges (length < c_short x (r_a + r_b)). Root nodes stay alone.
    Returns (cluster id per node, internal edges)."""
    uf = UF()
    internal = set()
    for n, d in G.nodes.items():
        uf.find(n)
    for e, d in G.edges.items():
        a, b = d["a"], d["b"]
        if a == b:
            continue
        na, nb = G.nodes[a], G.nodes[b]
        if na["kind"] != "junc" or nb["kind"] != "junc" or na["root"] >= 0 or nb["root"] >= 0:
            continue
        if G.length(e) < c_short * (na["r"] + nb["r"]):
            uf.union(a, b)
            internal.add(e)
    cl = {n: uf.find(n) for n in G.nodes}
    # internal = edges whose both ends ended up in the same cluster (including loops inside a cluster)
    internal = {
        e
        for e, d in G.edges.items()
        if cl[d["a"]] == cl[d["b"]] and G.nodes[d["a"]]["kind"] == "junc" and G.nodes[d["a"]]["root"] < 0
    }
    return cl, internal


def edge_dir(P, r, s0, W):
    """Direction (unit) of polyline P (oriented away from a junction) between arc s0 and s0 + W, and the median
    radius there."""
    s = arclen(P)
    L = s[-1]
    if L <= 1e-9:
        return np.zeros(3), float(np.median(r))
    a = min(s0, 0.5 * L)
    b = min(a + W, L)
    pa = np.array([np.interp(a, s, P[:, k]) for k in range(3)])
    pb = np.array([np.interp(b, s, P[:, k]) for k in range(3)])
    u = pb - pa
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        u, nu = P[-1] - P[0], np.linalg.norm(P[-1] - P[0])
    m = (s >= a) & (s <= b)
    rr = float(np.median(r[m])) if m.any() else float(np.median(r))
    return (u / nu if nu > 0 else np.zeros(3)), rr


class _BudgetExceeded(Exception):
    pass


def best_matching(ends, score, budget=200000):
    """Maximum total score matching over a small set of edge ends; score(i, j) -> float or None (not allowed).

    Exhaustive recursion (unchanged result) while it visits at most `budget` nodes. A junction cluster with many ends
    makes the exhaustive search explode: the number of matchings grows faster than exponentially (one production
    scan: one cluster with 32 ends and 110 allowed pairs = 3.7e10 matchings, the deterministic 1800 s stall
    of PATHS-B). Beyond the budget the maximum-weight matching is computed with networkx (Edmonds' blossom algorithm,
    polynomial). Every allowed score is > 0 (>= 1 + cos(max_turn) - w_ratio ln(max_ratio) = 1.04 with the defaults), so
    a maximum-weight matching is also maximal, as the exhaustive tie rule (more pairs on equal score) prefers; exact
    score ties may still be broken differently. Scores are computed once per pair (score is symmetric)."""
    n = len(ends)
    S = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = score(i, j)
            if s is not None:
                S[(i, j)] = s
    best = (0.0, [])
    visits = [0]

    def rec(rem, acc, tot):
        nonlocal best
        visits[0] += 1
        if visits[0] > budget:
            raise _BudgetExceeded
        if tot > best[0] or (tot == best[0] and len(acc) > len(best[1])):
            best = (tot, list(acc))
        if len(rem) < 2:
            return
        i = rem[0]
        rest = rem[1:]
        rec(rest, acc, tot)  # i unpaired
        for j in rest:
            s = S.get((i, j))
            if s is None:
                continue
            rec([x for x in rest if x != j], acc + [(i, j)], tot + s)

    try:
        rec(list(range(n)), [], 0.0)
        return best[1]
    except _BudgetExceeded:
        import networkx as nx

        G = nx.Graph()
        for (i, j), s in S.items():
            G.add_edge(i, j, weight=s)
        M = nx.max_weight_matching(G, maxcardinality=False)
        return sorted(tuple(sorted(p)) for p in M)


def pair_junctions(G, cl, internal, BL, max_turn=60.0, max_ratio=2.5, w_ratio=0.5):
    """Pair external edge ends at every junction cluster. Returns dict cluster -> {"ends": [(e, node)], "pairs":
    [(k1, k2)], "dirs", "rads", "centre", "r"}."""
    by_c = {}
    for n, d in G.nodes.items():
        if d["kind"] != "junc" or d["root"] >= 0:
            continue
        by_c.setdefault(cl[n], []).append(n)
    out = {}
    cos_max = np.cos(np.radians(max_turn))
    for c, mem in by_c.items():
        centre = np.mean([G.nodes[m]["pos"] for m in mem], axis=0)
        rC = max(G.nodes[m]["r"] for m in mem)
        ends = []
        for e, d in G.edges.items():
            if e in internal:
                continue
            for nd in (d["a"], d["b"]):
                if nd in mem:
                    ends.append((e, nd))
            # a self loop at a cluster node appears twice (both ends)
        dirs, rads = [], []
        for e, nd in ends:
            P, r = G.oriented(e, nd)
            # start the direction window outside the cluster's own radius
            dist0 = np.linalg.norm(P - centre, axis=1)
            W = float(np.clip(3.0 * np.median(r), 0.015 * BL, 0.04 * BL))
            s = arclen(P)
            out_i = np.where(dist0 > 1.0 * rC)[0]
            s0 = s[out_i[0]] if len(out_i) else 0.0
            u, rr = edge_dir(P, r, s0, W)
            dirs.append(u)
            rads.append(rr)

        def score(i, j):
            if ends[i][0] == ends[j][0]:
                return None  # both ends of a self loop
            cosv = -float(np.dot(dirs[i], dirs[j]))
            ratio = max(rads[i], rads[j]) / max(min(rads[i], rads[j]), 1e-6)
            if cosv < cos_max or ratio > max_ratio:
                return None
            return 1.0 + cosv - w_ratio * np.log(ratio)

        pairs = best_matching(ends, score) if len(ends) >= 2 else []
        out[c] = {"members": mem, "ends": ends, "pairs": pairs, "dirs": dirs, "rads": rads, "centre": centre, "r": rC}
    return out


def extract_tubes(G, cl, internal, J):
    """Walk through paired junctions. Returns a list of tubes: {edges [(e, from_node)], P, r, start, end} where start /
    end = {"kind": "tip" | "root" | "contact" | "loop", "node", "cluster", "root_part"}."""
    partner = {}  # (e, node) -> (e2, node2) through the cluster
    for c, info in J.items():
        for i, j in info["pairs"]:
            a, b = info["ends"][i], info["ends"][j]
            partner[a] = b
            partner[b] = a
    in_cluster = {n for c, info in J.items() for n in info["members"]}

    def term(node, e):
        nd = G.nodes[node]
        if node in in_cluster:
            return {"kind": "contact", "node": node, "cluster": cl[node], "root_part": -1}
        if nd["root"] >= 0:
            return {"kind": "root", "node": node, "cluster": None, "root_part": nd["root"]}
        if nd["kind"] == "end":
            return {"kind": "tip", "node": node, "cluster": None, "root_part": -1}
        return {"kind": "contact", "node": node, "cluster": cl[node], "root_part": -1}

    used = set()
    tubes = []
    starts = []
    for e, d in G.edges.items():
        if e in internal:
            continue
        for nd in (d["a"], d["b"]):
            if (e, nd) not in partner:
                starts.append((e, nd))
    for e0, n0 in starts + [(e, d["a"]) for e, d in G.edges.items() if e not in internal]:
        if e0 in used:
            continue
        seq = []
        e, n = e0, n0
        start = (
            term(n0, e0)
            if (e0, n0) not in partner
            else {"kind": "loop", "node": n0, "cluster": cl[n0], "root_part": -1}
        )
        end = None
        while True:
            if e in used:
                end = {"kind": "loop", "node": n, "cluster": cl.get(n), "root_part": -1}
                break
            used.add(e)
            seq.append((e, n))
            m = G.other(e, n)
            if (e, m) in partner:
                e, n = partner[(e, m)]
                continue
            end = term(m, e)
            break
        Ps, rs = [], []
        for k, (e_, n_) in enumerate(seq):
            P, r = G.oriented(e_, n_)
            if k:
                P, r = P[1:], r[1:]
            Ps.append(P)
            rs.append(r)
        P = np.vstack(Ps)
        r = np.concatenate(rs)
        tubes.append({"edges": seq, "P": P, "r": r, "start": start, "end": end})
    return tubes
