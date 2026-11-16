"""Good-continuation routes on the limb skeleton graph.

Junction clusters are contracted to super-nodes. A route from a free tip to the trunk is a shortest path over
directed edges where passing a junction cluster from edge i to edge j costs a turn penalty (the angle between the
tube direction arriving on i and leaving on j) plus a radius penalty when the tube gets thinner towards the root
(limbs thicken proximally; a route that thins proximally has jumped into another, thinner limb through a contact).
Routes are claimed one by one (cleanest first); a claimed route's edges become expensive for the others, so a tip
whose cheapest route ran through a claimed limb is rerouted along its own tube when one exists.
"""

import heapq

import numpy as np

from limbgraph import arclen  # noqa: F401


def turn_pen(cosv, BL, lam):
    """cosv = cos of the angle between the arriving direction and the leaving direction (1 = straight on)."""
    th = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
    return lam * BL * (th / 90.0) ** 2


class Router:
    def __init__(self, G, cl, internal, J, BL, lam_turn=0.05, mu_rad=0.05, rad_free=1.3):
        self.G, self.cl, self.internal, self.J, self.BL = G, cl, internal, J, BL
        self.lam, self.mu, self.rfree = lam_turn, mu_rad, rad_free
        self.in_cluster = {n: c for c, info in J.items() for n in info["members"]}
        # end index lookup per cluster
        self.end_idx = {}
        for c, info in J.items():
            for k, (e, nd) in enumerate(info["ends"]):
                self.end_idx[(e, nd)] = (c, k)
        self.ext_edges = [e for e in G.edges if e not in internal]
        self.len = {e: G.length(e) for e in self.ext_edges}
        # node incidence (external edges only)
        self.inc = {}
        for e in self.ext_edges:
            d = G.edges[e]
            self.inc.setdefault(d["a"], []).append(e)
            if d["b"] != d["a"]:
                self.inc.setdefault(d["b"], []).append(e)

    def trans_cost(self, c, i, j):
        info = self.J[c]
        cosv = -float(np.dot(info["dirs"][i], info["dirs"][j]))
        pen = turn_pen(cosv, self.BL, self.lam)
        ri, rj = info["rads"][i], info["rads"][j]  # i = distal side (arriving), j = proximal side (leaving)
        if rj * self.rfree < ri:
            pen += self.mu * self.BL * np.log(ri / (rj * self.rfree))
        return pen

    def route(self, tip, roots_ok, edge_extra=None, max_cost=None):
        """Cheapest route from node `tip` to any root node whose part is in roots_ok. edge_extra: {edge: extra cost}.
        Returns (cost, [(edge, from_node)], end node, penalty sum) or None."""
        G = self.G
        edge_extra = edge_extra or {}
        start = [(e, tip) for e in self.inc.get(tip, [])]
        best = {}
        heap = []
        for e, n in start:
            c0 = self.len[e] + edge_extra.get(e, 0.0)
            heapq.heappush(heap, (c0, 0.0, e, n, None))
            best[(e, n)] = (c0, None, 0.0)
        done = set()
        while heap:
            cost, pen, e, n, prev = heapq.heappop(heap)
            if (e, n) in done:
                continue
            done.add((e, n))
            m = G.other(e, n)
            nd = G.nodes[m]
            if nd["root"] >= 0:
                if nd["root"] in roots_ok:
                    seq = [(e, n)]
                    k = (e, n)
                    while best[k][1] is not None:
                        k = best[k][1]
                        seq.append(k)
                    return cost, seq[::-1], m, pen
                continue
            if max_cost is not None and cost > max_cost:
                continue
            if m in self.in_cluster:
                c = self.in_cluster[m]
                ci, ki = self.end_idx[(e, m)]
                for kj, (e2, n2) in enumerate(self.J[c]["ends"]):
                    if kj == ki or e2 == e:
                        continue
                    tc = self.trans_cost(c, ki, kj)
                    nc = cost + tc + self.len[e2] + edge_extra.get(e2, 0.0)
                    key = (e2, n2)
                    if key not in best or nc < best[key][0]:
                        best[key] = (nc, (e, n), pen + tc)
                        heapq.heappush(heap, (nc, pen + tc, e2, n2, (e, n)))
            else:
                for e2 in self.inc.get(m, []):
                    if e2 == e:
                        continue
                    nc = cost + self.len[e2] + edge_extra.get(e2, 0.0)
                    key = (e2, m)
                    if key not in best or nc < best[key][0]:
                        best[key] = (nc, (e, n), pen)
                        heapq.heappush(heap, (nc, pen, e2, m, (e, n)))
        return None

    def polyline(self, seq):
        Ps, rs = [], []
        for k, (e, n) in enumerate(seq):
            P, r = self.G.oriented(e, n)
            Ps.append(P if k == 0 else P[1:])
            rs.append(r if k == 0 else r[1:])
        return np.vstack(Ps), np.concatenate(rs)


def tips_of(G, internal):
    return [
        n
        for n, d in G.nodes.items()
        if d["kind"] == "end" and d["root"] < 0 and any(e not in internal for e in G.inc(n))
    ]


def claim_routes(
    router, tips, BL, families=((1, 2), (0,)), block_factor=3.0, min_len_pct=3.0, spur_max_pct=4.0, max_overlap=0.5
):
    """Route every tip to the trunk (both families; the cheaper relative to its length wins), then claim routes
    cleanest-first; a route overlapping claimed edges (outside the root zone) is rerouted with claimed edges costed
    block_factor x their length extra. Returns list of routes {tip, seq, P, r, cost, pen, root_node, root_part,
    overlap}."""
    G = router.G
    first = []
    for t in tips:
        best = None
        for fam in families:
            r = router.route(t, set(fam))
            if r is None:
                continue
            cost, seq, m, pen = r
            L = sum(router.len[e] for e, _ in seq)
            if L < min_len_pct / 100 * BL:
                continue
            key = cost - L  # penalty-only (detour) cost
            if best is None or key < best[0]:
                best = (key, cost, seq, m, pen, L)
        if best is not None:
            first.append((best[4] / max(best[5], 1e-9), t, best))
    first.sort(key=lambda x: (x[0], -x[2][5]))
    claimed_edges = {}
    routes = []
    dropped = set()

    def own_before(edges_, owner_set):
        a = 0.0
        for e in edges_:
            if e in owner_set:
                break
            a += router.len[e]
        return a

    def first_shared(edges_, owner_set):
        return next((i for i, e in enumerate(edges_) if e in owner_set), None)

    def trans_at(seq_, i):
        """Transition cost of a route from seq_[i-1] into seq_[i] (0 when no junction cluster is passed)."""
        if i is None or i <= 0:
            return 0.0
        e1, n1 = seq_[i - 1]
        m = G.other(e1, n1)
        e2, n2 = seq_[i]
        if m not in router.in_cluster or (e1, m) not in router.end_idx or (e2, n2) not in router.end_idx:
            return 0.0
        c, ki = router.end_idx[(e1, m)]
        c2, kj = router.end_idx[(e2, n2)]
        return router.trans_cost(c, ki, kj) if c == c2 else 0.0

    for _, t, (key, cost, seq, m, pen, L) in first:
        edges = [e for e, _ in seq]
        ov = sum(router.len[e] for e in edges if e in claimed_edges)
        if ov > 0.25 * L:
            own = own_before(edges, claimed_edges)
            # the claimed route(s) this one joins: if their own distal part (before the shared edges) is a short
            # side branch and this route's own part is longer, this route is the limb and the claimed one a spur
            hit = sorted({claimed_edges[e] for e in edges if e in claimed_edges})
            replaced = False
            if len(hit) == 1 and hit[0] not in dropped:
                C = routes[hit[0]]
                ce = [e for e, _ in C["seq"]]
                own_c = own_before(ce, set(edges))
                t_r = trans_at(seq, first_shared(edges, claimed_edges))
                t_c = trans_at(C["seq"], first_shared(ce, set(edges)))
                spur_like = own_c < spur_max_pct / 100 * BL and own > own_c
                side_branch = own_c < 0.5 * own and t_r < 0.5 * t_c  # C turned in where R runs straight on
                if spur_like or side_branch:
                    dropped.add(C["id"])
                    for e in ce:
                        if claimed_edges.get(e) == C["id"]:
                            del claimed_edges[e]
                    ov = sum(router.len[e] for e in edges if e in claimed_edges)
                    replaced = True
            if not replaced:
                if own < spur_max_pct / 100 * BL:
                    continue  # a side branch (spur, claw, hair) of a limb already claimed
                extra = {e: block_factor * router.len[e] for e in claimed_edges}
                best = None
                for fam in families:
                    r = router.route(t, set(fam), edge_extra=extra)
                    if r is None:
                        continue
                    c2, s2, m2, p2 = r
                    L2 = sum(router.len[e] for e, _ in s2)
                    if best is None or c2 - L2 < best[0]:
                        best = (c2 - L2, c2, s2, m2, p2, L2)
                if best is not None:
                    key, cost, seq, m, pen, L = best
                    edges = [e for e, _ in seq]
                    ov = sum(router.len[e] for e in edges if e in claimed_edges)
                if ov > max_overlap * L:
                    continue
        P, r = router.polyline(seq)
        rid = len(routes)
        routes.append(
            {
                "id": rid,
                "tip": t,
                "seq": seq,
                "P": P,
                "r": r,
                "cost": cost,
                "pen": pen,
                "len": L,
                "root_node": m,
                "root_part": G.nodes[m]["root"],
                "overlap": ov / max(L, 1e-9),
                "overlap_with": sorted({claimed_edges[e] for e in edges if e in claimed_edges}),
            }
        )
        for e in edges:
            claimed_edges.setdefault(e, rid)
    routes = [ro for ro in routes if ro["id"] not in dropped]
    return routes, claimed_edges


def flow_routes(router, tips, BL, families=((1, 2), (0,)), kappa=1.5, min_len_pct=3.0):
    """Edge-disjoint tip -> trunk routes chosen jointly (min-cost flow on the directed-edge graph): every skeleton edge
    carries at most one limb, passing a junction cluster costs the turn / radius penalty, and a tip that reaches the
    trunk earns kappa x the length of its own cheapest route. A spur and the tarsus it branches from then compete for
    the same tibia and the longer limb wins; a tip whose cheapest route runs through a contact into another limb is
    sent along its own tube when the other limb needs those edges."""
    import networkx as nx

    G = router.G
    roots_ok = {p for fam in families for p in fam}
    bonus, first = {}, {}
    for t in tips:
        best = None
        for fam in families:
            r = router.route(t, set(fam))
            if r is None:
                continue
            cost, seq, m, pen = r
            L = sum(router.len[e] for e, _ in seq)
            if L < min_len_pct / 100 * BL:
                continue
            if best is None or cost - L < best[0]:
                best = (cost - L, cost, seq, m, pen, L)
        if best is not None:
            bonus[t] = kappa * best[5]
            first[t] = best
    N = nx.DiGraph()
    S, T = "S", "T"
    ext = [e for e in router.ext_edges if G.edges[e]["a"] != G.edges[e]["b"]]
    for e in ext:
        d = G.edges[e]
        N.add_edge(("mi", e), ("mo", e), capacity=1, weight=int(round(router.len[e])))
        for nd in (d["a"], d["b"]):
            N.add_edge(("in", e, nd), ("mi", e), capacity=1, weight=0)
            N.add_edge(("mo", e), ("out", e, nd), capacity=1, weight=0)
    for c, info in router.J.items():
        ends = [(e, nd) for e, nd in info["ends"] if e in set(ext)]
        for ki, (ei, ni) in enumerate(info["ends"]):
            if (ei, ni) not in ends:
                continue
            for kj, (ej, nj) in enumerate(info["ends"]):
                if kj == ki or ej == ei or (ej, nj) not in ends:
                    continue
                N.add_edge(("out", ei, ni), ("in", ej, nj), capacity=1, weight=int(round(router.trans_cost(c, ki, kj))))
    for e in ext:
        d = G.edges[e]
        for nd in (d["a"], d["b"]):
            if G.nodes[nd]["root"] in roots_ok and nd not in router.in_cluster:
                N.add_edge(("out", e, nd), T, capacity=1, weight=0)
    ntip = 0
    for t, b in bonus.items():
        for e in router.inc.get(t, []):
            if e in set(ext):
                N.add_edge(S, ("tipnode", t), capacity=1, weight=-int(round(b)))
                N.add_edge(("tipnode", t), ("in", e, t), capacity=1, weight=0)
                ntip += 1
    if ntip == 0:
        return [], {}
    N.add_edge(S, T, capacity=ntip, weight=0)
    N.nodes[S]["demand"] = -ntip
    N.nodes[T]["demand"] = ntip
    flow = nx.min_cost_flow(N)
    routes, claimed = [], {}
    for t in bonus:
        if flow[S].get(("tipnode", t), 0) < 1:
            continue
        node = ("tipnode", t)
        seq = []
        cur_from = t
        guard = 0
        while node != T and guard < 10000:
            guard += 1
            nxt = [v for v, f in flow[node].items() if f > 0]
            if not nxt:
                break
            v = nxt[0]
            flow[node][v] -= 1
            if v[0] == "in":
                cur_from = v[2]
                seq.append((v[1], cur_from))
            node = v
        if node != T or not seq:
            continue
        m = G.other(seq[-1][0], seq[-1][1])
        P, r = router.polyline(seq)
        L = sum(router.len[e] for e, _ in seq)
        pen = 0.0
        for k in range(1, len(seq)):
            e1, n1 = seq[k - 1]
            mm = G.other(e1, n1)
            if mm in router.in_cluster and (e1, mm) in router.end_idx and seq[k] in router.end_idx:
                c, ki = router.end_idx[(e1, mm)]
                c2, kj = router.end_idx[seq[k]]
                if c2 == c:
                    pen += router.trans_cost(c, ki, kj)
        rid = len(routes)
        routes.append(
            {
                "id": rid,
                "tip": t,
                "seq": seq,
                "P": P,
                "r": r,
                "cost": L + pen,
                "pen": pen,
                "len": L,
                "root_node": m,
                "root_part": G.nodes[m]["root"],
                "overlap": 0.0,
                "overlap_with": [],
                "detour": L + pen - first[t][1],
            }
        )
        for e, _ in seq:
            claimed[e] = rid
    return routes, claimed


def add_secondary(
    router,
    routes,
    claimed,
    tips,
    BL,
    families=((1, 2), (0,)),
    block_factor=3.0,
    min_len_pct=3.0,
    spur_max_pct=4.0,
    max_overlap=0.5,
):
    """Tips left without a route by the flow get their cheapest route with the flow's edges made expensive; kept as
    secondary candidates when they share at most max_overlap of their length with flow routes and their own part
    before the shared edges is longer than a spur (the naming step resolves conflicts)."""
    G = router.G
    have = {ro["tip"] for ro in routes}
    extra = {e: block_factor * router.len[e] for e in claimed}
    out = list(routes)
    for t in tips:
        if t in have:
            continue
        best = None
        for fam in families:
            r = router.route(t, set(fam), edge_extra=extra)
            if r is None:
                continue
            c2, s2, m2, p2 = r
            L2 = sum(router.len[e] for e, _ in s2)
            if L2 < min_len_pct / 100 * BL:
                continue
            if best is None or c2 - L2 < best[0]:
                best = (c2 - L2, c2, s2, m2, p2, L2)
        if best is None:
            continue
        _, cost, seq, m, pen, L = best
        edges = [e for e, _ in seq]
        ov = sum(router.len[e] for e in edges if e in claimed)
        own = 0.0
        for e in edges:
            if e in claimed:
                break
            own += router.len[e]
        if ov > max_overlap * L or (ov > 0 and own < spur_max_pct / 100 * BL):
            continue
        P, r = router.polyline(seq)
        out.append(
            {
                "id": len(out),
                "tip": t,
                "seq": seq,
                "P": P,
                "r": r,
                "cost": cost,
                "pen": pen,
                "len": L,
                "root_node": m,
                "root_part": G.nodes[m]["root"],
                "overlap": ov / max(L, 1e-9),
                "overlap_with": sorted({claimed[e] for e in edges if e in claimed}),
                "secondary": True,
            }
        )
    return out
