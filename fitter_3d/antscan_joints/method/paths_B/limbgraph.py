"""Limb skeleton graph: the curve skeleton of the scan's solid outside the trunk core, as a mutable graph with
polyline edges (micrometres, working frame), radius profiles, trunk attachments, spur pruning and junction clusters.

Nodes: {pos (3,), kind 'end' | 'junc', r (radius at the node, um), root (trunk part label 0..3 the node touches, or
-1)}. Edges: {a, b, P (n, 3) polyline from node a to node b, r (n,) radius along it}.
"""

import numpy as np
from scipy import ndimage as ndi

import skgraph


def arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))] if len(P) > 1 else np.zeros(len(P))


class LimbGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self._nid = 0
        self._eid = 0

    # ------------------------------------------------------------------ basic ops
    def add_node(self, **kw):
        i = self._nid
        self._nid += 1
        self.nodes[i] = dict(kw)
        return i

    def add_edge(self, a, b, P, r):
        i = self._eid
        self._eid += 1
        self.edges[i] = {"a": a, "b": b, "P": np.asarray(P, float), "r": np.asarray(r, float)}
        return i

    def inc(self, n):
        return [e for e, d in self.edges.items() if d["a"] == n or d["b"] == n]

    def degree(self, n):
        return sum((d["a"] == n) + (d["b"] == n) for d in self.edges.values())

    def other(self, e, n):
        d = self.edges[e]
        return d["b"] if d["a"] == n else d["a"]

    def oriented(self, e, n):
        """Polyline and radii of edge e oriented to start at node n."""
        d = self.edges[e]
        if d["a"] == n:
            return d["P"], d["r"]
        return d["P"][::-1], d["r"][::-1]

    def length(self, e):
        return float(arclen(self.edges[e]["P"])[-1])

    def remove_edge(self, e):
        del self.edges[e]

    def merge_deg2(self, n):
        """Replace a degree-2 non-root node by one edge."""
        es = self.inc(n)
        if len(es) != 2 or es[0] == es[1] or self.nodes[n]["root"] >= 0:
            return False
        P1, r1 = self.oriented(es[0], n)
        P2, r2 = self.oriented(es[1], n)
        a, b = self.other(es[0], n), self.other(es[1], n)
        P = np.vstack([P1[::-1], P2[1:]])
        r = np.r_[r1[::-1], r2[1:]]
        self.remove_edge(es[0])
        self.remove_edge(es[1])
        del self.nodes[n]
        self.add_edge(a, b, P, r)
        return True

    def cleanup(self):
        changed = True
        while changed:
            changed = False
            for n in list(self.nodes):
                if n not in self.nodes:
                    continue
                dg = self.degree(n)
                if dg == 0 and self.nodes[n]["root"] < 0:
                    del self.nodes[n]
                    changed = True
                elif dg == 2 and self.nodes[n]["kind"] == "junc":
                    if self.merge_deg2(n):
                        changed = True
        for n in self.nodes:
            dg = self.degree(n)
            if self.nodes[n]["kind"] == "junc" and dg == 1:
                self.nodes[n]["kind"] = "end"


def build(occ, core_lab, lo, h, D_um, sk=None):
    """occ: fine occupancy; core_lab: int8 grid (-1 none, 0..3 trunk core parts); lo: centre of voxel 0; h: pitch;
    D_um: distance transform of occ in um. Returns (LimbGraph, skeleton mask)."""
    from skimage.morphology import skeletonize

    if sk is None:
        sk = skeletonize(occ) > 0
    core = core_lab >= 0
    limb_sk = sk & ~core
    G0 = skgraph.skeleton_graph(limb_sk)
    # trunk attachment: node voxel within 1 voxel (26-nbhd) of a core voxel; label = majority core part nearby
    dist, inds = ndi.distance_transform_edt(~core, return_indices=True)
    G = LimbGraph()
    nmap = {}
    shp = np.array(occ.shape)  # noqa: F841
    for nd in G0.nodes:
        vx = nd["vox"]
        dd = dist[tuple(vx.T)]
        root = -1
        if dd.min() <= 1.75:
            k = vx[np.argmin(dd)]
            q = inds[:, k[0], k[1], k[2]]
            root = int(core_lab[q[0], q[1], q[2]])
        c = nd["vox"].mean(0)
        nmap[nd["id"]] = G.add_node(
            pos=lo + c * h,
            kind=nd["kind"] if nd["kind"] != "iso" else "end",
            r=float(D_um[tuple(vx.T)].max()),
            root=root,
            vox=vx,
        )
    for e in G0.edges:
        vx = e["vox"]
        P = lo + vx * h
        r = D_um[tuple(vx.T)]
        G.add_edge(nmap[e["a"]], nmap[e["b"]], P, r)
    # isolated nodes that are not roots are dropped; self loops kept
    G.cleanup()
    return G, sk


def _reach(G, e, end, jn, W):
    """Distance from the free end of terminal edge e to a reference point W um into the junction's other edges
    (mean over them): a straight continuation of the tube reaches further than a side spur of the same length."""
    refs = []
    inc = [e2 for e2 in G.inc(jn) if e2 != e]
    nonterm = [e2 for e2 in inc if G.nodes[G.other(e2, jn)]["kind"] != "end" or G.nodes[G.other(e2, jn)]["root"] >= 0]
    for e2 in nonterm or inc:
        P, _ = G.oriented(e2, jn)
        s = arclen(P)
        t = min(W, s[-1])
        refs.append(np.array([np.interp(t, s, P[:, k]) for k in range(3)]))
    if not refs:
        return G.length(e)
    tipp = G.nodes[end]["pos"]
    return float(np.mean([np.linalg.norm(tipp - q) for q in refs]))


def prune_spurs(G, BL, c_r=2.0, c_bl=1.5, max_iter=50, reach_pct=3.0):
    """Remove terminal edges (free end -> junction) shorter than max(c_r x junction radius, c_bl % BL), one per
    junction and round, repeated until stable; junctions left with degree 2 are merged. When a junction holds more
    than one terminal edge (claws, a tarsal seta next to the tarsus tip, a forked club end), the one removed is the one
    with the smallest reach (distance of its free end from the junction's other side), so the tube's own tip, which
    continues straight on, is kept even when a side spur is longer."""
    W = reach_pct / 100 * BL
    for _ in range(max_iter):
        term = {}
        for e, d in G.edges.items():
            for end, jn in ((d["a"], d["b"]), (d["b"], d["a"])):
                if end == jn:
                    continue
                ne = G.nodes[end]
                if ne["kind"] == "end" and ne["root"] < 0 and G.degree(end) == 1 and G.degree(jn) >= 3:
                    term.setdefault(jn, []).append((e, end))
        cand = []
        for jn, lst in term.items():
            thr = max(c_r * G.nodes[jn]["r"], c_bl / 100 * BL)
            short = [(e, end) for e, end in lst if G.length(e) < thr]
            if not short:
                continue
            if len(lst) >= 2 and len(lst) == len(short) and G.degree(jn) - len(lst) >= 1:
                # several short terminal edges: drop the one reaching least far
                e, end = min(short, key=lambda x: _reach(G, x[0], x[1], jn, W))
            elif len(lst) >= 2:
                long_ = [x for x in lst if x not in short]
                e, end = min(short, key=lambda x: _reach(G, x[0], x[1], jn, W)) if long_ else short[0]
            else:
                e, end = short[0]
            cand.append((G.length(e), e, end))
        if not cand:
            break
        for L, e, end in cand:
            if e not in G.edges:
                continue
            G.remove_edge(e)
            del G.nodes[end]
        G.cleanup()
    return G
