"""Curve-skeleton graph of a 3D skeleton voxel mask (26-connectivity).

skeleton_graph(sk) -> SkGraph with
  nodes: list of dicts {id, kind ('end' | 'junc' | 'iso'), vox (k, 3) voxel indices of the node}
  edges: list of dicts {id, a, b (node ids), vox (m, 3) voxel chain from node a to node b, interior only}
A junction node is a 26-connected cluster of skeleton voxels with >= 3 skeleton neighbours. Closed loops without a
junction get one artificial node. All voxel coordinates are integer indices of the input grid.
"""

import numpy as np
from scipy import ndimage as ndi

OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


class SkGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def adj(self):
        a = {n["id"]: [] for n in self.nodes}
        for e in self.edges:
            a[e["a"]].append(e["id"])
            if e["b"] != e["a"]:
                a[e["b"]].append(e["id"])
            else:
                a[e["a"]].append(e["id"])
        return a


def skeleton_graph(sk):
    sk = sk.astype(bool)
    idx = np.argwhere(sk)
    if len(idx) == 0:
        return SkGraph([], [])
    key = {tuple(p): i for i, p in enumerate(idx)}
    nb = []
    for p in idx:
        q = p + OFFS
        nb.append([key[tuple(x)] for x in q if tuple(x) in key])
    deg = np.array([len(x) for x in nb])
    is_j = deg >= 3
    # junction clusters
    node_of = -np.ones(len(idx), int)
    nodes = []
    jm = np.zeros(sk.shape, bool)
    jm[tuple(idx[is_j].T)] = True
    lab, nl = ndi.label(jm, structure=np.ones((3, 3, 3)))
    labs = lab[tuple(idx.T)]
    nodes = []
    for c in range(1, nl + 1):
        mem = np.where(labs == c)[0]
        nid = len(nodes)
        nodes.append({"id": nid, "kind": "junc", "vox": idx[mem], "members": mem})
        node_of[mem] = nid
    for i in np.where(deg <= 1)[0]:
        nid = len(nodes)
        nodes.append({"id": nid, "kind": "end" if deg[i] == 1 else "iso", "vox": idx[[i]], "members": np.array([i])})
        node_of[i] = nid
    edges = []
    visited = np.zeros(len(idx), bool)
    visited[node_of >= 0] = True
    used_pair = set()  # (node voxel, first interior voxel) starts already traced

    def trace(start_node_vox, first):
        chain = []
        prev, cur = start_node_vox, first
        while True:
            if node_of[cur] >= 0:
                return chain, node_of[cur], prev
            chain.append(cur)
            visited[cur] = True
            nxt = [x for x in nb[cur] if x != prev and (node_of[x] >= 0 or not visited[x])]
            if len(chain) <= 2:  # do not fall straight back into the start node through a neighbouring voxel
                alt = [x for x in nxt if node_of[x] != node_of[start_node_vox]]
                if alt:
                    nxt = alt
            nn = [x for x in nxt if node_of[x] >= 0]  # prefer stopping at an adjacent node voxel
            if not nxt:
                return chain, -1, cur
            if nn:
                prev, cur = cur, nn[0]
                continue
            prev, cur = cur, nxt[0]

    for n in nodes:
        for m in n["members"]:
            for x in nb[m]:
                if node_of[x] == n["id"]:
                    continue
                if node_of[x] >= 0:
                    # two nodes directly adjacent: zero-length edge
                    pr = (min(m, x), max(m, x))
                    if pr in used_pair:
                        continue
                    used_pair.add(pr)
                    edges.append(
                        {"id": len(edges), "a": n["id"], "b": int(node_of[x]), "vox": idx[[m, x]], "chain": [m, x]}
                    )
                    continue
                if visited[x]:
                    continue
                chain, end_node, last = trace(m, x)
                if end_node < 0:
                    # dead end without node (should not happen); make an end node
                    nid = len(nodes)
                    nodes.append({"id": nid, "kind": "end", "vox": idx[[last]], "members": np.array([last])})
                    node_of[last] = nid
                    end_node = nid
                    if chain and chain[-1] == last:
                        chain = chain[:-1]
                end_vox = [x_ for x_ in nb[chain[-1]] if node_of[x_] == end_node] if chain else [last]
                full = [m] + chain + ([end_vox[0]] if end_vox else [])
                edges.append({"id": len(edges), "a": n["id"], "b": int(end_node), "vox": idx[full], "chain": full})
    # loops without any node
    rest = np.where(~visited)[0]
    while len(rest):
        i = rest[0]
        nid = len(nodes)
        nodes.append({"id": nid, "kind": "junc", "vox": idx[[i]], "members": np.array([i])})
        node_of[i] = nid
        visited[i] = True
        for x in nb[i]:
            if not visited[x]:
                chain, end_node, last = trace(i, x)
                full = [i] + chain + [i]
                edges.append({"id": len(edges), "a": nid, "b": nid, "vox": idx[full], "chain": full})
                break
        rest = np.where(~visited)[0]
    # de-duplicate edges traced from both ends
    seen = {}
    out = []
    for e in edges:
        k = (
            min(e["a"], e["b"]),
            max(e["a"], e["b"]),
            frozenset(e["chain"][1:-1]) if len(e["chain"]) > 2 else frozenset(e["chain"]),
        )
        if k in seen:
            continue
        seen[k] = 1
        e["id"] = len(out)
        out.append(e)
    for n in nodes:
        n["centre"] = n["vox"].mean(0)
    return SkGraph(nodes, out)
