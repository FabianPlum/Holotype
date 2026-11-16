"""Stage B: closed-form bone-wise initialisation from the joints.

Per bone j (segment from joint j to its child joint):  x -> P_j + s * GR_j @ S_j @ (x - J_j)
  P_j   scan joint (pivot), J_j template rest pivot, s global scale (similarity of the root-rigid joints)
  GR_j  bone rotation: Kabsch for bones with several child joints (mesosoma, head); for chain bones the minimal
        rotation from the parent's frame onto the scan bone direction plus a twist about the bone
  S_j   = g_j I + (a_j - g_j) b b^T: axial stretch a_j (scan / template bone length, relative to s) along the rest
        bone direction b, girth g_j across. Scales are per bone and do not propagate to children (each child starts
        at its own scan joint).
Vertices are blended with the skinning weights: V_i = sum_j W_ij T_j(v_i).

Twist of a chain bone (not observable from its two joints): chosen on a grid so that (1) its own local rotation and
(2) the local rotation of its child (look-ahead: the bend plane at the next joint) violate the joint limits least,
ties -> smallest twist. `refine_twists` (optional) then refines it with the scan surface.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .model import WINGS
from .pose import kabsch_uncentred, limit_violation, min_rotation, similarity_root, unresolved_bones

PSI = np.deg2rad(np.arange(-180, 180, 5.0))
PSI = PSI[np.argsort(np.abs(PSI), kind="stable")]


class BoneFit:
    """Bone transforms of one specimen. A (nJ,3,3), T (nJ,3): x_scan = A_j @ x_model + T_j."""

    def __init__(self, tpl, P):
        self.tpl = tpl
        self.P = P
        self.ok = np.isfinite(P).all(1)
        self.unres = unresolved_bones(tpl, P)
        self.s, self.R0, self.t0 = similarity_root(tpl, P)
        nJ = tpl.nJ
        self.GR = np.tile(np.eye(3), (nJ, 1, 1))
        self.piv = np.zeros((nJ, 3))  # pivot used for bone j (scan joint, or continuation for unresolved)
        self.a = np.ones(nJ)  # axial stretch relative to s
        self.g = np.ones(nJ)  # girth relative to s
        self.b = np.zeros((nJ, 3))  # rest bone direction (model)
        self.kid = np.full(nJ, -1)  # the chain child that defines the bone direction (-1: none)
        self.psi = np.zeros(nJ)
        self.twist_dof = np.zeros(nJ, bool)
        J = tpl.J0
        for j in range(nJ):
            kids = [c for c in tpl.children[j] if tpl.names[c] not in WINGS]
            if len(kids) == 1:
                self.kid[j] = kids[0]
                d = J[kids[0]] - J[j]
                self.b[j] = d / np.linalg.norm(d)
            elif len(kids) == 0 and j > 0:
                self.b[j] = self.b[tpl.parents[j]]

    # -------------------------------------------------------------------------------------------------------
    def _valid_kids(self, j):
        return [c for c in self.tpl.children[j] if self.ok[c] and self.tpl.names[c] not in WINGS]

    def _chain_rot(self, j, GRp, psi):
        """Rotation of chain bone j given its parent's rotation and a twist psi about the bone."""
        c = self.kid[j]
        u = self.P[c] - self.piv[j]
        u = u / np.linalg.norm(u)
        Rm = min_rotation(GRp @ self.b[j], u) @ GRp
        if psi == 0:
            return Rm
        return Rotation.from_rotvec(u * psi).as_matrix() @ Rm

    def _viol(self, j, GRp, GRj):
        aa = Rotation.from_matrix(GRp.T @ GRj).as_rotvec()
        return float(limit_violation(aa, self.tpl.limits[j, :, 0], self.tpl.limits[j, :, 1]))

    def _twist_cost_limits(self, j, GRp):
        """Limit violation per twist candidate: own joint + best achievable violation of the chain child."""
        tpl = self.tpl
        c = self.kid[j]
        u = self.P[c] - self.piv[j]
        u = u / np.linalg.norm(u)
        Rm = min_rotation(GRp @ self.b[j], u) @ GRp
        Rs = Rotation.from_rotvec(u[None] * PSI[:, None]).as_matrix() @ Rm  # (K,3,3)
        aa = Rotation.from_matrix(np.einsum("ab,kbc->kac", GRp.T, Rs)).as_rotvec()
        cost = limit_violation(aa, tpl.limits[j, :, 0][None], tpl.limits[j, :, 1][None])
        g = self.kid[c] if c >= 0 else -1
        if g >= 0 and self.ok[g] and self.ok[c]:
            uc = self.P[g] - self.P[c]
            uc = uc / np.linalg.norm(uc)
            look = np.zeros(len(PSI))
            for k in range(len(PSI)):
                Rc = min_rotation(Rs[k] @ self.b[c], uc) @ Rs[k]
                Rcs = Rotation.from_rotvec(uc[None] * PSI[::3, None]).as_matrix() @ Rc
                aac = Rotation.from_matrix(np.einsum("ab,kbc->kac", Rs[k].T, Rcs)).as_rotvec()
                look[k] = limit_violation(aac, tpl.limits[c, :, 0][None], tpl.limits[c, :, 1][None]).min()
            cost = cost + look
        return cost, Rs

    def solve(self, lookahead=True):
        tpl, P, J, s = self.tpl, self.P, tpl_J(self.tpl), self.s
        for j in range(tpl.nJ):
            if tpl.names[j] in WINGS:
                continue
            p = tpl.parents[j]
            GRp = self.R0 if p < 0 else self.GR[p]
            # pivot: the scan joint, else continuation by the parent's transform
            if self.ok[j]:
                self.piv[j] = P[j]
            else:
                self.piv[j] = self.apply_bone(p, J[j][None])[0] if p >= 0 else self.s * self.R0 @ J[j] + self.t0
            kids = self._valid_kids(j)
            if p < 0:
                self.GR[j] = self.R0
                self.a[j] = self.g[j] = 1.0
                continue
            if not self.ok[j] or not kids:
                # leaf, or unresolved: continuation of the parent (rest relative pose, parent's girth)
                self.GR[j] = GRp
                self.a[j] = self.g[j] = self.g[p]
                continue
            if len(kids) > 1:
                U = np.array([J[c] - J[j] for c in kids])
                V = np.array([P[c] - P[j] for c in kids]) / s
                self.GR[j] = kabsch_uncentred(U, V)
                sc = float(((U @ self.GR[j].T) * V).sum() / (U * U).sum())
                self.a[j] = self.g[j] = sc
                continue
            c = kids[0]
            if self.kid[j] < 0:
                # a bone with several child joints of which only one is given (a head with one antennal socket):
                # minimal rotation from the parent's frame, isotropic scale; the surface alignment has to find the rest
                d0, d1 = J[c] - J[j], P[c] - P[j]
                self.GR[j] = min_rotation(GRp @ d0, d1) @ GRp
                self.a[j] = self.g[j] = float(np.clip(np.linalg.norm(d1) / (s * np.linalg.norm(d0)), 0.5, 2.0))
                continue
            self.twist_dof[j] = True
            if lookahead:
                cost, Rs = self._twist_cost_limits(j, GRp)
                k = int(np.argmin(cost + 1e-6 * np.abs(PSI)))
                self.GR[j], self.psi[j] = Rs[k], PSI[k]
            else:
                self.GR[j] = self._chain_rot(j, GRp, 0.0)
            self.a[j] = np.linalg.norm(P[c] - P[j]) / (s * np.linalg.norm(J[c] - J[j]))
            self.g[j] = 1.0
        # the root bone (mesosoma): isotropic scale from its landmark joints is s itself
        return self

    # -------------------------------------------------------------------------------------------------------
    def S(self, j):
        b = self.b[j]
        return self.g[j] * np.eye(3) + (self.a[j] - self.g[j]) * np.outer(b, b)

    def affine(self):
        """A (nJ,3,3), T (nJ,3) with x_scan = A x_model + T."""
        tpl = self.tpl
        J = tpl_J(tpl)
        A = np.zeros((tpl.nJ, 3, 3))
        T = np.zeros((tpl.nJ, 3))
        for j in range(tpl.nJ):
            A[j] = self.s * self.GR[j] @ self.S(j)
            T[j] = self.piv[j] - A[j] @ J[j]
        return A, T

    def apply_bone(self, j, X):
        A = self.s * self.GR[j] @ self.S(j)
        return (X - tpl_J(self.tpl)[j]) @ A.T + self.piv[j]

    def vertices(self, A=None, T=None):
        if A is None:
            A, T = self.affine()
        return blend(self.tpl, A, T)


def tpl_J(tpl):
    return tpl.J0


def blend(tpl, A, T, V0=None):
    V0 = tpl.V0 if V0 is None else V0
    WA = np.einsum("vj,jab->vab", tpl.W, A)
    return np.einsum("vab,vb->va", WA, V0) + tpl.W @ T
