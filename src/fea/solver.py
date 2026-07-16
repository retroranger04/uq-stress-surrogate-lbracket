"""
FEniCSx plane-stress linear-elasticity solver for the L-bracket.

Consumes a .msh produced by src/fea/mesh.py and returns the displacement and
von Mises stress fields along with the peak stress value.

Every modeling choice here is deliberate; each is annotated inline. The same
justifications appear in the accompanying paper's Methods section.

Run environment: Kaggle CPU notebook with FEniCSx (dolfinx) installed. This
module imports dolfinx lazily so the rest of the package remains importable
on a Windows-native dev machine where FEniCSx is not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import E_MPA, NU, T_OUT_OF_PLANE_MM, B_HORIZ_LEN_MM
from .geometry import LBracketParams


# Physical-group tags must stay in sync with build_geo() in geometry.py.
TAG_DOMAIN = 1
TAG_CLAMPED = 10
TAG_LOADED = 20
TAG_FILLET = 30
TAG_HOLE1 = 40
TAG_HOLE2 = 41


@dataclass
class FEAResult:
    params: LBracketParams
    load_w_mpa: float              # applied distributed load [MPa]
    peak_vm_mpa: float             # peak von Mises stress over all DOFs
    peak_location_xy: tuple        # (x, y) of the peak vm DOF, mm
    n_dofs: int                    # problem size (scalar field count)
    h_fine: float                  # mesh refinement at fillet
    h_coarse: float                # far-field mesh size
    vm_field: Optional[np.ndarray] = None   # nodal von Mises [MPa]
    coords: Optional[np.ndarray] = None     # nodal (x,y) coords [mm]


def solve_lbracket(msh_path,
                   params: LBracketParams,
                   load_w_mpa: float,
                   h_fine: float,
                   h_coarse: float,
                   return_fields: bool = True) -> FEAResult:
    """Solve 2D plane-stress linear elasticity on the given L-bracket mesh.

    Parameters
    ----------
    msh_path : path-like
        .msh file (quadratic triangles, physical groups per geometry.build_geo).
    params : LBracketParams
        Design parameters — recorded on the result for provenance.
    load_w_mpa : float
        Magnitude of the downward distributed traction on the top of the
        horizontal flange. Units MPa (= N/mm^2). Applied as -y traction; its
        resultant force per unit depth equals load_w_mpa * (B - (W+R)).
    h_fine, h_coarse : float
        Recorded on the result for the convergence-study sweep.
    return_fields : bool
        If True, attach the nodal von Mises field + coordinates (useful for
        plotting & GNN training later). If False, only the summary stats are
        returned (cheap).

    Notes on the modeling choices
    -----------------------------
    - Plane stress: justified by t/A = 1/10; out-of-plane stress << in-plane.
    - Linear elastic isotropic material, AISI 304: E = 193 GPa, nu = 0.29.
    - Quadratic triangular elements for the displacement (Lagrange order 2).
      Stresses recovered by differentiation are piecewise linear within each
      element and projected to a nodal field for plotting and peak-finding.
    - Self-weight omitted (see constants.py).
    - BC: homogeneous Dirichlet (u = 0) on the back face of the vertical
      flange (physical tag 10). Everything else is a free or loaded surface.
    - Load: constant traction -w*e_y on the top of the horizontal flange
      (physical tag 20). All other surfaces (holes, fillet, free tip, bottom
      of horizontal flange, top of vertical flange) are traction-free.
    """
    # Imports kept local so the rest of the package stays import-safe on
    # machines without dolfinx. On Kaggle these imports succeed.
    from mpi4py import MPI
    import dolfinx
    import ufl
    from dolfinx import fem, mesh as dmesh, io
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD

    # --- Read the mesh ----------------------------------------------------
    mesh_obj, cell_tags, facet_tags = io.gmshio.read_from_msh(
        str(msh_path), comm, gdim=2
    )

    # --- Function space: vector Lagrange order 2 on triangles -------------
    V = fem.functionspace(mesh_obj, ("Lagrange", 2, (mesh_obj.geometry.dim,)))

    # --- Plane-stress constitutive law -----------------------------------
    # sigma = lam_ps * tr(eps) * I + 2 * mu * eps
    # with the plane-stress-effective Lamé first parameter
    #     lam_ps = E*nu / (1 - nu^2)
    # and mu = E / (2*(1+nu)).
    E = E_MPA
    nu = NU
    mu = E / (2.0 * (1.0 + nu))
    lam_ps = E * nu / (1.0 - nu * nu)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        eps = epsilon(u)
        return lam_ps * ufl.tr(eps) * ufl.Identity(2) + 2.0 * mu * eps

    # --- Trial / test / BC -----------------------------------------------
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Clamped facets are those tagged 10 on the mesh boundary.
    clamped_facets = facet_tags.find(TAG_CLAMPED)
    clamped_dofs = fem.locate_dofs_topological(V, 1, clamped_facets)
    zero = np.zeros(mesh_obj.geometry.dim, dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(zero, clamped_dofs, V)

    # Measure restricted to the loaded facets.
    ds = ufl.Measure("ds", domain=mesh_obj, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh_obj)

    # Traction: -y direction, magnitude load_w_mpa (N/mm^2 = MPa).
    traction = fem.Constant(mesh_obj,
                            PETSc.ScalarType((0.0, -load_w_mpa)))

    # --- Weak form -------------------------------------------------------
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(traction, v) * ds(TAG_LOADED)

    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options={
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "pc_factor_mat_solver_type": "mumps",
                            })
    uh = problem.solve()

    # --- Recover von Mises stress at mesh nodes --------------------------
    # Project sigma onto a DG-1 tensor space then compute sqrt(3/2 * s:s)
    # where s = sigma - 1/3 tr(sigma) I (deviatoric). For plane stress we
    # must include the through-thickness component sigma_zz = 0 in the
    # deviator to recover the correct 3D von Mises from 2D fields:
    #     svm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
    sig = sigma(uh)
    s_xx = sig[0, 0]
    s_yy = sig[1, 1]
    s_xy = sig[0, 1]
    vm_expr = ufl.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)

    # Project onto a Lagrange order-2 scalar space so nodal values coincide
    # with the displacement mesh (keeps field sizes compatible downstream).
    Vs = fem.functionspace(mesh_obj, ("Lagrange", 2))
    vm = fem.Function(Vs)
    vm_expr_fn = fem.Expression(vm_expr, Vs.element.interpolation_points())
    vm.interpolate(vm_expr_fn)

    vm_array = vm.x.array
    coords = Vs.tabulate_dof_coordinates()[:, :2]

    peak_idx = int(np.argmax(vm_array))
    peak_vm = float(vm_array[peak_idx])
    peak_xy = (float(coords[peak_idx, 0]), float(coords[peak_idx, 1]))

    result = FEAResult(
        params=params,
        load_w_mpa=load_w_mpa,
        peak_vm_mpa=peak_vm,
        peak_location_xy=peak_xy,
        n_dofs=vm_array.size,
        h_fine=h_fine,
        h_coarse=h_coarse,
        vm_field=(vm_array.copy() if return_fields else None),
        coords=(coords.copy() if return_fields else None),
    )
    return result
