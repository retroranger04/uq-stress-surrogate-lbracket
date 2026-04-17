"""Rerun the four-level mesh-convergence study locally in WSL2 FEniCSx.

Mirrors the Day-2 study exactly (same (h_coarse, h_fine, refine_dist) per
level, same nominal sample R=6.5 / p=57 / W=19, same unit load 1 MPa).
Produces a JSON result bundle consumed by the paper's Table and the
convergence.png figure.

Run inside the WSL2 fenicsx conda env:
    conda activate fenicsx
    python scripts/run_convergence_local.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.fea.constants import E_MPA, NU
from src.fea.geometry import LBracketParams
from src.fea.mesh import write_msh

from mpi4py import MPI
import ufl
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

TAG_CLAMPED = 10
TAG_LOADED = 20

LEVELS = [
    dict(level=0, h_coarse=4.0, h_fine=1.5, refine_dist=6.0),
    dict(level=1, h_coarse=3.0, h_fine=0.8, refine_dist=6.0),
    dict(level=2, h_coarse=2.5, h_fine=0.4, refine_dist=7.0),
    dict(level=3, h_coarse=2.0, h_fine=0.2, refine_dist=8.0),
]

PARAMS = LBracketParams(R=6.5, p=57.0, W=19.0)
W_MPA = 1.0


def solve_one(L, tmp: Path):
    msh = tmp / f"L{L['level']}.msh"
    write_msh(PARAMS, msh,
              h_coarse=L["h_coarse"],
              h_fine=L["h_fine"],
              refine_dist=L["refine_dist"])
    comm = MPI.COMM_WORLD
    mesh_obj, _, facet_tags = io.gmshio.read_from_msh(str(msh), comm, gdim=2)

    V = fem.functionspace(mesh_obj, ("Lagrange", 2, (mesh_obj.geometry.dim,)))
    Vs = fem.functionspace(mesh_obj, ("Lagrange", 2))
    mu = E_MPA / (2.0 * (1.0 + NU))
    lam = E_MPA * NU / (1.0 - NU * NU)

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sig(u):
        return lam * ufl.tr(eps(u)) * ufl.Identity(2) + 2.0 * mu * eps(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    clamped = fem.locate_dofs_topological(V, 1, facet_tags.find(TAG_CLAMPED))
    bc = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), clamped, V)
    ds = ufl.Measure("ds", domain=mesh_obj, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh_obj)
    trac = fem.Constant(mesh_obj, PETSc.ScalarType((0.0, -W_MPA)))

    a = ufl.inner(sig(u), eps(v)) * dx
    Lf = ufl.inner(trac, v) * ds(TAG_LOADED)
    prob = LinearProblem(
        a, Lf, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = prob.solve()

    sg = sig(uh)
    vm_expr = ufl.sqrt(
        sg[0, 0] ** 2 - sg[0, 0] * sg[1, 1] + sg[1, 1] ** 2 + 3.0 * sg[0, 1] ** 2
    )
    vm = fem.Function(Vs)
    vm.interpolate(fem.Expression(vm_expr, Vs.element.interpolation_points()))
    peak = float(np.max(vm.x.array))
    n_dofs = int(vm.x.array.size)
    return n_dofs, peak


def main():
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="conv_local_"))
    for L in LEVELS:
        t0 = time.monotonic()
        n_dofs, peak = solve_one(L, tmp)
        dt = time.monotonic() - t0
        results.append(dict(**L, n_dofs=n_dofs, peak_vm=peak, solve_s=dt))
    for i, r in enumerate(results):
        prev = results[i - 1]["peak_vm"] if i > 0 else None
        d = (r["peak_vm"] - prev) / prev * 100.0 if prev is not None else None
        dstr = "--" if d is None else f"{d:+.3f}%"
        print(f"L{r['level']}  hc={r['h_coarse']:.1f}  hf={r['h_fine']:.1f}  "
              f"dofs={r['n_dofs']:>6}  peak={r['peak_vm']:.3f}  delta={dstr}  t={r['solve_s']:.2f}s")

    out_json = PROJECT / "data" / "day2_validation" / "convergence_local.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
