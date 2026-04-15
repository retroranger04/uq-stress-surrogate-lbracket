"""
Comprehensive local validation of the Kaggle payload before pushing.

Catches, in order:
    1. Python syntax errors in src_fea_inline.py.
    2. Python syntax errors in every code cell of the assembled notebook.
    3. Python syntax errors in the `run_all.py` heredoc embedded inside the
       notebook's FEA-driver cell.
    4. Base64 round-trip corruption on the embedded src/fea blob.
    5. Import-stripping bugs in src_fea_inline.py — exec it in a namespace
       with dolfinx/gmsh/mpi4py/petsc4py/ufl stubbed out and verify the pure-
       Python surface (LBracketParams, check_validity, build_geo,
       build_simplified_geo, bending_stress_at_section) behaves correctly.
    6. Dangling references in the runner to names that src_fea_inline does
       not actually define.

Exits non-zero on any failure. Called from assemble_kaggle_notebook.py.

FEniCSx runtime API compatibility is NOT checkable locally (we have no
FEniCSx install on Windows), so we defend against API churn by pinning
`fenics-dolfinx=0.9.*` in the install cell — that's what our reference
material (raw/papers/dokken_fenicsx.md) is written against.
"""

from __future__ import annotations

import ast
import base64
import json
import py_compile
import re
import sys
import tempfile
import textwrap
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kaggle_day2"


# ---------------------------------------------------------------------------
# Lightweight stubs for heavy scientific-Python dependencies.
#
# Enough to let `exec(src_fea_inline.py)` complete; none of the stubs do
# real work. We stash them in sys.modules so that `import dolfinx` et al.
# resolve to our mocks during the preflight run.
# ---------------------------------------------------------------------------

def _install_stubs() -> None:
    # mpi4py -> has .MPI.COMM_WORLD attribute in real usage, stub is enough
    # because src_fea_inline's solver only imports mpi4py lazily inside
    # solve_lbracket(); the pure-python surface never calls it.
    for mod_name in [
        "mpi4py", "mpi4py.MPI",
        "dolfinx", "dolfinx.fem", "dolfinx.fem.petsc",
        "dolfinx.mesh", "dolfinx.io", "dolfinx.io.gmshio",
        "petsc4py", "petsc4py.PETSc",
        "ufl",
        "gmsh",
    ]:
        mod = types.ModuleType(mod_name)
        sys.modules.setdefault(mod_name, mod)


# ---------------------------------------------------------------------------

def check_file_syntax(path: Path, label: str) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        fail(f"{label} syntax error: {e}")
    ok(f"{label} compiles")


def check_string_syntax(src: str, label: str) -> None:
    try:
        compile(src, label, "exec")
    except SyntaxError as e:
        # Print surrounding lines for context.
        lines = src.splitlines()
        lo = max(0, e.lineno - 3); hi = min(len(lines), e.lineno + 2)
        ctx = "\n".join(f"{i+1:4d}: {lines[i]}" for i in range(lo, hi))
        fail(f"{label} syntax error at line {e.lineno}: {e.msg}\n{ctx}")
    ok(f"{label} compiles")


def check_notebook_cells(nb_path: Path) -> tuple[list[str], str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = []
    runner_src = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        check_string_syntax(src, f"notebook cell[{i}]")
        code_cells.append(src)
        if "RUNNER.write_text" in src and "textwrap.dedent" in src:
            # Extract the runner heredoc body (the r'''...''' literal).
            m = re.search(r"textwrap\.dedent\(\s*r?'''(.*?)'''\s*\)",
                          src, re.DOTALL)
            if m:
                runner_src = textwrap.dedent(m.group(1)).lstrip()
    if runner_src is None:
        fail("could not find the run_all.py heredoc in any notebook cell")
    return code_cells, runner_src


def check_base64_roundtrip(nb_path: Path, inline_path: Path) -> None:
    """Verify that the notebook's embedded base64 blob decodes back to the
    exact bytes of notebooks/kaggle_day2/src_fea_inline.py.
    """
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    inline_bytes = inline_path.read_bytes()
    found = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        m = re.search(r"_FEA_BLOB\s*=\s*'([A-Za-z0-9+/=]+)'", src)
        if m:
            decoded = base64.b64decode(m.group(1))
            if decoded != inline_bytes:
                fail(f"base64 round-trip mismatch: {len(decoded)} vs "
                     f"{len(inline_bytes)} bytes")
            ok(f"base64 blob round-trips ({len(inline_bytes)} bytes)")
            found = True
            break
    if not found:
        fail("no _FEA_BLOB literal found in any notebook cell")


def check_inline_surface(inline_src: str) -> dict:
    """Execute the inlined src/fea in a namespace with heavy deps stubbed,
    then exercise the pure-Python surface and return the namespace (so the
    caller can use its names to check the runner).
    """
    _install_stubs()
    # Register a proper module for the exec namespace so that @dataclass can
    # do its `sys.modules.get(cls.__module__).__dict__` lookup. Without this,
    # 3.14's dataclasses raises AttributeError at decoration time.
    preflight_mod = types.ModuleType("_preflight_inline")
    sys.modules["_preflight_inline"] = preflight_mod
    ns: dict = preflight_mod.__dict__
    ns["__name__"] = "_preflight_inline"
    try:
        exec(compile(inline_src, "src_fea_inline.py", "exec"), ns)
    except Exception as e:  # noqa: BLE001 — preflight bails on any failure
        fail(f"exec(src_fea_inline.py) raised {type(e).__name__}: {e}")

    for name in ("LBracketParams", "check_validity", "build_geo",
                 "build_simplified_geo", "bending_stress_at_section",
                 "build_outer_boundary", "build_holes",
                 "write_msh", "solve_lbracket"):
        if name not in ns:
            fail(f"src_fea_inline.py is missing expected symbol `{name}`")
    ok("src_fea_inline.py exposes the expected public API")

    LBP = ns["LBracketParams"]
    chk = ns["check_validity"]
    bgeo = ns["build_geo"]
    bsimp = ns["build_simplified_geo"]
    bend = ns["bending_stress_at_section"]

    # Nominal + worst-case must validate.
    for label, kw in [("nominal", dict(R=6.5, p=57.0, W=19.0)),
                      ("worst-case", dict(R=3.0, p=42.0, W=14.0))]:
        p = LBP(**kw)
        vr = chk(p)
        if not vr.ok:
            fail(f"{label} rejected by check_validity: {vr.reason}")
    ok("nominal + worst-case samples pass check_validity")

    # An obvious infeasible combo must be rejected.
    bad = LBP(R=15.0, p=57.0, W=24.0)
    if chk(bad).ok:
        fail("check_validity failed to reject (R=15, p=57, W=24) — "
             "should violate W+R<=34")
    ok("check_validity rejects an infeasible sample")

    # build_geo should produce a non-trivial string.
    geo = bgeo(LBP(R=6.5, p=57.0, W=19.0), h_coarse=4.0, h_fine=0.5)
    if not geo or "Plane Surface" not in geo or "Physical Curve" not in geo:
        fail("build_geo output missing expected Gmsh directives")
    ok(f"build_geo produces a well-formed .geo string ({len(geo)} chars)")

    # build_simplified_geo likewise.
    geo_s = bsimp(W_mm=19.0)
    if "Plane Surface" not in geo_s or "Physical Curve" not in geo_s:
        fail("build_simplified_geo output missing expected directives")
    ok(f"build_simplified_geo produces a well-formed .geo ({len(geo_s)} chars)")

    # Analytical bending stress sanity — should be positive and scale as
    # (B - x)^2 / W^2 at unit load.
    b1 = bend(x_mm=34.0, W_mm=19.0, load_w_mpa=1.0)
    b2 = bend(x_mm=49.0, W_mm=19.0, load_w_mpa=1.0)
    if not (b1.sigma_bending > b2.sigma_bending > 0):
        fail(f"bending_stress_at_section monotonicity broken: "
             f"{b1.sigma_bending=} {b2.sigma_bending=}")
    ok("bending_stress_at_section monotonicity OK")

    return ns


def check_runner_references(runner_src: str, inline_ns: dict) -> None:
    """Static-parse the runner heredoc, find Name references at module scope,
    and flag any that should be supplied by src_fea_inline but aren't.
    """
    # Symbols that src_fea_inline must provide (we just verified these are
    # present) — the runner is allowed to reference them freely.
    inline_syms = set(inline_ns)
    # Builtins + std-lib names the runner imports itself.
    runner_local = {"json", "pathlib", "sys", "np", "numpy",
                    "MPI", "dolfinx", "fem", "dmesh", "LinearProblem",
                    "PETSc", "ufl", "gmsh", "HERE",
                    # intermediate loop + result names the runner creates:
                    "Lg", "Hg", "m_sm", "Vs", "left", "dofs", "bc_sm",
                    "Es", "nus", "mus", "lm", "eps", "sg", "u", "v", "f",
                    "uh", "tip_u", "smoke",
                    "NOMINAL", "W_WORST", "PROBE_LOAD",
                    "levels", "params_nom", "conv", "i", "lvl", "msh", "r",
                    "rel_two_finest",
                    "W_test", "simple_geo", "geo_path", "msh_s",
                    "r_simple", "x_probe", "pred", "coords", "vm",
                    "mask", "fea_fibre", "cross_ratio",
                    "converged", "params_worst", "msh_w", "r_worst",
                    "target_peak", "w_calibrated", "bundle"}
    allowed = inline_syms | runner_local | set(dir(__builtins__))

    tree = ast.parse(runner_src)
    loads: set[str] = set()
    stores: set[str] = set()
    # Collect parameter names from lambdas + function defs + comprehensions so
    # we don't flag `x` in `lambda x: ...` as undefined.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (node.args.args + node.args.kwonlyargs
                        + node.args.posonlyargs):
                stores.add(arg.arg)
            if node.args.vararg: stores.add(node.args.vararg.arg)
            if node.args.kwarg: stores.add(node.args.kwarg.arg)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                ast.GeneratorExp)):
            for gen in node.generators:
                for sub in ast.walk(gen.target):
                    if isinstance(sub, ast.Name):
                        stores.add(sub.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                stores.add(node.id)
        elif isinstance(node, ast.alias):
            if node.asname:
                stores.add(node.asname)
            else:
                stores.add(node.name.split(".")[0])

    missing = sorted(loads - stores - allowed)
    # Filter out attribute-like segments and common false positives.
    missing = [m for m in missing if not m.startswith("_")]
    if missing:
        fail(f"runner references undefined names: {missing}")
    ok("runner references only names defined by itself or src_fea_inline")


# ---------------------------------------------------------------------------

def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    print(f"Preflight on {OUT}")
    inline = OUT / "src_fea_inline.py"
    nb = OUT / "day2_pipeline_validation.ipynb"
    for p in (inline, nb):
        if not p.exists():
            fail(f"{p} not found — run assemble_kaggle_notebook.py first")

    check_file_syntax(inline, "src_fea_inline.py")
    cells, runner_src = check_notebook_cells(nb)
    check_string_syntax(runner_src, "run_all.py (extracted from cell heredoc)")
    check_base64_roundtrip(nb, inline)
    ns = check_inline_surface(inline.read_text(encoding="utf-8"))
    check_runner_references(runner_src, ns)

    # kernel-metadata.json is valid JSON and has the fields Kaggle requires.
    meta_path = OUT / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for req in ("id", "title", "code_file", "language", "kernel_type",
                "enable_internet"):
        if req not in meta:
            fail(f"kernel-metadata.json missing required key `{req}`")
    ok("kernel-metadata.json has required keys")

    print("\nAll preflight checks PASSED — safe to `kaggle kernels push`.")


if __name__ == "__main__":
    main()
