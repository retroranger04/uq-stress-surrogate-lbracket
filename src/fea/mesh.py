"""
Gmsh wrapper: takes an LBracketParams and writes a .msh for FEniCSx.

Runs via the `gmsh` Python API when available; falls back to the `gmsh` CLI
otherwise. Both paths consume the .geo file produced by geometry.build_geo.

This module is designed to run on Kaggle CPU (where FEniCSx + gmsh are both
installed). It is import-safe locally — the gmsh import is lazy.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .geometry import LBracketParams, build_geo


def write_msh(params: LBracketParams,
              out_path,
              h_coarse: float = 4.0,
              h_fine: float = 0.5,
              refine_dist: float = 6.0,
              keep_geo: bool = False) -> Path:
    """Generate a .msh file for the given parameters.

    Returns the path to the written .msh. If `keep_geo` is True, also retains
    the intermediate .geo next to the .msh (useful when debugging the mesher).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    geo_content = build_geo(params,
                            h_coarse=h_coarse,
                            h_fine=h_fine,
                            refine_dist=refine_dist)

    # Write the .geo near the .msh so relative paths work for either backend.
    geo_path = out_path.with_suffix(".geo")
    geo_path.write_text(geo_content, encoding="utf-8")

    # Prefer the Python API — it reports errors via exceptions which surface
    # cleaner tracebacks through Kaggle logs than the CLI's stderr does.
    try:
        import gmsh  # noqa: WPS433 — lazy import, optional locally
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.merge(str(geo_path))
            gmsh.model.mesh.generate(2)
            # Force order 2 in case the .geo directive is overridden by the API.
            gmsh.model.mesh.setOrder(2)
            gmsh.write(str(out_path))
        finally:
            gmsh.finalize()
    except ImportError:
        # CLI fallback. `gmsh -2 <geo> -o <msh> -order 2` builds a 2D mesh
        # with Lagrange order 2 elements.
        gmsh_bin = shutil.which("gmsh")
        if gmsh_bin is None:
            raise RuntimeError(
                "gmsh is not available: neither the Python `gmsh` package nor "
                "the `gmsh` CLI binary could be found on PATH."
            )
        cmd = [gmsh_bin, "-2", str(geo_path),
               "-o", str(out_path), "-order", "2", "-v", "2"]
        subprocess.run(cmd, check=True)

    if not keep_geo:
        geo_path.unlink(missing_ok=True)

    return out_path
