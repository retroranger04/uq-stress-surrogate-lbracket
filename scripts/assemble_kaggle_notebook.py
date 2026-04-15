"""
Assemble the Kaggle notebook payload.

Reads src/fea/{constants,geometry,mesh,solver,analytical}.py, concatenates them
(stripping internal ``from .constants import ...`` lines that only make sense
as a package), and emits a single importable script:

    notebooks/kaggle_day2/src_fea_inline.py

Also copies notebooks/day2_pipeline_validation.ipynb and writes the Kaggle
kernel-metadata.json beside it so that `kaggle kernels push -p notebooks/kaggle_day2`
uploads a self-contained payload.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEA = ROOT / "src" / "fea"
OUT = ROOT / "notebooks" / "kaggle_day2"
OUT.mkdir(parents=True, exist_ok=True)

MODULES = ["constants", "geometry", "analytical", "mesh", "solver"]

HEADER = """\"\"\"
Inlined copy of src/fea/ for the self-contained Kaggle notebook.
GENERATED — do not edit by hand. Regenerate with
    python scripts/assemble_kaggle_notebook.py
\"\"\"

from __future__ import annotations
"""

# Regex for relative imports like `from .constants import X, Y` — we drop
# them because after concatenation everything lives in the same namespace.
# Handles both single-line and multi-line parenthesized forms:
#     from .constants import A, B
#     from .constants import (A, B, C,)   (may span multiple lines)
REL_IMPORT_RE = re.compile(
    r"^\s*from\s+\.[a-zA-Z_][a-zA-Z_0-9]*\s+import\s+"
    r"(?:\([^)]*\)|[^\n]+)\s*$",
    re.MULTILINE,
)
# `from __future__ import annotations` appears in each module; keep only the
# top-level one from the header and strip per-module duplicates.
FUTURE_IMPORT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.*$",
                              re.MULTILINE)


def strip_relative_imports(src: str) -> str:
    src = REL_IMPORT_RE.sub("", src)
    src = FUTURE_IMPORT_RE.sub("", src)
    return src


def main() -> None:
    chunks = [HEADER]
    for name in MODULES:
        path = FEA / f"{name}.py"
        src = path.read_text(encoding="utf-8")
        src = strip_relative_imports(src)
        chunks.append(f"\n# " + "=" * 70 + f"\n# BEGIN {name}.py\n# " + "=" * 70)
        chunks.append(src)
    body = "\n".join(chunks)
    # write_bytes (not write_text) so Windows doesn't translate \n -> \r\n;
    # that would desynchronise the disk file from the base64 blob embedded
    # into the notebook (Kaggle writes bytes back out and runs on Linux).
    (OUT / "src_fea_inline.py").write_bytes(body.encode("utf-8"))
    print(f"wrote {OUT/'src_fea_inline.py'} ({len(body)} bytes)")

    # Copy the notebook next to the payload, then patch its cell that writes
    # src_fea_inline.py to disk on Kaggle. Kaggle's `kernels push` only
    # uploads the notebook file itself — sibling files in the payload dir are
    # NOT uploaded. Embedding the inline content as a base64 literal inside
    # the notebook sidesteps that limitation entirely.
    import base64
    nb_src = ROOT / "notebooks" / "day2_pipeline_validation.ipynb"
    nb_dst = OUT / "day2_pipeline_validation.ipynb"
    shutil.copyfile(nb_src, nb_dst)

    blob = base64.b64encode(body.encode("utf-8")).decode("ascii")
    nb = json.loads(nb_dst.read_text(encoding="utf-8"))
    # Find the cell that references ./src_fea_inline.py and rewrite it to
    # decode the base64 blob instead of reading a neighbour file.
    patched = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src_join = "".join(cell["source"])
        if "src_fea_inline.py" in src_join and "shutil.copyfile" in src_join:
            new = (
                "import base64, pathlib, subprocess, textwrap\n\n"
                "OUT = pathlib.Path('/kaggle/working/day2')\n"
                "OUT.mkdir(parents=True, exist_ok=True)\n\n"
                "# Embedded src/fea payload (base64 of the assembled inline file).\n"
                f"_FEA_BLOB = '{blob}'\n"
                "(OUT / 'src_fea_inline.py').write_bytes(base64.b64decode(_FEA_BLOB))\n"
                "print('wrote', OUT / 'src_fea_inline.py')\n\n"
                + "\n".join(
                    line for line in src_join.splitlines()
                    if "pathlib, shutil" not in line
                    and "OUT = pathlib.Path" not in line
                    and "OUT.mkdir" not in line
                    and "shutil.copyfile" not in line
                    and not line.strip().startswith("if False else")
                )
            )
            cell["source"] = new.splitlines(keepends=True)
            patched = True
            break
    if not patched:
        raise RuntimeError("could not locate the src_fea_inline.py cell to patch")
    nb_dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"patched notebook with embedded blob -> {nb_dst} "
          f"(blob {len(blob)} chars)")

    # Kaggle kernel metadata.
    meta = {
        # Kaggle username is `retroranger` (distinct from GitHub `retroranger04`).
        # Kernel slug is auto-derived from the title; we match ids to minimize
        # push warnings.
        "id": "retroranger/uq-l-bracket-day-2-validation",
        "title": "UQ L-Bracket Day-2 Validation",
        "code_file": "day2_pipeline_validation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT/'kernel-metadata.json'}")

    # Preflight is MANDATORY — exits non-zero on any failure, which makes
    # subsequent `kaggle kernels push` in the same shell abort via &&.
    print("\nRunning preflight checks...")
    import subprocess as _sp
    r = _sp.run([sys.executable, str(Path(__file__).parent / "preflight_kaggle.py")])
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    import sys  # noqa: E402 — kept out of top-level to avoid shadowing `sys.exit` semantics
    main()
