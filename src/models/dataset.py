"""
PyTorch Geometric Dataset + feature pack for the L-bracket stress surrogate.

Consumes per-sample `.npz` files produced by the Day-3 Kaggle sweeps
(keys documented in `notebooks/day3_main_sweep.ipynb`). One `.npz` per
bracket sample maps to one PyG `Data` object with:

- `pos`         : (N, 2) T3 corner-node coordinates [mm]
- `x`           : (N, F) node feature pack (see `make_node_features`)
- `edge_index`  : (2, E) directed edge list, both (i, j) and (j, i) present
- `edge_attr`   : (E, 4) edge features (dx, dy, ||d||, 1 / ||d||)
- `y`           : (N, 1) per-node von Mises stress [MPa]
- `params`      : (1, 3) graph-level (R, p, W)
- `peak_vm`     : (1,)   graph-level peak von Mises [MPa]
- `direction`   : str    OOD direction label, '' for in-distribution

Phase 1 trains the surrogate on the T3 (linear) view; the richer L2 view is
preserved in the .npz for possible Phase-2/3 experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


# --- Feature assembly --------------------------------------------------------

# Boundary categories match the solver's physical-group tagging.
BOUNDARY_CATEGORIES = ("clamped", "loaded", "fillet", "hole1", "hole2")
NUM_NODE_FEATURES = 2 + len(BOUNDARY_CATEGORIES) + 1 + 3 + 2
#                   |     |                       |   |   |
#                   pos   boundary one-hot (5)    free  params  distances
NUM_EDGE_FEATURES = 4


def _edge_index_from_tris(elem_t3: np.ndarray) -> np.ndarray:
    """Build a (2, E) directed edge index from a (M, 3) triangle list.

    Each triangle contributes 6 directed edges; duplicates are removed.
    """
    a, b, c = elem_t3[:, 0], elem_t3[:, 1], elem_t3[:, 2]
    src = np.concatenate([a, b, b, c, a, c])
    dst = np.concatenate([b, a, c, b, c, a])
    edges = np.stack([src, dst], axis=0).astype(np.int64)
    # dedupe via a structured view
    view = edges.T.copy().view(dtype=[('s', np.int64), ('d', np.int64)]).reshape(-1)
    uniq = np.unique(view)
    out = np.empty((2, uniq.size), dtype=np.int64)
    out[0] = uniq['s']
    out[1] = uniq['d']
    return out


def make_node_features(coords: np.ndarray,
                       boundary_sets: dict[str, np.ndarray],
                       params: np.ndarray) -> np.ndarray:
    """Assemble per-node feature matrix.

    Features: [x, y, one_hot(boundary, 5)+is_free, R, p, W, d_fillet, d_hole].
    """
    N = coords.shape[0]
    feats = np.zeros((N, NUM_NODE_FEATURES), dtype=np.float32)

    feats[:, 0:2] = coords.astype(np.float32)

    # Boundary one-hots (5) + is_free.
    is_any = np.zeros(N, dtype=bool)
    for i, cat in enumerate(BOUNDARY_CATEGORIES):
        idx = boundary_sets.get(cat, np.empty(0, dtype=np.int64))
        if idx.size:
            feats[idx, 2 + i] = 1.0
            is_any[idx] = True
    feats[~is_any, 2 + len(BOUNDARY_CATEGORIES)] = 1.0   # is_free

    # Params broadcast (R, p, W) at indices 2 + len + 1 .. + 4
    base = 2 + len(BOUNDARY_CATEGORIES) + 1
    feats[:, base:base + 3] = params.astype(np.float32)

    # Distance to fillet / nearest hole (Euclidean, O(N*k) with small k).
    fillet_nodes = boundary_sets.get("fillet", np.empty(0, dtype=np.int64))
    hole_nodes = np.concatenate([
        boundary_sets.get("hole1", np.empty(0, dtype=np.int64)),
        boundary_sets.get("hole2", np.empty(0, dtype=np.int64)),
    ])
    if fillet_nodes.size:
        d = np.linalg.norm(
            coords[:, None, :] - coords[fillet_nodes][None, :, :], axis=-1)
        feats[:, base + 3] = d.min(axis=1).astype(np.float32)
    if hole_nodes.size:
        d = np.linalg.norm(
            coords[:, None, :] - coords[hole_nodes][None, :, :], axis=-1)
        feats[:, base + 4] = d.min(axis=1).astype(np.float32)

    return feats


def make_edge_features(pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Edge features: (dx, dy, ||d||, 1/||d||). Inverse distance is clipped."""
    src, dst = edge_index[0], edge_index[1]
    diff = (pos[dst] - pos[src]).astype(np.float32)
    dist = np.linalg.norm(diff, axis=1).astype(np.float32)
    inv = 1.0 / np.clip(dist, 1e-6, None)
    return np.stack([diff[:, 0], diff[:, 1], dist, inv], axis=1).astype(np.float32)


# --- Dataset ----------------------------------------------------------------

@dataclass
class FeatureStats:
    """Per-feature mean/std used for input normalization."""
    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def apply(self, data: Data) -> Data:
        data.x = (data.x - self.node_mean) / self.node_std
        data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
        # y is kept in physical units; loss operates on the raw MPa scale so
        # MAPE/peak-MAPE metrics downstream are interpretable directly.
        return data


def npz_to_data(path: Path) -> Data:
    """Convert one Day-3 sweep .npz to a PyG Data object (T3 view)."""
    blob = np.load(path, allow_pickle=False)
    coords = blob["coords_t3"]
    vm = blob["vm_t3"]
    elem = blob["elem_t3"]
    params = blob["params"]

    boundary_sets = {
        "clamped": blob["dof_clamped_t3"],
        "loaded":  blob["dof_loaded_t3"],
        "fillet":  blob["dof_fillet_t3"],
        "hole1":   blob["dof_hole1_t3"],
        "hole2":   blob["dof_hole2_t3"],
    }

    edge_index = _edge_index_from_tris(elem)
    node_feats = make_node_features(coords, boundary_sets, params)
    edge_feats = make_edge_features(coords, edge_index)

    data = Data(
        x=torch.from_numpy(node_feats),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_feats),
        pos=torch.from_numpy(coords.astype(np.float32)),
        y=torch.from_numpy(vm.astype(np.float32)).unsqueeze(-1),
        params=torch.from_numpy(params.astype(np.float32)).unsqueeze(0),
        peak_vm=torch.tensor([float(blob["peak_vm"])], dtype=torch.float32),
    )
    if "direction" in blob.files:
        data.direction = str(blob["direction"])
    else:
        data.direction = ""
    return data


class LBracketStressDataset(Dataset):
    """PyG Dataset over a directory of `.npz` sample shards.

    Expects `root/samples/*.npz` (or `root/*.npz` if `flat=True`). `split`
    selects which subset of files to expose: values are produced by
    `split_indices` and are arbitrary here (the Dataset doesn't reshuffle
    internally so the caller controls stratification).
    """

    def __init__(self, root: str | Path, *,
                 sample_paths: list[Path] | None = None,
                 stats: FeatureStats | None = None,
                 flat: bool = False):
        super().__init__(root=str(root))
        root = Path(root)
        if sample_paths is None:
            glob = (root / "*.npz") if flat else (root / "samples" / "*.npz")
            sample_paths = sorted(Path(root).glob(str(glob.relative_to(root))))
        self._paths = list(sample_paths)
        self._stats = stats

    def len(self) -> int:
        return len(self._paths)

    def get(self, idx: int) -> Data:
        data = npz_to_data(self._paths[idx])
        if self._stats is not None:
            data = self._stats.apply(data)
        return data

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)


# --- Splitting --------------------------------------------------------------

def lhs_stratified_split(param_rows: np.ndarray,
                          train_frac: float = 0.8,
                          val_frac: float = 0.1,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort samples by a Hilbert-like ordering of the (R, p, W) cube, then
    deal round-robin into train/val/test. This preserves LHS coverage inside
    each split \u2014 no split ends up clustered in a sub-region of the box.

    Inputs
    ------
    param_rows : (N, 3) array of (R, p, W) per sample.
    """
    N = param_rows.shape[0]
    # Normalize each column to [0,1] and encode as an interleaved bit string.
    col_mins = param_rows.min(axis=0)
    col_maxs = param_rows.max(axis=0)
    norm = (param_rows - col_mins) / np.maximum(col_maxs - col_mins, 1e-12)
    # 10-bit per coordinate is enough to separate 1000 LHS samples.
    q = np.clip((norm * 1023).astype(np.uint32), 0, 1023)
    keys = np.zeros(N, dtype=np.uint64)
    for b in range(10):
        for c in range(3):
            keys |= ((q[:, c] >> b) & 1).astype(np.uint64) << (3 * b + c)
    order = np.argsort(keys)

    # Deal round-robin into splits with the target fractions.
    rng = np.random.default_rng(seed)
    # Shuffle within small chunks to break ties from equal-key neighbours
    # without disturbing the global ordering.
    chunk = 10
    for i in range(0, N, chunk):
        rng.shuffle(order[i:i + chunk])

    n_train = int(round(N * train_frac))
    n_val = int(round(N * val_frac))
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]
    return (np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))


# --- Stats computation ------------------------------------------------------

def compute_stats(dataset: LBracketStressDataset,
                  max_samples: int = 200) -> FeatureStats:
    """Estimate per-feature mean/std from a subset of samples."""
    xs, es, ys = [], [], []
    n = min(len(dataset._paths), max_samples)
    for i in range(n):
        d = dataset.get(i)
        xs.append(d.x.numpy())
        es.append(d.edge_attr.numpy())
        ys.append(d.y.numpy())
    X = np.concatenate(xs, axis=0)
    E = np.concatenate(es, axis=0)
    Y = np.concatenate(ys, axis=0)
    return FeatureStats(
        node_mean=torch.from_numpy(X.mean(0)),
        node_std=torch.from_numpy(np.maximum(X.std(0), 1e-4)),
        edge_mean=torch.from_numpy(E.mean(0)),
        edge_std=torch.from_numpy(np.maximum(E.std(0), 1e-4)),
        y_mean=torch.from_numpy(Y.mean(0)),
        y_std=torch.from_numpy(np.maximum(Y.std(0), 1e-4)),
    )
