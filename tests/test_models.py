"""
Unit tests for the Phase-1 GNN skeleton.

Uses tiny synthetic `.npz` shards matching the Day-3 sweep schema so that
dataset.py and train.py exercise the real code path before any FEA data is
available. Runs on CPU to remain laptop-friendly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.gnn import MeshGNN, MeshGNNConfig, build_default_model
from src.models.dataset import (
    NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
    LBracketStressDataset, npz_to_data,
    _edge_index_from_tris, make_node_features, make_edge_features,
    lhs_stratified_split, compute_stats,
)
from src.models.train import TrainConfig, train


# --- Helpers ----------------------------------------------------------------

def _synthetic_npz(path: Path, n_nodes: int = 40, seed: int = 0) -> None:
    """Write a synthetic .npz matching the Day-3 sweep schema."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 80, size=(n_nodes, 2)).astype(np.float32)
    # Triangulate via a simple Delaunay over the random points.
    from scipy.spatial import Delaunay
    tri = Delaunay(pts)
    elem_t3 = tri.simplices.astype(np.int32)

    n_l2 = n_nodes + elem_t3.shape[0]  # Lagrange-2 adds edge-midpoint-ish count
    coords_l2 = np.vstack([pts, rng.uniform(0, 80, (n_l2 - n_nodes, 2))]).astype(np.float32)
    vm_l2 = rng.uniform(5, 100, size=n_l2).astype(np.float32)
    vm_t3 = rng.uniform(5, 100, size=n_nodes).astype(np.float32)
    params = np.array([rng.uniform(3, 10), rng.uniform(42, 72),
                       rng.uniform(14, 24)], dtype=np.float32)

    # Pretend a handful of nodes live on each boundary tag.
    def sample(k):
        return rng.choice(n_nodes, size=min(k, n_nodes), replace=False).astype(np.int32)

    np.savez_compressed(
        path,
        params=params,
        peak_vm=np.float32(vm_t3.max()),
        peak_xy=pts[int(np.argmax(vm_t3))],
        load_w_mpa=np.float32(0.8555),
        coords_l2=coords_l2, vm_l2=vm_l2,
        coords_t3=pts, vm_t3=vm_t3, elem_t3=elem_t3,
        dof_clamped_l2=sample(5), dof_loaded_l2=sample(5),
        dof_fillet_l2=sample(3), dof_hole1_l2=sample(4), dof_hole2_l2=sample(4),
        dof_clamped_t3=sample(4), dof_loaded_t3=sample(4),
        dof_fillet_t3=sample(3), dof_hole1_t3=sample(3), dof_hole2_t3=sample(3),
    )


# --- GNN architecture tests -------------------------------------------------

def test_gnn_forward_shapes():
    cfg = MeshGNNConfig(in_node_dim=NUM_NODE_FEATURES,
                        in_edge_dim=NUM_EDGE_FEATURES,
                        hidden=32, num_layers=2, out_dim=1)
    model = MeshGNN(cfg)
    N, E = 20, 50
    x = torch.randn(N, NUM_NODE_FEATURES)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, NUM_EDGE_FEATURES)
    y = model(x, ei, ea)
    assert y.shape == (N, 1)


def test_gnn_grad_flow():
    model = build_default_model(NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
                                hidden=16, num_layers=2)
    N, E = 10, 24
    x = torch.randn(N, NUM_NODE_FEATURES)
    ei = torch.randint(0, N, (2, E))
    ea = torch.randn(E, NUM_EDGE_FEATURES)
    y_hat = model(x, ei, ea)
    y = torch.randn_like(y_hat)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    loss.backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all().item()
             for p in model.parameters()]
    assert all(grads)


# --- Dataset feature-pack tests --------------------------------------------

def test_edge_index_from_tris_dedupe():
    tris = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    ei = _edge_index_from_tris(tris)
    # 2 triangles, 3 shared edges unique \u2014 each counted in both directions.
    # (0-1, 1-2, 0-2, 1-3, 2-3) \u2192 5 undirected \u2192 10 directed.
    assert ei.shape[0] == 2
    assert ei.shape[1] == 10


def test_make_node_features_shape():
    coords = np.random.uniform(0, 80, (15, 2)).astype(np.float32)
    bsets = {"clamped": np.array([0, 1], dtype=np.int32),
             "loaded":  np.array([2, 3], dtype=np.int32),
             "fillet":  np.array([4], dtype=np.int32),
             "hole1":   np.array([5, 6], dtype=np.int32),
             "hole2":   np.array([7, 8], dtype=np.int32)}
    params = np.array([6.0, 50.0, 20.0], dtype=np.float32)
    X = make_node_features(coords, bsets, params)
    assert X.shape == (15, NUM_NODE_FEATURES)
    # is_free one-hot \u2014 nodes 9..14 have no boundary tag.
    base = 2 + 5
    assert (X[9:, base] == 1.0).all()


def test_make_edge_features_basic():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    ei = np.array([[0, 1], [1, 0]], dtype=np.int64)
    E = make_edge_features(coords, ei)
    assert E.shape == (2, 4)
    assert np.allclose(E[0, 2], 5.0)      # ||(3,4)|| = 5
    assert np.allclose(E[0, 3], 1 / 5.0)


def test_dataset_roundtrip(tmp_path):
    samples = tmp_path / "samples"; samples.mkdir()
    for i in range(5):
        _synthetic_npz(samples / f"s{i:05d}.npz", n_nodes=25, seed=i)
    ds = LBracketStressDataset(tmp_path)
    assert len(ds.paths) == 5
    data = ds.get(0)
    assert data.x.shape[1] == NUM_NODE_FEATURES
    assert data.edge_attr.shape[1] == NUM_EDGE_FEATURES
    assert data.y.shape == (25, 1)
    assert data.params.shape == (1, 3)


def test_lhs_stratified_split_covers_disjoint():
    params = np.random.default_rng(0).uniform(0, 1, size=(100, 3))
    tr, va, te = lhs_stratified_split(params, 0.8, 0.1, seed=0)
    assert len(set(tr) | set(va) | set(te)) == 100
    assert set(tr).isdisjoint(va)
    assert set(tr).isdisjoint(te)
    assert set(va).isdisjoint(te)
    # Sizes within rounding.
    assert 78 <= len(tr) <= 82
    assert 8 <= len(va) <= 12
    assert 8 <= len(te) <= 12


# --- End-to-end training-step smoke test -----------------------------------

def test_train_one_step_smoke(tmp_path):
    samples = tmp_path / "samples"; samples.mkdir()
    for i in range(12):
        _synthetic_npz(samples / f"s{i:05d}.npz", n_nodes=30, seed=i)
    cfg = TrainConfig(
        data_root=str(tmp_path), out_dir=str(tmp_path / "run"),
        hidden=16, num_layers=2, lr=1e-3, epochs=2,
        batch_size=2, patience=5, seed=0, device="cpu",
    )
    result = train(cfg)
    # Loss should be finite and history recorded.
    assert np.isfinite(result["best_val"])
    assert len(result["history"]) >= 1


def test_batched_forward_matches_per_graph_shapes(tmp_path):
    samples = tmp_path / "samples"; samples.mkdir()
    for i in range(3):
        _synthetic_npz(samples / f"s{i:05d}.npz", n_nodes=20, seed=i)
    ds = LBracketStressDataset(tmp_path)
    stats = compute_stats(ds)
    ds._stats = stats
    loader = DataLoader(ds, batch_size=3)
    batch = next(iter(loader))
    model = build_default_model(NUM_NODE_FEATURES, NUM_EDGE_FEATURES,
                                hidden=16, num_layers=2)
    y_hat = model(batch.x, batch.edge_index, batch.edge_attr)
    assert y_hat.shape == batch.y.shape
