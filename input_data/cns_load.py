"""
input_data/cns_load.py
======================
Custom data loader for the CNS removal-rate experiment.

Bug fix (v2):
    Negative sampling now uses HARD negatives — pairs that have BT/SMS/calls
    contact but are NOT fb_friends. This prevents AUC inflation caused by
    easy random negatives that have zero temporal/structural signal.
    If there are not enough hard negatives, the remainder is filled with
    random non-fb pairs.
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl import AddReverse, ToSimple

from input_data.load import (
    load_netf,
    build_supra_adj,
    build_identity_matrix,
    load_features,
)

TARGET_LAYER = 3   # fb_friends (0-based), layer_id=4 in net.edges


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_sym_dgl(rows, cols, n: int) -> dgl.DGLGraph:
    values = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((values, (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj = sp.triu(adj, k=1)

    g_up = dgl.from_scipy(adj, eweight_name="w")
    transform_rev  = AddReverse(copy_edata=True)
    transform_simp = ToSimple(aggregator="arbitrary", return_counts="w")
    g = transform_simp(transform_rev(g_up))
    g.edata.pop(dgl.NID, None)
    if g.num_nodes() < n:
        g = dgl.add_nodes(g, n - g.num_nodes())
    g.edata["w"] = g.edata["w"].float()
    return g


def _sample_hard_negatives(
    fb_adj:      sp.csr_matrix,
    context_adj: sp.csr_matrix,
    n_needed:    int,
    rng:         np.random.RandomState,
) -> np.ndarray:
    """
    Sample hard negatives: pairs that have context-layer contact (BT/SMS/calls)
    but are NOT fb_friends. Falls back to random non-fb pairs if not enough.

    fb_adj      : full fb_friends adjacency (to exclude all fb pairs)
    context_adj : union of BT + SMS + calls adjacency (hard candidate pool)
    n_needed    : number of negatives required
    rng         : reproducible random state

    Returns (n_needed, 2) int64 array.
    """
    n = fb_adj.shape[0]

    # Symmetrize both
    fb   = fb_adj + fb_adj.T
    fb.data = np.ones_like(fb.data)

    ctx  = context_adj + context_adj.T
    ctx.data = np.ones_like(ctx.data)

    # Hard pool: context contact AND NOT fb_friends AND NOT self-loop
    # Hard candidates: in context but not in fb
    hard_mat = ctx - ctx.multiply(fb)
    hard_mat.data = np.ones_like(hard_mat.data)
    hard_mat.setdiag(0)
    hard_mat.eliminate_zeros()
    hard_mat = sp.triu(hard_mat, k=1)

    hard_u, hard_v = hard_mat.nonzero()
    n_hard = len(hard_u)

    if n_hard >= n_needed:
        # Enough hard negatives
        idx = rng.choice(n_hard, n_needed, replace=False)
        return np.stack([hard_u[idx], hard_v[idx]], axis=1).astype(np.int64)

    # Not enough hard negatives — use all hard + fill with random non-fb pairs
    print(f"  [neg] Hard negatives available: {n_hard} / {n_needed} needed. "
          f"Filling remainder with random non-fb pairs.")

    hard_neg = np.stack([hard_u, hard_v], axis=1).astype(np.int64)

    # Random pool: NOT fb_friends, NOT self-loop, NOT already in hard_neg
    adj_neg_full = 1 - fb.todense() - np.eye(n)
    # Also exclude hard negatives already selected
    for u, v in hard_neg:
        adj_neg_full[u, v] = 0
        adj_neg_full[v, u] = 0
    adj_neg_full = np.triu(adj_neg_full)

    rand_u, rand_v = np.where(np.asarray(adj_neg_full) != 0)
    n_rand_needed = n_needed - n_hard
    n_rand = min(n_rand_needed, len(rand_u))

    idx = rng.choice(len(rand_u), n_rand, replace=False)
    rand_neg = np.stack([rand_u[idx], rand_v[idx]], axis=1).astype(np.int64)

    return np.vstack([hard_neg, rand_neg]).astype(np.int64)


# ── public API ─────────────────────────────────────────────────────────────────

def prepare_cns_data(
    removal_rate: float,
    run_seed:     int,
    dataset:      str   = "cns",
    src_dir:      str   = "./data/nets",
    val_frac:     float = 0.10,
):
    rng = np.random.RandomState(run_seed)

    # ── Load ──────────────────────────────────────────────────────────────────
    net, n, n_layers, directed, mpx, layers_id, p = load_netf(
        dataset, src_dir=src_dir
    )
    assert n_layers == 4, f"Expected 4 layers, got {n_layers}"
    directed = False

    edges_per_layer = []
    for li in range(n_layers):
        mask  = net[:, 0] == (li + 1)
        edges = net[mask][:, 1:].T
        edges_per_layer.append(edges)

    # ── Split fb_friends ──────────────────────────────────────────────────────
    fb_all = edges_per_layer[TARGET_LAYER].T
    E_fb   = len(fb_all)

    perm   = rng.permutation(E_fb)
    fb_all = fb_all[perm]

    n_test   = max(1, int(round(removal_rate * E_fb)))
    n_remain = E_fb - n_test
    n_val    = max(1, int(round(val_frac * n_remain)))
    n_train  = n_remain - n_val

    print(f"  [fb_friends split] total={E_fb}  "
          f"train={n_train}  val={n_val}  test={n_test}  "
          f"(removal_rate={removal_rate:.0%})")

    train_edges = fb_all[:n_train]
    val_edges   = fb_all[n_train : n_train + n_val]
    test_edges  = fb_all[n_train + n_val:]

    # Full fb_friends adjacency (used to exclude positives from negatives)
    fb_rows = np.concatenate([fb_all[:, 0], fb_all[:, 1]])
    fb_cols = np.concatenate([fb_all[:, 1], fb_all[:, 0]])
    adj_full_fb = sp.csr_matrix(
        (np.ones(len(fb_rows)), (fb_rows, fb_cols)), shape=(n, n)
    )

    # ── Build context union adjacency for hard negatives ───────────────────────
    # Union of BT (layer 2) + SMS (layer 1) + Calls (layer 0)
    ctx_rows, ctx_cols = [], []
    for li in [0, 1, 2]:   # calls, sms, bluetooth
        r, c = edges_per_layer[li][0], edges_per_layer[li][1]
        ctx_rows.append(r); ctx_rows.append(c)
        ctx_cols.append(c); ctx_cols.append(r)
    ctx_rows = np.concatenate(ctx_rows)
    ctx_cols = np.concatenate(ctx_cols)
    adj_context = sp.csr_matrix(
        (np.ones(len(ctx_rows)), (ctx_rows, ctx_cols)), shape=(n, n)
    )

    # Training adjacency for fb_friends
    tr_rows = np.concatenate([train_edges[:, 0], train_edges[:, 1]])
    tr_cols = np.concatenate([train_edges[:, 1], train_edges[:, 0]])
    adj_train_fb = sp.csr_matrix(
        (np.ones(len(tr_rows)), (tr_cols, tr_rows)), shape=(n, n)
    )

    # ── Build DGL graphs ──────────────────────────────────────────────────────
    g_train     = []
    g_train_pos = []
    g_test_pos  = []
    g_test_neg  = []
    g_val_pos   = []
    g_val_neg   = []
    adjs        = []

    empty_g = lambda: dgl.graph(([], []), num_nodes=n)

    for li in range(n_layers):
        if li != TARGET_LAYER:
            rows, cols = edges_per_layer[li][0], edges_per_layer[li][1]
            g_ctx = _build_sym_dgl(rows, cols, n)
            adjs.append(g_ctx.adj_external(scipy_fmt="csr"))
            g_train.append(g_ctx)
            for lst in (g_train_pos, g_test_pos, g_test_neg,
                        g_val_pos, g_val_neg):
                lst.append(empty_g())
        else:
            g_tr  = _build_sym_dgl(train_edges[:, 0], train_edges[:, 1], n)
            adjs.append(g_tr.adj_external(scipy_fmt="csr"))
            g_train.append(g_tr)

            tr_u, tr_v = g_tr.edges()
            g_train_pos.append(dgl.graph((tr_u, tr_v), num_nodes=n))

            # ── Test: hard negatives, fixed seed ──────────────────────────
            fixed_rng = np.random.RandomState(42)
            test_neg_edges = _sample_hard_negatives(
                adj_full_fb, adj_context,
                n_needed=len(test_edges), rng=fixed_rng,
            )
            te_src = np.concatenate([test_edges[:, 0],     test_edges[:, 1]])
            te_dst = np.concatenate([test_edges[:, 1],     test_edges[:, 0]])
            tn_src = np.concatenate([test_neg_edges[:, 0], test_neg_edges[:, 1]])
            tn_dst = np.concatenate([test_neg_edges[:, 1], test_neg_edges[:, 0]])

            g_test_pos.append(dgl.graph((te_src, te_dst), num_nodes=n))
            g_test_neg.append(dgl.graph((tn_src, tn_dst), num_nodes=n))

            # ── Val: hard negatives, fixed seed ───────────────────────────
            fixed_rng = np.random.RandomState(42)
            val_neg_edges = _sample_hard_negatives(
                adj_full_fb, adj_context,
                n_needed=len(val_edges), rng=fixed_rng,
            )
            ve_src = np.concatenate([val_edges[:, 0],     val_edges[:, 1]])
            ve_dst = np.concatenate([val_edges[:, 1],     val_edges[:, 0]])
            vn_src = np.concatenate([val_neg_edges[:, 0], val_neg_edges[:, 1]])
            vn_dst = np.concatenate([val_neg_edges[:, 1], val_neg_edges[:, 0]])

            g_val_pos.append(dgl.graph((ve_src, ve_dst), num_nodes=n))
            g_val_neg.append(dgl.graph((vn_src, vn_dst), num_nodes=n))

    # ── Supra-adjacency ───────────────────────────────────────────────────────
    supra_adj = build_supra_adj(adjs, p, directed=False)
    g_supra   = dgl.from_scipy(supra_adj)
    g_supra   = dgl.add_self_loop(g_supra)

    # ── Node features ─────────────────────────────────────────────────────────
    features = load_features(
        os.path.join(src_dir, dataset), features="features.pt"
    )
    if features is None:
        features = build_identity_matrix(n, n_layers)

    n_info = (n, n_layers, directed, mpx, layers_id, p)
    return (
        g_supra, g_train, g_train_pos,
        g_test_pos, g_test_neg,
        g_val_pos,  g_val_neg,
        features, n_info,
    )
