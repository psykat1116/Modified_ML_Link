"""
models/temporal_encoder.py  (v6 - Time2Vec + Transformer, side information)
=============================================================================
Uses the original Time2Vec + TransformerEncoder backbone (pre-DySAT).
Output is used as side information (node embeddings) appended to ML-Link
node features — NOT as a separate prediction branch.

Architecture:
    node_features (N, T, NODE_DIM)
        ↓
    Linear(NODE_DIM → d_model)
        ↓
    + Time2Vec positional encoding
        ↓
    TransformerEncoder (n_layers, n_heads)
        ↓
    mean pool over T
        ↓
    h_temp (N, d_model)   ← concatenated to ML-Link feats

NODE_DIM = 9 features per window per node.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_DIM     = 9
CONTEXT_WINS = 5


# ══════════════════════════════════════════════════════════════════════════════
# Time2Vec
# ══════════════════════════════════════════════════════════════════════════════

class Time2Vec(nn.Module):
    """
    Learnable time encoding (Kazemi et al., 2019).
    v[t, 0]   = w0*t + b0          (linear / trend)
    v[t, i>0] = sin(w_i * t + b_i) (periodic)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1) * 0.1)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(torch.randn(d_model - 1) * 0.1)
        self.b  = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (T,) → (T, d_model)"""
        linear   = (self.w0 * t + self.b0).unsqueeze(-1)
        periodic = torch.sin(self.w * t.unsqueeze(-1) + self.b)
        return torch.cat([linear, periodic], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Transformer backbone (original pre-DySAT version)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBackbone(nn.Module):
    """
    Input:  (B, T, input_dim)
    Output: (B, d_model)
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers, T, dropout):
        super().__init__()
        self.T    = T
        self.proj = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.time2vec = Time2Vec(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → (B, d_model)"""
        x     = self.proj(x)
        t_idx = torch.arange(self.T, dtype=torch.float32, device=x.device)
        x     = x + self.time2vec(t_idx).unsqueeze(0)
        x     = self.transformer(x)
        return x.mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Temporal Node Encoder
# ══════════════════════════════════════════════════════════════════════════════

class TemporalNodeEncoder(nn.Module):
    """
    Encodes each node's temporal activity into a d_model embedding
    using Time2Vec + Transformer.

    Output h_temp (N, d_model) is appended to ML-Link node features.
    """
    def __init__(
        self,
        node_dim: int   = NODE_DIM,
        d_model:  int   = 64,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        T:        int   = 7,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.backbone = TransformerBackbone(
            node_dim, d_model, n_heads, n_layers, T, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, NODE_DIM) → (N, d_model)"""
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def get_node_features_all(snapshots: dict, device: torch.device) -> torch.Tensor:
    """
    Compute temporal node features for ALL N nodes.
    Called once per trial — result reused every epoch.
    Returns (N, T, NODE_DIM).
    """
    n     = snapshots['n_nodes']
    nodes = np.arange(n, dtype=np.int32)
    return get_node_features(nodes, snapshots, device)


def get_node_features(
    node_arr:  np.ndarray,
    snapshots: dict,
    device:    torch.device,
) -> torch.Tensor:
    """
    Returns (E, T, NODE_DIM=9) for a given set of nodes.

    Feature layout per window:
      [0] calls_count_norm
      [1] calls_partners_norm
      [2] sms_count_norm
      [3] sms_partners_norm
      [4] bt_count_norm
      [5] bt_rssi_mean_norm
      [6] bt_partners_norm
      [7] bt_active          binary
      [8] calls_active       binary
    """
    T_win = snapshots['T']
    E     = len(node_arr)
    stats = snapshots['stats']

    denom_calls = np.log1p(stats['max_calls_count']) or 1.0
    denom_sms   = np.log1p(stats['max_sms_count'])   or 1.0
    denom_bt    = np.log1p(stats['max_bt_count'])     or 1.0

    features = np.zeros((E, T_win, NODE_DIM), dtype=np.float32)

    for t in range(T_win):
        c_cnt  = snapshots['calls']['degree'][t][node_arr].astype(np.float32)
        c_part = np.array(
            (snapshots['calls']['count'][t][node_arr] > 0).sum(axis=1)
        ).flatten().astype(np.float32)

        s_cnt  = snapshots['sms']['degree'][t][node_arr].astype(np.float32)
        s_part = np.array(
            (snapshots['sms']['count'][t][node_arr] > 0).sum(axis=1)
        ).flatten().astype(np.float32)

        b_cnt  = snapshots['bluetooth']['degree'][t][node_arr].astype(np.float32)
        b_part = np.array(
            (snapshots['bluetooth']['count'][t][node_arr] > 0).sum(axis=1)
        ).flatten().astype(np.float32)

        b_rssi_node = np.array(
            snapshots['bluetooth']['rssi_sum'][t][node_arr].sum(axis=1)
        ).flatten().astype(np.float32)
        b_rssi_mean = np.where(b_cnt > 0, b_rssi_node / np.maximum(b_cnt, 1), 0.0)
        b_rssi_norm = np.where(
            b_cnt > 0,
            np.clip((b_rssi_mean + 100.0) / 60.0, 0.0, 1.0),
            0.0,
        )

        features[:, t, 0] = np.log1p(c_cnt)  / denom_calls
        features[:, t, 1] = np.log1p(c_part) / denom_calls
        features[:, t, 2] = np.log1p(s_cnt)  / denom_sms
        features[:, t, 3] = np.log1p(s_part) / denom_sms
        features[:, t, 4] = np.log1p(b_cnt)  / denom_bt
        features[:, t, 5] = b_rssi_norm
        features[:, t, 6] = np.log1p(b_part) / denom_bt
        features[:, t, 7] = (b_cnt > 0).astype(np.float32)
        features[:, t, 8] = (c_cnt > 0).astype(np.float32)

    return torch.from_numpy(features).to(device)


# ══════════════════════════════════════════════════════════════════════════════
# Self-supervised pretraining
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_node_encoder(
    encoder:      TemporalNodeEncoder,
    snapshots:    dict,
    device:       torch.device,
    n_epochs:     int   = 20,
    lr:           float = 1e-3,
    batch_size:   int   = 512,
    context_wins: int   = CONTEXT_WINS,
):
    """
    Self-supervised pretraining:
      Input : node features for windows 0..context_wins-1 (future zeroed out)
      Target: did (u,v) interact in windows context_wins..T-1?
    A temporary dot-product head is used and discarded after pretraining.
    """
    T   = snapshots['T']
    rng = np.random.RandomState(0)

    pos_set = set()
    for t in range(context_wins, T):
        for lk in ['calls', 'sms', 'bluetooth']:
            rs, cs = snapshots[lk]['count'][t].nonzero()
            for r, c in zip(rs, cs):
                if r != c:
                    pos_set.add((min(r, c), max(r, c)))

    ctx_set = set()
    for t in range(context_wins):
        for lk in ['calls', 'sms', 'bluetooth']:
            rs, cs = snapshots[lk]['count'][t].nonzero()
            for r, c in zip(rs, cs):
                if r != c:
                    ctx_set.add((min(r, c), max(r, c)))

    neg_list = list(ctx_set - pos_set)
    pos_list = list(pos_set)

    print(f"  [Pretrain] context_wins={context_wins}  "
          f"pos={len(pos_list):,}  neg={len(neg_list):,}")

    if len(pos_list) == 0 or len(neg_list) == 0:
        print("  [Pretrain] not enough pairs — skipping.")
        return

    d    = encoder.d_model
    head = nn.Linear(d * 2, 1).to(device)
    nn.init.xavier_uniform_(head.weight)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()), lr=lr
    )
    encoder.train()
    head.train()

    all_node_feats = get_node_features_all(snapshots, device)  # (N, T, NODE_DIM)

    for epoch in range(1, n_epochs + 1):
        n_pos = min(batch_size // 2, len(pos_list))
        n_neg = min(batch_size // 2, len(neg_list))

        pos_idx = rng.choice(len(pos_list), n_pos, replace=False)
        neg_idx = rng.choice(len(neg_list), n_neg, replace=False)

        pairs = [pos_list[i] for i in pos_idx] + [neg_list[i] for i in neg_idx]
        u_arr = np.array([p[0] for p in pairs], dtype=np.int64)
        v_arr = np.array([p[1] for p in pairs], dtype=np.int64)

        node_feats = all_node_feats.clone()
        node_feats[:, context_wins:, :] = 0.0

        h_all  = encoder(node_feats)
        h_u    = h_all[u_arr]
        h_v    = h_all[v_arr]
        scores = torch.sigmoid(head(torch.cat([h_u, h_v], dim=-1)))
        labels = torch.cat([
            torch.ones(n_pos, 1), torch.zeros(n_neg, 1)
        ]).to(device)

        optimizer.zero_grad()
        loss = F.binary_cross_entropy(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(head.parameters()), 1.0
        )
        optimizer.step()

        if epoch % 5 == 0:
            print(f"    Pretrain epoch {epoch:>3}/{n_epochs}  loss={loss.item():.4f}")

    print("  [Pretrain] done.")
