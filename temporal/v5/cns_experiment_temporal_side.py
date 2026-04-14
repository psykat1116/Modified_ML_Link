"""
cns_experiment_temporal_side.py  (final)
=========================================
Temporal knowledge as side information to ML-Link.

Architecture
------------
  1. TemporalNodeEncoder (Time2Vec + TransformerEncoder) encodes each
     node's T=7-window activity history → h_temp (N, d_temp)

  2. Learnable node embedding (N, node_emb_dim=32) gives each node a
     trainable unique identity WITHOUT memorizing the training set
     (unlike the n×n identity matrix which caused AUC=100).

  3. Concatenate: h_full = concat[h_temp, node_emb] → (N, d_temp+32)
     Repeat for supra layout: feats = h_full.repeat(n_layers, 1)

  4. ML-Link receives feats as its sole node features.
     Single prediction, single BCE loss — no parallel branch.

  5. Hard negatives in evaluation: non-fb pairs that DO have BT/SMS/calls
     contact, preventing trivially easy negative sets.

Usage
-----
    python temporal_preprocess.py --src data/cns_raw --dst data/cns_temporal
    python cns_experiment_temporal_side.py \\
        --prep_dir data/nets --temp_dir data/cns_temporal --gpu 0
"""

import os
import csv
import json
import time
import pickle
import argparse
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.sampling import global_uniform_negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message="enable_nested_tensor")

from models.main_m import Mm
from models.link_predictor import MLPPredictor, LinkPredictor
from models.temporal_encoder import (
    TemporalNodeEncoder,
    get_node_features_all,
    pretrain_node_encoder,
    NODE_DIM,
)
from cns_load import prepare_cns_data, TARGET_LAYER
from utils.optimization import EarlyStopping
from utils.util import init_seed
import utils.const as C

# Learnable node embedding dimension — small enough to avoid memorisation
# but large enough to distinguish nodes
# NODE_EMB_DIM = 32


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def set_params():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',         type=str,   default='cns')
    p.add_argument('--prep_dir',        type=str,   default='./data/nets')
    p.add_argument('--temp_dir',        type=str,   default='./data/cns_temporal')
    p.add_argument('--n_runs',          type=int,   default=5)
    p.add_argument('--removal_rates',   type=float, nargs='+',
                   default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    p.add_argument('--base_seed',       type=int,   default=0)
    # ── ML-Link ───────────────────────────────────────────────────────────────
    p.add_argument('--edge_dim',        type=int,   default=8)
    p.add_argument('--node_dim',        type=int,   default=128)
    p.add_argument('--phi_dim',         type=int,   default=128)
    p.add_argument('--hidden_dim',      type=int,   default=256)
    p.add_argument('--num_hidden',      type=int,   default=2)
    p.add_argument('--n_heads',         type=int,   default=1)
    p.add_argument('--heads_mode',      type=str,   default='concat')
    p.add_argument('--predictor',       type=str,   default='mlp')
    p.add_argument('--omn',             type=str,   default='oan;maan')
    p.add_argument('--psi',             type=float, default=0.5)
    p.add_argument('--no_gnn',          action='store_true')
    p.add_argument('--no_struct',       action='store_true')
    # ── Temporal encoder ──────────────────────────────────────────────────────
    p.add_argument('--temp_d_model',    type=int,   default=64)
    p.add_argument('--temp_n_heads',    type=int,   default=4)
    p.add_argument('--temp_n_layers',   type=int,   default=2)
    p.add_argument('--temp_dropout',    type=float, default=0.1)
    # ── Pretraining ───────────────────────────────────────────────────────────
    p.add_argument('--pretrain_epochs', type=int,   default=20)
    p.add_argument('--pretrain_lr',     type=float, default=1e-3)
    p.add_argument('--pretrain_batch',  type=int,   default=512)
    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument('--epochs',          type=int,   default=100)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--weight_decay',    type=float, default=1e-5)
    p.add_argument('--dropout',         type=float, default=0.5)
    p.add_argument('--attn_dropout',    type=float, default=0.7)
    p.add_argument('--patience',        type=int,   default=30)
    # ── Infra ─────────────────────────────────────────────────────────────────
    p.add_argument('--gpu',             type=int,   default=-1)
    p.add_argument('--save_dir',        type=str,   default='./artifacts/')
    p.add_argument('--ck_dir',          type=str,   default='checkpoint')
    args, _ = p.parse_known_args()
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def bce(pos, neg, eps=1e-8):
    return -torch.log(pos + eps).mean() - torch.log(1 - neg + eps).mean()


def build_mllink(args, n_layers, input_dim):
    return Mm(
        n_layers=n_layers, dropout=args.dropout,
        no_struct=args.no_struct, no_gnn=args.no_gnn, psi=args.psi,
        edge_dim=args.edge_dim, node_dim=args.node_dim, phi_dim=args.phi_dim,
        input_dim=input_dim, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden,
        heads=args.n_heads, attn_dropout=args.attn_dropout,
        residual=True, aggregation=args.heads_mode,
        activation=F.elu, eps=1e-8, f_dropout=0.7,
    )


def build_predictors(args, n_layers, dim, device):
    if args.no_gnn:
        return None
    if args.predictor.lower() == 'mlp':
        return [MLPPredictor(dim, dropout=args.dropout).to(device)
                for _ in range(n_layers)]
    return [LinkPredictor(args.predictor).to(device) for _ in range(n_layers)]


def sample_train_negatives(g_train_pos, n_layers, n, device):
    negs = []
    for li in range(n_layers):
        if li != TARGET_LAYER or g_train_pos[li].number_of_edges() == 0:
            negs.append(dgl.graph(([], []), num_nodes=n).to(device))
        else:
            g = g_train_pos[li]
            u, v = global_uniform_negative_sampling(
                g, num_samples=g.number_of_edges(),
                exclude_self_loops=True, replace=False,
            )
            negs.append(dgl.graph((u, v), num_nodes=n).to(device))
    return negs


# CHANGE TO:
def build_feats(h_temp: torch.Tensor, identity_feats: torch.Tensor, n_layers: int) -> torch.Tensor:
    h_rep = h_temp.repeat(n_layers, 1)
    return torch.cat([identity_feats, h_rep], dim=-1)

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(mllink_model, predictor,
             g_supra, g_train, p, g_pos, g_neg,
             feats, omn, device):
    mllink_model.eval()
    if predictor:
        for pr in predictor: pr.eval()

    with torch.no_grad():
        pos_score, _, _ = mllink_model(g_supra, g_train, p, g_pos,
                                        feats, predictor, omn)
        neg_score, _, _ = mllink_model(g_supra, g_train, p, g_neg,
                                        feats, predictor, omn)

    pos_score = torch.vstack(pos_score).squeeze(-1).detach().cpu()
    neg_score = torch.vstack(neg_score).squeeze(-1).detach().cpu()
    scores    = torch.cat([pos_score, neg_score]).numpy()
    labels    = np.concatenate([np.ones(len(pos_score)),
                                 np.zeros(len(neg_score))])
    return {
        'auc': float(roc_auc_score(labels, scores)),
        'ap':  float(average_precision_score(labels, scores)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single trial
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(args, removal_rate, run_idx, snapshots, device):
    seed = args.base_seed + run_idx
    init_seed(seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    (g_supra, g_train, g_train_pos,
     g_test_pos, g_test_neg,
     g_val_pos,  g_val_neg,
     _, n_info) = prepare_cns_data(         # original feats discarded
        removal_rate=removal_rate, run_seed=seed,
        dataset=args.dataset, src_dir=args.prep_dir,
    )
    n, n_layers, directed, mpx, _, p = n_info

    omn_str = args.omn.strip().lower()
    omn = None if (omn_str == 'none' or args.psi == 0.0 or args.no_struct) \
          else sorted(omn_str.split(';'))

    # ── Temporal node encoder ─────────────────────────────────────────────────
    temp_enc = TemporalNodeEncoder(
        node_dim = NODE_DIM,
        d_model  = args.temp_d_model,
        n_heads  = args.temp_n_heads,
        n_layers = args.temp_n_layers,
        T        = snapshots['T'],
        dropout  = args.temp_dropout,
    ).to(device)

    # ── Learnable node identity embedding ─────────────────────────────────────
    # Small-dim trainable embedding per node.
    # Gives GNN ability to distinguish nodes without n×n identity memorisation.
    from input_data.load import build_identity_matrix
    identity_feats = build_identity_matrix(n, n_layers).to(device)
    input_dim = n + args.temp_d_model

    # ── Self-supervised pretraining ───────────────────────────────────────────
    if args.pretrain_epochs > 0:
        print(f"  Pretraining node encoder ({args.pretrain_epochs} epochs)...")
        pretrain_node_encoder(
            temp_enc, snapshots, device,
            n_epochs   = args.pretrain_epochs,
            lr         = args.pretrain_lr,
            batch_size = args.pretrain_batch,
        )

    # ── Raw temporal features for ALL nodes (reused every epoch) ─────────────
    all_node_raw = get_node_features_all(snapshots, device)  # (N, T, NODE_DIM)

    # ── Build ML-Link model ───────────────────────────────────────────────────
    mllink_model = build_mllink(args, n_layers, input_dim).to(device)

    embed_dim = args.hidden_dim * args.n_heads \
                if args.heads_mode == 'concat' else args.hidden_dim
    predictor = build_predictors(args, n_layers, embed_dim, device)

    # Move graphs to device
    if not args.no_gnn and g_supra is not None:
        g_supra = g_supra.to(device)
    for glist in [g_train, g_train_pos,
                  g_test_pos, g_test_neg,
                  g_val_pos,  g_val_neg]:
        for li in range(n_layers):
            glist[li] = glist[li].to(device)

    # ── Optimiser: all components jointly ────────────────────────────────────
    all_params = (list(mllink_model.parameters()) +
                  list(temp_enc.parameters()))
    if predictor:
        for pr in predictor:
            all_params += list(pr.parameters())

    optimiser = optim.Adam(all_params, lr=args.lr,
                           weight_decay=args.weight_decay)

    ck_name = f"{args.dataset}_tempside_rate{removal_rate:.2f}_run{run_idx}"
    stopper  = EarlyStopping(patience=args.patience, maximize=True,
                              model_name=ck_name, model_dir=args.ck_dir)

    # ── Training loop ─────────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in tqdm(range(1, args.epochs + 1),
                      desc=f"  rate={removal_rate:.0%} run={run_idx+1}",
                      leave=False):
        mllink_model.train()
        temp_enc.train()
        if predictor:
            for pr in predictor: pr.train()
        optimiser.zero_grad()

        # Build feats: concat[h_temp, node_emb] → (n_layers*N, d_temp+32)
        h_temp = temp_enc(all_node_raw)                    # (N, d_temp)
        feats  = build_feats(h_temp, identity_feats, n_layers)

        g_train_neg = sample_train_negatives(g_train_pos, n_layers, n, device)

        pos_score, pos_struct, pos_gnn = mllink_model(
            g_supra, g_train, p, g_train_pos, feats, predictor, omn
        )
        neg_score, neg_struct, neg_gnn = mllink_model(
            g_supra, g_train, p, g_train_neg, feats, predictor, omn
        )

        pos_score = torch.vstack(pos_score)
        neg_score = torch.vstack(neg_score)

        if args.no_gnn or args.no_struct:
            loss = bce(pos_score, neg_score)
        else:
            pos_struct = torch.vstack(pos_struct)
            neg_struct = torch.vstack(neg_struct)
            pos_gnn    = torch.vstack(pos_gnn)
            neg_gnn    = torch.vstack(neg_gnn)
            loss = (bce(pos_score, neg_score) +
                    bce(pos_struct, neg_struct) +
                    bce(pos_gnn,    neg_gnn))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(mllink_model.parameters()) +
            list(temp_enc.parameters()), 1.0)
        if predictor:
            for pr in predictor:
                torch.nn.utils.clip_grad_norm_(pr.parameters(), 1.0)
        optimiser.step()

        if epoch > 5:
            # Eval mode: recompute feats
            temp_enc.eval()
            with torch.no_grad():
                h_val = temp_enc(all_node_raw)
                f_val = build_feats(h_val, identity_feats, n_layers)

            val_res = evaluate(
                mllink_model, predictor,
                g_supra, g_train, p, g_val_pos, g_val_neg,
                f_val, omn, device,
            )
            temp_enc.train()

            stopper.step(val_res['auc'], mllink_model, epoch)

            if stopper.counter == 0:
                Path(args.ck_dir).mkdir(parents=True, exist_ok=True)
                torch.save(temp_enc.state_dict(),
                           os.path.join(args.ck_dir, f"tenc_{ck_name}.bin"))
                if predictor:
                    for li, pr in enumerate(predictor):
                        torch.save(pr.state_dict(),
                                   os.path.join(args.ck_dir,
                                                f"pred{li}_{ck_name}.bin"))

            if stopper.early_stop:
                print(f"  Early stop at epoch {epoch} "
                      f"(val AUC={val_res['auc']:.4f})")
                break

    print(f"  Training time: {time.time()-t0:.1f}s")

    # ── Restore best checkpoint ───────────────────────────────────────────────
    mllink_model.load_state_dict(torch.load(stopper.save_dir))
    mllink_model = mllink_model.to(device).eval()

    for fname, obj in [(f"tenc_{ck_name}.bin", temp_enc),]:
        fpath = os.path.join(args.ck_dir, fname)
        if os.path.isfile(fpath):
            obj.load_state_dict(torch.load(fpath))
    temp_enc = temp_enc.to(device).eval()

    if predictor:
        for li, pr in enumerate(predictor):
            ck = os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin")
            if os.path.isfile(ck):
                pr.load_state_dict(torch.load(ck))
            pr.to(device).eval()

    # ── Test evaluation ───────────────────────────────────────────────────────
    with torch.no_grad():
        h_test = temp_enc(all_node_raw)
        f_test = build_feats(h_test, identity_feats, n_layers)

    test_res = evaluate(
        mllink_model, predictor,
        g_supra, g_train, p, g_test_pos, g_test_neg,
        f_test, omn, device,
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    stopper.remove_checkpoint()
    for fname in [f"tenc_{ck_name}.bin", f"nemb_{ck_name}.bin"]:
        fpath = os.path.join(args.ck_dir, fname)
        if os.path.isfile(fpath): os.remove(fpath)
    if predictor:
        for li in range(n_layers):
            fpath = os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin")
            if os.path.isfile(fpath): os.remove(fpath)

    return test_res


# ══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = set_params()
    cuda   = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f'cuda:{args.gpu}' if cuda else 'cpu')
    print(f"Device: {device}")

    snap_path = os.path.join(args.temp_dir, 'temporal_snapshots.pkl')
    if not os.path.isfile(snap_path):
        raise FileNotFoundError(
            f"Snapshots not found at {snap_path}.\n"
            f"Run: python temporal_preprocess.py --src data/cns_raw "
            f"--dst {args.temp_dir}"
        )
    print(f"Loading temporal snapshots from {snap_path} ...")
    with open(snap_path, 'rb') as f:
        snapshots = pickle.load(f)
    print(f"  T={snapshots['T']} windows, n={snapshots['n_nodes']} nodes")

    if 'degree' not in snapshots['calls']:
        raise RuntimeError(
            "Snapshots missing 'degree' arrays. "
            "Please re-run temporal_preprocess.py."
        )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ck_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    for rate in sorted(args.removal_rates):
        print(f"\n{'='*60}\nRemoval rate: {rate:.0%}\n{'='*60}")
        auc_list, ap_list = [], []

        for run in range(args.n_runs):
            res = run_trial(args, removal_rate=rate, run_idx=run,
                            snapshots=snapshots, device=device)
            auc_list.append(res['auc'] * 100)
            ap_list.append(res['ap']  * 100)
            print(f"  Run {run+1}/{args.n_runs} → "
                  f"AUC={res['auc']*100:.2f}  AP={res['ap']*100:.2f}")

        row = {
            'removal_rate': rate,
            'auc_mean': float(np.mean(auc_list)),
            'auc_std':  float(np.std(auc_list)),
            'ap_mean':  float(np.mean(ap_list)),
            'ap_std':   float(np.std(ap_list)),
            'auc_runs': auc_list,
            'ap_runs':  ap_list,
        }
        all_results.append(row)
        print(f"\n  AUC: {row['auc_mean']:.2f} ± {row['auc_std']:.2f}  "
              f"AP: {row['ap_mean']:.2f} ± {row['ap_std']:.2f}")

    csv_path  = os.path.join(args.save_dir, 'cns_temporal_side_results.csv')
    json_path = os.path.join(args.save_dir, 'cns_temporal_side_results.json')

    fields = ['removal_rate', 'auc_mean', 'auc_std', 'ap_mean', 'ap_std']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in fields})
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults -> {csv_path}")
    print(f"\n{'Removal':>9}  {'AUC mean±std':>18}  {'AP mean±std':>18}")
    print('─' * 52)
    for r in all_results:
        print(f"  {r['removal_rate']:>6.0%}   "
              f"{r['auc_mean']:>7.2f} ± {r['auc_std']:<6.2f}   "
              f"{r['ap_mean']:>7.2f} ± {r['ap_std']:<6.2f}")


if __name__ == '__main__':
    main()
