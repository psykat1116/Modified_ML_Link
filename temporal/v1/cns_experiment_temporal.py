"""
cns_experiment_temporal.py  (v4 - DySAT + confidence combining)
================================================================
Changes from v3:
  1. temporal_encoder uses DySATBackbone (structural + causal temporal attention)
  2. combine_scores: agreement→max/min, disagreement→confidence-weighted
  3. per-layer beta removed — combining is fully rule-based + learnable alpha
  4. Docstring updated

Usage:
    python temporal_preprocess.py --src data/cns_raw --dst data/cns_temporal
    python cns_experiment_temporal.py --prep_dir data/nets --temp_dir data/cns_temporal --gpu 0
"""

import os
import csv
import json
import time
import pickle
import argparse
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

from models.main_m import Mm
from models.link_predictor import MLPPredictor, LinkPredictor
from models.temporal_encoder import (
    MixtureTemporalEncoder,
    get_pair_features,
    get_node_features,
    pretrain_temporal_encoder,
    PAIR_DIM, NODE_DIM,
)
from input_data.cns_load import prepare_cns_data, TARGET_LAYER
from utils.optimization import EarlyStopping
from utils.util import init_seed
import utils.const as C


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
    # ── Warm-start ────────────────────────────────────────────────────────────
    p.add_argument('--warmup_epochs',   type=int,   default=10)
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


def set_requires_grad(modules_and_params, value: bool):
    for item in modules_and_params:
        if isinstance(item, nn.Module):
            for p in item.parameters():
                p.requires_grad = value
        elif isinstance(item, nn.Parameter):
            item.requires_grad = value


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


def pairs_from_graph(g):
    u, v = g.edges()
    return u.cpu().numpy().astype(np.int32), v.cpu().numpy().astype(np.int32)


def get_all_features(u_arr, v_arr, snapshots, device):
    pair_feat = get_pair_features(u_arr, v_arr, snapshots, device)
    node_u    = get_node_features(u_arr, snapshots, device)
    node_v    = get_node_features(v_arr, snapshots, device)
    return pair_feat, node_u, node_v


# ══════════════════════════════════════════════════════════════════════════════
# Agreement + confidence-based combining
# ══════════════════════════════════════════════════════════════════════════════

def combine_scores(p_ml, p_temp, alpha_param):
    """
    Agreement rule:
      - Both >= 0.5 (both say edge):   return max(p_ml, p_temp)
      - Both  < 0.5 (both say no edge): return min(p_ml, p_temp)
      - Disagree: confidence-weighted blend (more confident branch wins)

    alpha_param: global learnable scalar that soft-blends between the
                 rule-based result and a simple weighted average.
                 Initialized to 0 → sigmoid=0.5 → equal starting weight.
    """
    threshold = 0.5
    alpha     = torch.sigmoid(alpha_param)

    ml_pos   = (p_ml   >= threshold)
    temp_pos = (p_temp >= threshold)

    # Agreement: both predict edge → take max; both predict no edge → take min
    p_agree = torch.where(
        ml_pos & temp_pos,
        torch.maximum(p_ml, p_temp),
        torch.where(
            ~ml_pos & ~temp_pos,
            torch.minimum(p_ml, p_temp),
            (p_ml + p_temp) / 2.0,   # fallback (not reached in disagree case)
        )
    )

    # Disagreement: weight by confidence (distance from 0.5)
    conf_ml   = torch.abs(p_ml   - threshold)
    conf_temp = torch.abs(p_temp - threshold)
    total     = conf_ml + conf_temp + 1e-8
    p_disagree = (conf_ml / total) * p_ml + (conf_temp / total) * p_temp

    # Binary agreement indicator
    agree = ((ml_pos & temp_pos) | (~ml_pos & ~temp_pos)).float()

    # Rule-based result
    p_rule = agree * p_agree + (1 - agree) * p_disagree

    # Soft blend: alpha lets the model learn to shift away from pure rule
    p_avg  = alpha * p_ml + (1 - alpha) * p_temp
    return alpha * p_rule + (1 - alpha) * p_avg


# ══════════════════════════════════════════════════════════════════════════════
# Combined forward
# ══════════════════════════════════════════════════════════════════════════════

def forward_combined(
    mllink_model, predictor, temporal_model,
    alpha_param,
    g_supra, g_train, p, g_edges_pos, g_edges_neg,
    feats, omn, snapshots, device,
    compute_full_loss=True,
):
    # ── ML-Link ───────────────────────────────────────────────────────────────
    ml_pos, st_pos, gn_pos = mllink_model(g_supra, g_train, p, g_edges_pos,
                                           feats, predictor, omn)
    ml_neg, st_neg, gn_neg = mllink_model(g_supra, g_train, p, g_edges_neg,
                                           feats, predictor, omn)
    p_ml_pos = ml_pos[0]
    p_ml_neg = ml_neg[0]

    # ── Temporal (DySAT mixture) ──────────────────────────────────────────────
    u_pos, v_pos = pairs_from_graph(g_edges_pos[TARGET_LAYER])
    u_neg, v_neg = pairs_from_graph(g_edges_neg[TARGET_LAYER])

    pf_pos, nu_pos, nv_pos = get_all_features(u_pos, v_pos, snapshots, device)
    pf_neg, nu_neg, nv_neg = get_all_features(u_neg, v_neg, snapshots, device)

    p_temp_pos = temporal_model(pf_pos, nu_pos, nv_pos)
    p_temp_neg = temporal_model(pf_neg, nu_neg, nv_neg)

    # ── Confidence-based combine ──────────────────────────────────────────────
    p_final_pos = combine_scores(p_ml_pos, p_temp_pos, alpha_param)
    p_final_neg = combine_scores(p_ml_neg, p_temp_neg, alpha_param)

    if not compute_full_loss:
        return p_final_pos, p_final_neg, None

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss = bce(p_final_pos, p_final_neg)
    loss = loss + bce(p_ml_pos,   p_ml_neg)
    loss = loss + bce(p_temp_pos, p_temp_neg)
    if st_pos is not None:
        loss = loss + bce(torch.vstack(st_pos), torch.vstack(st_neg))
    if gn_pos is not None:
        loss = loss + bce(torch.vstack(gn_pos), torch.vstack(gn_neg))

    return p_final_pos, p_final_neg, loss


# ══════════════════════════════════════════════════════════════════════════════
# Temporal-only forward (warm-start phase)
# ══════════════════════════════════════════════════════════════════════════════

def forward_temporal_only(temporal_model, g_edges_pos, g_edges_neg,
                           snapshots, device):
    u_pos, v_pos = pairs_from_graph(g_edges_pos[TARGET_LAYER])
    u_neg, v_neg = pairs_from_graph(g_edges_neg[TARGET_LAYER])

    pf_pos, nu_pos, nv_pos = get_all_features(u_pos, v_pos, snapshots, device)
    pf_neg, nu_neg, nv_neg = get_all_features(u_neg, v_neg, snapshots, device)

    p_pos = temporal_model(pf_pos, nu_pos, nv_pos)
    p_neg = temporal_model(pf_neg, nu_neg, nv_neg)
    return bce(p_pos, p_neg)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    mllink_model, predictor, temporal_model,
    alpha_param,
    g_supra, g_train, p, g_pos, g_neg,
    feats, omn, snapshots, device,
):
    mllink_model.eval()
    temporal_model.eval()
    if predictor:
        for pr in predictor: pr.eval()

    with torch.no_grad():
        p_pos, p_neg, _ = forward_combined(
            mllink_model, predictor, temporal_model,
            alpha_param,
            g_supra, g_train, p, g_pos, g_neg,
            feats, omn, snapshots, device,
            compute_full_loss=False,
        )

    scores = torch.cat([p_pos, p_neg]).squeeze(-1).cpu().numpy()
    labels = np.concatenate([np.ones(len(p_pos)), np.zeros(len(p_neg))])
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
     feats, n_info) = prepare_cns_data(
        removal_rate=removal_rate, run_seed=seed,
        dataset=args.dataset, src_dir=args.prep_dir,
    )
    n, n_layers, directed, mpx, _, p = n_info

    omn_str = args.omn.strip().lower()
    omn = None if (omn_str == 'none' or args.psi == 0.0 or args.no_struct) \
          else sorted(omn_str.split(';'))

    input_dim = None
    if feats is not None:
        input_dim = feats.shape[1]
        feats = feats.to(device)

    if not args.no_gnn and g_supra is not None:
        g_supra = g_supra.to(device)
    for glist in [g_train, g_train_pos,
                  g_test_pos, g_test_neg,
                  g_val_pos,  g_val_neg]:
        for li in range(n_layers):
            glist[li] = glist[li].to(device)

    # ── Build models ──────────────────────────────────────────────────────────
    mllink_model = build_mllink(args, n_layers, input_dim).to(device)
    embed_dim    = args.hidden_dim * args.n_heads \
                   if args.heads_mode == 'concat' else args.hidden_dim
    predictor    = build_predictors(args, n_layers, embed_dim, device)

    temporal_model = MixtureTemporalEncoder(
        pair_dim  = PAIR_DIM,
        node_dim  = NODE_DIM,
        d_model   = args.temp_d_model,
        n_heads   = args.temp_n_heads,
        n_layers  = args.temp_n_layers,
        T         = snapshots['T'],
        dropout   = args.temp_dropout,
    ).to(device)

    # Global learnable blend parameter: sigmoid(0) = 0.5 → equal start
    log_alpha = nn.Parameter(torch.zeros(1, device=device))

    # ── Step 1: Self-supervised pretraining ───────────────────────────────────
    if args.pretrain_epochs > 0:
        print(f"  Pretraining temporal encoder ({args.pretrain_epochs} epochs)...")
        pretrain_temporal_encoder(
            temporal_model, snapshots, device,
            n_epochs   = args.pretrain_epochs,
            lr         = args.pretrain_lr,
            batch_size = args.pretrain_batch,
        )

    # ── Step 2: Warm-start (temporal only, ML-Link frozen) ────────────────────
    if args.warmup_epochs > 0:
        print(f"  Warm-start: temporal branch only ({args.warmup_epochs} epochs)...")
        set_requires_grad([mllink_model] + (predictor or []), False)
        wu_optim = optim.Adam(
            list(temporal_model.parameters()) + [log_alpha],
            lr=args.lr, weight_decay=args.weight_decay,
        )
        temporal_model.train()
        for _ in range(args.warmup_epochs):
            wu_optim.zero_grad()
            g_train_neg = sample_train_negatives(g_train_pos, n_layers, n, device)
            loss = forward_temporal_only(
                temporal_model, g_train_pos, g_train_neg, snapshots, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), 1.0)
            wu_optim.step()
        set_requires_grad([mllink_model] + (predictor or []), True)
        print("  Warm-start done. Unfreezing ML-Link.")

    # ── Step 3: Joint training ────────────────────────────────────────────────
    all_params = (list(mllink_model.parameters()) +
                  list(temporal_model.parameters()) +
                  [log_alpha])
    if predictor:
        for pr in predictor:
            all_params += list(pr.parameters())

    optimiser = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)

    ck_name = f"{args.dataset}_temp_rate{removal_rate:.2f}_run{run_idx}"
    stopper  = EarlyStopping(patience=args.patience, maximize=True,
                              model_name=ck_name, model_dir=args.ck_dir)

    t0 = time.time()
    for epoch in tqdm(range(1, args.epochs + 1),
                      desc=f"  rate={removal_rate:.0%} run={run_idx+1}",
                      leave=False):
        mllink_model.train()
        temporal_model.train()
        if predictor:
            for pr in predictor: pr.train()
        optimiser.zero_grad()

        g_train_neg = sample_train_negatives(g_train_pos, n_layers, n, device)

        _, _, loss = forward_combined(
            mllink_model, predictor, temporal_model,
            log_alpha,
            g_supra, g_train, p, g_train_pos, g_train_neg,
            feats, omn, snapshots, device,
            compute_full_loss=True,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(mllink_model.parameters()) + list(temporal_model.parameters()), 1.0
        )
        if predictor:
            for pr in predictor:
                torch.nn.utils.clip_grad_norm_(pr.parameters(), 1.0)
        optimiser.step()

        if epoch > 5:
            val_res = evaluate(
                mllink_model, predictor, temporal_model,
                log_alpha,
                g_supra, g_train, p, g_val_pos, g_val_neg,
                feats, omn, snapshots, device,
            )
            stopper.step(val_res['auc'], mllink_model, epoch)

            if stopper.counter == 0:
                Path(args.ck_dir).mkdir(parents=True, exist_ok=True)
                torch.save(temporal_model.state_dict(),
                           os.path.join(args.ck_dir, f"temp_{ck_name}.bin"))
                torch.save(log_alpha.data,
                           os.path.join(args.ck_dir, f"alpha_{ck_name}.bin"))
                if predictor:
                    for li, pr in enumerate(predictor):
                        torch.save(pr.state_dict(),
                                   os.path.join(args.ck_dir,
                                                f"pred{li}_{ck_name}.bin"))

            if stopper.early_stop:
                print(f"  Early stop at epoch {epoch} "
                      f"(val AUC={val_res['auc']:.4f})")
                break

    print(f"  Training time: {time.time()-t0:.1f}s  "
          f"alpha={torch.sigmoid(log_alpha).item():.3f}")

    # ── Restore best checkpoint ───────────────────────────────────────────────
    mllink_model.load_state_dict(torch.load(stopper.save_dir))
    mllink_model = mllink_model.to(device).eval()

    temp_ck = os.path.join(args.ck_dir, f"temp_{ck_name}.bin")
    if os.path.isfile(temp_ck):
        temporal_model.load_state_dict(torch.load(temp_ck))
    temporal_model = temporal_model.to(device).eval()

    alpha_ck = os.path.join(args.ck_dir, f"alpha_{ck_name}.bin")
    if os.path.isfile(alpha_ck):
        log_alpha.data = torch.load(alpha_ck)

    if predictor:
        for li, pr in enumerate(predictor):
            ck = os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin")
            if os.path.isfile(ck):
                pr.load_state_dict(torch.load(ck))
            pr.to(device).eval()

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_res = evaluate(
        mllink_model, predictor, temporal_model,
        log_alpha,
        g_supra, g_train, p, g_test_pos, g_test_neg,
        feats, omn, snapshots, device,
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    stopper.remove_checkpoint()
    for fname in [f"temp_{ck_name}.bin", f"alpha_{ck_name}.bin"]:
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

    csv_path  = os.path.join(args.save_dir, 'cns_temporal_v4_results.csv')
    json_path = os.path.join(args.save_dir, 'cns_temporal_v4_results.json')

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
