"""
cns_experiment.py  (k-hop extended version)
============================================
Same as before, plus --k_hop argument (1 = original, 3–5 = extended).

K-hop adjacency is computed ONCE per run (before the epoch loop) and
passed to every model(...) call via the g_khop argument.

The GNN-NE component is unaffected — it always uses the original 1-hop
supra-adjacency.  Only the structural components (ISL / MAAN / OAN) see
the wider neighbourhood.
"""

import os
import time
import argparse
import json
import csv
import numpy as np
import scipy.sparse as sp
import warnings
from scipy.sparse import SparseEfficiencyWarning
import torch
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl import AddReverse, ToSimple
from dgl.sampling import global_uniform_negative_sampling
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

from models.main_m import Mm
from models.link_predictor import MLPPredictor, LinkPredictor
from cns_load import prepare_cns_data, TARGET_LAYER
from utils.optimization import EarlyStopping
from utils.util import init_seed
import utils.const as C

warnings.filterwarnings("ignore", category = SparseEfficiencyWarning)

# ══════════════════════════════════════════════════════════════════════════════
# K-hop adjacency helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_khop_adj(adj_sp: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """
    Return the binary up-to-k-hop adjacency:
        sign(A + A^2 + ... + A^k)
    Self-loops are removed from the result.
    """
    result = adj_sp.astype(float).copy()
    power  = adj_sp.astype(float).copy()
    for hop in range(2, k + 1):
        power   = power @ adj_sp
        result += power * (1.0 / (2 ** (hop - 1)))
    result = result.tocsr()
    result.data = np.where(result.data > 0, result.data, 0)
    result = result.tolil()
    result.setdiag(0)
    result = result.tocsr()
    result.eliminate_zeros()
    return result

def build_khop_dgl(adj_khop: sp.csr_matrix, n: int) -> dgl.DGLGraph:
    """
    Build a symmetric (bidirectional) DGL graph from a k-hop adjacency matrix.
    Edge weight 'w' is set to 1.0 for compatibility with sfg.py.
    """
    # Keep upper-triangle unique edges, then add reverse
    adj_up = sp.triu(adj_khop, k=1)
    g_up = dgl.from_scipy(adj_up, eweight_name="w")

    transform_rev  = AddReverse(copy_edata=True)
    transform_simp = ToSimple(aggregator="arbitrary", return_counts="w")
    g = transform_simp(transform_rev(g_up))
    g.edata["w"] = g.edata["w"].float()
    g.edata.pop(dgl.NID, None)

    if g.num_nodes() < n:
        g = dgl.add_nodes(g, n - g.num_nodes())

    return g


def precompute_khop_graphs(g_train: list, k: int, device) -> list:
    """
    Given the list of 1-hop training DGL graphs (one per layer),
    compute k-hop DGL graphs and move them to `device`.

    For k=1 returns g_train unchanged (no extra computation).
    """
    if k == 1:
        return g_train   # no-op: original 1-hop behaviour

    n = g_train[0].num_nodes()
    g_khop = []
    for li, g in enumerate(g_train):
        adj_1hop = g.adj_external(scipy_fmt="csr")
        adj_k    = compute_khop_adj(adj_1hop, k)
        gk       = build_khop_dgl(adj_k, n).to(device)
        g_khop.append(gk)
        print(f"  [k-hop] layer {li}: 1-hop edges={adj_1hop.nnz//2}  "
              f"{k}-hop edges={adj_k.nnz//2}")
    return g_khop


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str,   default="cns")
    parser.add_argument("--prep_dir", type=str,   default="./data/nets")
    parser.add_argument("--n_runs",   type=int,   default=5)
    parser.add_argument("--removal_rates", type=float, nargs="+",
                        default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    parser.add_argument("--base_seed", type=int,  default=0)
    parser.add_argument("--k_hop", type=int, default=1,
                        help="Neighbourhood hop depth for structural features "
                             "(1=original, 3-5=extended).")
    parser.add_argument("--edge_dim",     type=int,   default=8)
    parser.add_argument("--node_dim",     type=int,   default=128)
    parser.add_argument("--phi_dim",      type=int,   default=128)
    parser.add_argument("--hidden_dim",   type=int,   default=256)
    parser.add_argument("--num_hidden",   type=int,   default=2)
    parser.add_argument("--n_heads",      type=int,   default=1)
    parser.add_argument("--heads_mode",   type=str,   default="concat")
    parser.add_argument("--predictor",    type=str,   default="mlp")
    parser.add_argument("--omn",          type=str,   default="oan;maan")
    parser.add_argument("--psi",          type=float, default=0.5)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--attn_dropout", type=float, default=0.7)
    parser.add_argument("--patience",     type=int,   default=30)
    parser.add_argument("--no_gnn",    action="store_true")
    parser.add_argument("--no_struct", action="store_true")
    parser.add_argument("--gpu",      type=int,  default=-1)
    parser.add_argument("--save_dir", type=str,  default="./artifacts/")
    parser.add_argument("--ck_dir",   type=str,  default="checkpoint")

    args, _ = parser.parse_known_args()
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Model helpers  (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(args, n_layers, input_dim):
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
    if args.predictor.lower() == "mlp":
        return [MLPPredictor(dim, dropout=args.dropout).to(device)
                for _ in range(n_layers)]
    return [LinkPredictor(args.predictor).to(device) for _ in range(n_layers)]


def compute_bce(pos, neg, eps=1e-8):
    return -torch.log(pos + eps).mean() - torch.log(1 - neg + eps).mean()


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


def evaluate(model, predictor, g_supra, g_train, p, feats,
             g_pos, g_neg, inter, device, g_khop=None):
    model.eval()
    if predictor:
        for pr in predictor:
            pr.eval()

    with torch.no_grad():
        pos_s, _, _ = model(g_supra, g_train, p, g_pos,  feats, predictor, inter, g_khop)
        neg_s, _, _ = model(g_supra, g_train, p, g_neg,  feats, predictor, inter, g_khop)

    pos_s = torch.vstack(pos_s).detach().cpu().squeeze(-1)
    neg_s = torch.vstack(neg_s).detach().cpu().squeeze(-1)
    scores = torch.cat([pos_s, neg_s]).numpy()
    labels = np.concatenate([np.ones(pos_s.shape[0]), np.zeros(neg_s.shape[0])])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap":  float(average_precision_score(labels, scores)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single-trial training
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(args, removal_rate, run_idx, device):
    seed = args.base_seed + run_idx
    init_seed(seed)

    # ── data ──────────────────────────────────────────────────────────────────
    (g_supra, g_train, g_train_pos,
     g_test_pos, g_test_neg, g_val_pos, g_val_neg,
     feats, n_info) = prepare_cns_data(
        removal_rate=removal_rate, run_seed=seed,
        dataset=args.dataset, src_dir=args.prep_dir,
    )
    n, n_layers, directed, mpx, _, p = n_info

    # ── k-hop graphs (computed once, reused every epoch) ──────────────────────
    print(f"  Computing {args.k_hop}-hop adjacencies ...")
    g_khop = precompute_khop_graphs(g_train, k=args.k_hop, device=device)
    # g_khop is None when k==1 (precompute_khop_graphs returns g_train itself,
    # but we pass None to the model so sfg.py uses 1-hop paths)
    khop_arg = None if args.k_hop == 1 else g_khop

    # ── OMN setup ─────────────────────────────────────────────────────────────
    omn_str = args.omn.strip().lower()
    if omn_str == "none" or args.psi == 0.0 or args.no_struct:
        omn = None
    else:
        omn = sorted(omn_str.split(";"))

    # ── features ──────────────────────────────────────────────────────────────
    input_dim = None
    if feats is not None:
        input_dim = feats.shape[1]
        feats = feats.to(device)

    # ── move graphs to device ─────────────────────────────────────────────────
    if not args.no_gnn and g_supra is not None:
        g_supra = g_supra.to(device)
    for glist in [g_train, g_train_pos, g_test_pos, g_test_neg, g_val_pos, g_val_neg]:
        for li in range(n_layers):
            glist[li] = glist[li].to(device)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(args, n_layers, input_dim).to(device)
    embed_dim = args.hidden_dim * args.n_heads if args.heads_mode == "concat" else args.hidden_dim
    predictor = build_predictors(args, n_layers, embed_dim, device)

    all_params = list(model.parameters())
    if predictor:
        for pr in predictor:
            all_params += list(pr.parameters())
    optimiser = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)

    ck_name = f"{args.dataset}_k{args.k_hop}_rate{removal_rate:.2f}_run{run_idx}"
    stopper  = EarlyStopping(patience=args.patience, maximize=True,
                              model_name=ck_name, model_dir=args.ck_dir)

    # ── training loop ──────────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in tqdm(range(1, args.epochs + 1),
                      desc=f"  k={args.k_hop} rate={removal_rate:.0%} run={run_idx+1}",
                      leave=False):
        model.train()
        if predictor:
            for pr in predictor:
                pr.train()
        optimiser.zero_grad()

        g_train_neg = sample_train_negatives(g_train_pos, n_layers, n, device)

        if args.no_gnn or args.no_struct:
            pos_s, _, _ = model(g_supra, g_train, p, g_train_pos, feats, predictor, omn, khop_arg)
            neg_s, _, _ = model(g_supra, g_train, p, g_train_neg,  feats, predictor, omn, khop_arg)
            loss = compute_bce(torch.vstack(pos_s), torch.vstack(neg_s))
        else:
            pos_s, pos_st, pos_gnn = model(g_supra, g_train, p, g_train_pos, feats, predictor, omn, khop_arg)
            neg_s, neg_st, neg_gnn = model(g_supra, g_train, p, g_train_neg,  feats, predictor, omn, khop_arg)
            loss = (compute_bce(torch.vstack(pos_s),   torch.vstack(neg_s))   +
                    compute_bce(torch.vstack(pos_st),  torch.vstack(neg_st))  +
                    compute_bce(torch.vstack(pos_gnn), torch.vstack(neg_gnn)))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if predictor:
            for pr in predictor:
                torch.nn.utils.clip_grad_norm_(pr.parameters(), 1.0)
        optimiser.step()

        if epoch > 5:
            val_res = evaluate(model, predictor, g_supra, g_train, p, feats,
                               g_val_pos, g_val_neg, omn, device, khop_arg)
            stopper.step(val_res["auc"], model, epoch)

            if stopper.counter == 0 and predictor:
                Path(args.ck_dir).mkdir(parents=True, exist_ok=True)
                for li, pr in enumerate(predictor):
                    torch.save(pr.state_dict(),
                               os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin"))

            if stopper.early_stop:
                break

    print(f"  Training time: {time.time()-t0:.1f}s")

    # ── test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(stopper.save_dir))
    model = model.to(device).eval()
    if predictor:
        for li, pr in enumerate(predictor):
            ck = os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin")
            if os.path.isfile(ck):
                pr.load_state_dict(torch.load(ck))
            pr.to(device).eval()

    test_res = evaluate(model, predictor, g_supra, g_train, p, feats,
                        g_test_pos, g_test_neg, omn, device, khop_arg)

    # ── cleanup ───────────────────────────────────────────────────────────────
    stopper.remove_checkpoint()
    if predictor:
        for li in range(n_layers):
            ck = os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin")
            if os.path.isfile(ck):
                os.remove(ck)

    return test_res


# ══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = set_params()
    cuda   = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print(f"Device: {device}  |  k_hop: {args.k_hop}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ck_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    for rate in sorted(args.removal_rates):
        print(f"\n{'='*60}\nRemoval rate: {rate:.0%}  (k={args.k_hop})\n{'='*60}")
        auc_list, ap_list = [], []

        for run in range(args.n_runs):
            res = run_trial(args, removal_rate=rate, run_idx=run, device=device)
            auc_list.append(res["auc"] * 100)
            ap_list.append(res["ap"]  * 100)
            print(f"  Run {run+1}/{args.n_runs} → AUC={res['auc']*100:.2f}  AP={res['ap']*100:.2f}")

        row = {
            "k_hop":       args.k_hop,
            "removal_rate": rate,
            "auc_mean":    float(np.mean(auc_list)),
            "auc_std":     float(np.std(auc_list)),
            "ap_mean":     float(np.mean(ap_list)),
            "ap_std":      float(np.std(ap_list)),
            "auc_runs":    auc_list,
            "ap_runs":     ap_list,
        }
        all_results.append(row)
        print(f"\n  AUC: {row['auc_mean']:.2f} ± {row['auc_std']:.2f}  "
              f"AP: {row['ap_mean']:.2f} ± {row['ap_std']:.2f}")

    # ── save ──────────────────────────────────────────────────────────────────
    suffix   = f"k{args.k_hop}"
    csv_path  = os.path.join(args.save_dir, f"cns_removal_results_{suffix}.csv")
    json_path = os.path.join(args.save_dir, f"cns_removal_results_{suffix}.json")

    fields = ["k_hop","removal_rate","auc_mean","auc_std","ap_mean","ap_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in fields})

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults → {csv_path}")
    print(f"\n{'Removal':>9}  {'AUC mean±std':>18}  {'AP mean±std':>18}")
    print("─" * 52)
    for r in all_results:
        print(f"  {r['removal_rate']:>6.0%}   "
              f"{r['auc_mean']:>7.2f} ± {r['auc_std']:<6.2f}   "
              f"{r['ap_mean']:>7.2f} ± {r['ap_std']:<6.2f}")


if __name__ == "__main__":
    main()
