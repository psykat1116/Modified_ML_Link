"""
cns_experiment.py
=================
Removal-rate sweep experiment on the Copenhagen Networks Study (CNS) dataset.

What this script does
---------------------
For each removal_rate in {10%, 20%, …, 90%}:
  Run 5 independent trials (different random seeds).
  Each trial:
    1. Randomly remove `removal_rate` fraction of fb_friends edges → test set.
    2. Train ML-Link using:
         - calls, sms, bluetooth_proximity as CONTEXT (intact, no loss).
         - fb_friends train edges for structural features + loss.
    3. Evaluate on the removed fb_friends edges (AUC, AP).
  Report mean ± std across the 5 trials.

Key design decisions
--------------------
- Context layers contribute structural node features and GNN embeddings
  but have EMPTY g_train_pos / g_test_pos  →  zero loss / zero metric on them.
- Loss is computed on fb_friends ONLY (by passing empty g_edges for context).
- Negative sampling during training uses global_uniform_negative_sampling
  against the fb_friends TRAIN graph (to avoid sampling known edges).
- Early stopping monitors validation AUC on fb_friends.
- Results are saved to <save_dir>/cns_removal_results.csv.

Usage
-----
  python cns_experiment.py [options]

Main options
  --dataset       cns               # dataset name (subfolder of --prep_dir)
  --prep_dir      data/prep_nets    # folder with preprocessed data
  --epochs        100
  --n_runs        5                 # trials per removal rate
  --removal_rates 0.1 0.2 ... 0.9  # can be overridden on the command line
  --gpu           -1                # GPU index; -1 = CPU
  --save_dir      artifacts/
"""

import os
import time
import argparse
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl.sampling import global_uniform_negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm

# ── project imports ────────────────────────────────────────────────────────────
from models.main_m import Mm
from models.link_predictor import MLPPredictor, LinkPredictor
from input_data.cns_load import prepare_cns_data, TARGET_LAYER
from utils.optimization import EarlyStopping
from utils.util import init_seed
import utils.const as C


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def set_params():
    parser = argparse.ArgumentParser(
        description="CNS removal-rate sweep experiment."
    )
    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--dataset",  type=str, default="cns")
    parser.add_argument("--prep_dir", type=str, default="./data/nets",
                        help="Directory that contains the preprocessed dataset folder.")
    # ── experiment ────────────────────────────────────────────────────────────
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of independent trials per removal rate.")
    parser.add_argument("--removal_rates", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Fraction(s) of fb_friends edges to remove as test set.")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Base random seed; run i uses seed base_seed + i.")
    # ── model architecture ────────────────────────────────────────────────────
    parser.add_argument("--edge_dim",   type=int,   default=8)
    parser.add_argument("--node_dim",   type=int,   default=128)
    parser.add_argument("--phi_dim",    type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_hidden", type=int,   default=2)
    parser.add_argument("--n_heads",    type=int,   default=1)
    parser.add_argument("--heads_mode", type=str,   default="concat")
    parser.add_argument("--predictor",  type=str,   default="mlp")
    parser.add_argument("--omn",        type=str,   default="oan;maan",
                        help="Overlapping multilayer neighborhood types (oan, maan, oan;maan, none).")
    parser.add_argument("--psi",        type=float, default=0.5)
    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--attn_dropout", type=float, default=0.7)
    parser.add_argument("--patience",     type=int,   default=30,
                        help="Early-stopping patience (epochs).")
    parser.add_argument("--no_gnn",    action="store_true")
    parser.add_argument("--no_struct", action="store_true")
    # ── infrastructure ────────────────────────────────────────────────────────
    parser.add_argument("--gpu",      type=int,  default=-1)
    parser.add_argument("--save_dir", type=str,  default="./artifacts/")
    parser.add_argument("--ck_dir",   type=str,  default="checkpoint")

    args, _ = parser.parse_known_args()
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Model / predictor construction
# ══════════════════════════════════════════════════════════════════════════════

def build_model(args, n_layers: int, input_dim: int | None) -> Mm:
    return Mm(
        n_layers=n_layers,
        dropout=args.dropout,
        no_struct=args.no_struct,
        no_gnn=args.no_gnn,
        psi=args.psi,
        edge_dim=args.edge_dim,
        node_dim=args.node_dim,
        phi_dim=args.phi_dim,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        heads=args.n_heads,
        attn_dropout=args.attn_dropout,
        residual=True,
        aggregation=args.heads_mode,
        activation=F.elu,
        eps=1e-8,
        f_dropout=0.7,
    )


def build_predictors(args, n_layers: int, dim: int, device: torch.device):
    if args.no_gnn:
        return None
    op = args.predictor.lower()
    if op == "mlp":
        preds = [MLPPredictor(dim, dropout=args.dropout).to(device)
                 for _ in range(n_layers)]
    else:
        preds = [LinkPredictor(op).to(device) for _ in range(n_layers)]
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Loss helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_bce(pos: torch.Tensor, neg: torch.Tensor, eps: float = 1e-8):
    """Binary cross-entropy for link-existence scores."""
    loss  = -torch.log(pos + eps).mean()
    loss += -torch.log(1 - neg + eps).mean()
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# Negative sampling  (target layer only)
# ══════════════════════════════════════════════════════════════════════════════

def sample_train_negatives(g_train_pos, n_layers: int, n: int, device) -> list:
    """
    Sample training negatives for the TARGET layer only.
    Context layers receive empty graphs (no loss computed on them).
    """
    negs = []
    for li in range(n_layers):
        if li != TARGET_LAYER or g_train_pos[li].number_of_edges() == 0:
            negs.append(dgl.graph(([], []), num_nodes=n).to(device))
        else:
            g = g_train_pos[li]
            n_edges = g.number_of_edges()
            u, v = global_uniform_negative_sampling(
                g, num_samples=n_edges, exclude_self_loops=True, replace=False
            )
            neg_g = dgl.graph((u, v), num_nodes=n).to(device)
            negs.append(neg_g)
    return negs


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation  (fb_friends test / val only)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, predictor, g_supra, g_train, p, feats,
             g_pos, g_neg, inter, device) -> dict:
    """
    Compute AUC and AP on a single pos/neg graph pair (fb_friends only).
    Returns {'auc': float, 'ap': float}.
    """
    model.eval()
    if predictor:
        for pr in predictor:
            pr.eval()

    with torch.no_grad():
        pos_score, _, _ = model(g_supra, g_train, p, g_pos, feats, predictor, inter)
        neg_score, _, _ = model(g_supra, g_train, p, g_neg, feats, predictor, inter)

    pos_score = torch.vstack(pos_score).detach().cpu().squeeze(-1)
    neg_score = torch.vstack(neg_score).detach().cpu().squeeze(-1)

    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = np.concatenate([
        np.ones(pos_score.shape[0]),
        np.zeros(neg_score.shape[0]),
    ])

    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap":  float(average_precision_score(labels, scores)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single-trial training function
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(args, removal_rate: float, run_idx: int, device: torch.device):
    """
    Trains and evaluates ML-Link for one (removal_rate, run) combination.
    Returns {'auc': float, 'ap': float}.
    """
    seed = args.base_seed + run_idx
    init_seed(seed)

    # ── load data ──────────────────────────────────────────────────────────────
    (g_supra, g_train, g_train_pos,
     g_test_pos, g_test_neg,
     g_val_pos, g_val_neg,
     feats, n_info) = prepare_cns_data(
        removal_rate=removal_rate,
        run_seed=seed,
        dataset=args.dataset,
        src_dir=args.prep_dir,
    )

    n, n_layers, directed, mpx, _, p = n_info

    # ── configure OMN contexts ─────────────────────────────────────────────────
    omn_str = args.omn.strip().lower()
    if omn_str == "none" or args.psi == 0.0 or args.no_struct:
        omn = None
    else:
        omn = sorted(omn_str.split(";"))
        assert all(x in (C.OAN, C.MAAN) for x in omn), \
            f"Unknown OMN type(s): {omn}"

    # ── feature dimension ──────────────────────────────────────────────────────
    input_dim = None
    if feats is not None:
        input_dim = feats.shape[1]
        feats = feats.to(device)

    # ── move graphs to device ─────────────────────────────────────────────────
    if not args.no_gnn and g_supra is not None:
        g_supra = g_supra.to(device)

    graph_lists = [g_train, g_train_pos,
                   g_test_pos, g_test_neg,
                   g_val_pos,  g_val_neg]
    for glist in graph_lists:
        for li in range(n_layers):
            glist[li] = glist[li].to(device)

    # ── build model + predictors ───────────────────────────────────────────────
    model = build_model(args, n_layers, input_dim).to(device)

    embed_dim = args.hidden_dim
    if args.heads_mode == "concat":
        embed_dim *= args.n_heads

    predictor = build_predictors(args, n_layers, embed_dim, device)

    # ── optimiser ─────────────────────────────────────────────────────────────
    all_params = list(model.parameters())
    if predictor:
        for pr in predictor:
            all_params += list(pr.parameters())

    optimiser = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)

    ck_name = f"{args.dataset}_rate{removal_rate:.2f}_run{run_idx}"
    stopper = EarlyStopping(
        patience=args.patience,
        maximize=True,
        model_name=ck_name,
        model_dir=args.ck_dir,
    )

    lambda_all = 1.0   # weight on combined score loss
    lambda_st  = 1.0   # weight on struct-only loss
    lambda_gnn = 1.0   # weight on gnn-only loss

    # ── training loop ──────────────────────────────────────────────────────────
    t0 = time.time()
    for epoch in tqdm(
        range(1, args.epochs + 1),
        desc=f"  rate={removal_rate:.0%} run={run_idx+1}",
        leave=False,
    ):
        model.train()
        if predictor:
            for pr in predictor:
                pr.train()

        optimiser.zero_grad()

        # Negatives for fb_friends training edges
        g_train_neg = sample_train_negatives(g_train_pos, n_layers, n, device)

        if args.no_gnn or args.no_struct:
            # Only one component active → single loss
            pos_s, _, _ = model(g_supra, g_train, p, g_train_pos, feats, predictor, omn)
            neg_s, _, _ = model(g_supra, g_train, p, g_train_neg,  feats, predictor, omn)
            # Only TARGET_LAYER contributes (context layers filtered out as None)
            pos_s = torch.vstack(pos_s)
            neg_s = torch.vstack(neg_s)
            loss  = compute_bce(pos_s, neg_s)
        else:
            # Both components → three losses
            pos_s, pos_st, pos_gnn = model(g_supra, g_train, p, g_train_pos, feats, predictor, omn)
            neg_s, neg_st, neg_gnn = model(g_supra, g_train, p, g_train_neg,  feats, predictor, omn)

            pos_s   = torch.vstack(pos_s)
            neg_s   = torch.vstack(neg_s)
            pos_st  = torch.vstack(pos_st)
            neg_st  = torch.vstack(neg_st)
            pos_gnn = torch.vstack(pos_gnn)
            neg_gnn = torch.vstack(neg_gnn)

            loss = (lambda_all * compute_bce(pos_s,   neg_s)   +
                    lambda_st  * compute_bce(pos_st,  neg_st)  +
                    lambda_gnn * compute_bce(pos_gnn, neg_gnn))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if predictor:
            for pr in predictor:
                torch.nn.utils.clip_grad_norm_(pr.parameters(), 1.0)
        optimiser.step()

        # ── validation + early stopping (every epoch after warm-up) ───────────
        if epoch > 5:
            val_res = evaluate(
                model, predictor, g_supra, g_train, p, feats,
                g_val_pos, g_val_neg, omn, device,
            )
            stopper.step(val_res["auc"], model, epoch)

            # Save predictor checkpoints alongside model checkpoint
            if stopper.counter == 0 and predictor:
                Path(args.ck_dir).mkdir(parents=True, exist_ok=True)
                for li, pr in enumerate(predictor):
                    torch.save(
                        pr.state_dict(),
                        os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin"),
                    )

            if stopper.early_stop:
                print(f"  Early stop at epoch {epoch} "
                      f"(val AUC={val_res['auc']:.4f})")
                break

    print(f"  Training time: {time.time() - t0:.1f}s")

    # ── load best checkpoint → evaluate on test ────────────────────────────────
    model.load_state_dict(torch.load(stopper.save_dir))
    model = model.to(device).eval()

    if predictor:
        for li, pr in enumerate(predictor):
            ck = os.path.join(args.ck_dir, f"pred{li}_{ck_name}.bin")
            if os.path.isfile(ck):
                pr.load_state_dict(torch.load(ck))
            pr = pr.to(device).eval()

    test_res = evaluate(
        model, predictor, g_supra, g_train, p, feats,
        g_test_pos, g_test_neg, omn, device,
    )

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

    # ── device ────────────────────────────────────────────────────────────────
    cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f"cuda:{args.gpu}" if cuda else "cpu")
    print(f"Device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ck_dir).mkdir(parents=True, exist_ok=True)

    removal_rates = sorted(args.removal_rates)
    print(f"\nCNS fb_friends link-prediction experiment")
    print(f"  Removal rates : {removal_rates}")
    print(f"  Runs per rate : {args.n_runs}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  OMN contexts  : {args.omn}")
    print()

    # ── sweep ─────────────────────────────────────────────────────────────────
    all_results = []   # list of dicts for CSV export

    for rate in removal_rates:
        print(f"{'='*60}")
        print(f"Removal rate: {rate:.0%}")
        print(f"{'='*60}")

        auc_list, ap_list = [], []

        for run in range(args.n_runs):
            res = run_trial(args, removal_rate=rate, run_idx=run, device=device)
            auc_list.append(res["auc"] * 100)
            ap_list.append(res["ap"]  * 100)
            print(f"  Run {run+1}/{args.n_runs} → "
                  f"AUC={res['auc']*100:.2f}  AP={res['ap']*100:.2f}")

        auc_mean, auc_std = float(np.mean(auc_list)), float(np.std(auc_list))
        ap_mean,  ap_std  = float(np.mean(ap_list)),  float(np.std(ap_list))

        print(f"\n  ── Summary  rate={rate:.0%} ──────────────────────────────")
        print(f"     AUC : {auc_mean:.2f} ± {auc_std:.2f}")
        print(f"     AP  : {ap_mean:.2f}  ± {ap_std:.2f}\n")

        row = {
            "removal_rate": rate,
            "auc_mean":     auc_mean,
            "auc_std":      auc_std,
            "ap_mean":      ap_mean,
            "ap_std":       ap_std,
            "auc_runs":     auc_list,
            "ap_runs":      ap_list,
        }
        all_results.append(row)

    # ── save results ──────────────────────────────────────────────────────────
    csv_path  = os.path.join(args.save_dir, "cns_removal_results.csv")
    json_path = os.path.join(args.save_dir, "cns_removal_results.json")

    csv_fields = ["removal_rate",
                  "auc_mean", "auc_std",
                  "ap_mean",  "ap_std"]
    with open(csv_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in csv_fields})

    with open(json_path, "w") as fout:
        json.dump(all_results, fout, indent=2)

    print(f"\nResults saved to:\n  {csv_path}\n  {json_path}")

    # ── print final table ─────────────────────────────────────────────────────
    print(f"\n{'Removal':>9}  {'AUC (mean±std)':>18}  {'AP (mean±std)':>18}")
    print("─" * 52)
    for r in all_results:
        print(f"  {r['removal_rate']:>6.0%}   "
              f"{r['auc_mean']:>7.2f} ± {r['auc_std']:<6.2f}   "
              f"{r['ap_mean']:>7.2f} ± {r['ap_std']:<6.2f}")


if __name__ == "__main__":
    main()
