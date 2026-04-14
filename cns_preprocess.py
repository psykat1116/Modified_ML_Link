"""
cns_preprocess.py
=================
Converts the Copenhagen Networks Study (CNS) raw CSV files into the
ML-Link edge-list format (net.edges + meta_info.txt).

Expected raw-data layout
------------------------
<src_dir>/
  calls.csv                  — columns: source, target, timestamp, duration
  sms.csv                    — columns: source, target, timestamp, [extra]
  bluetooth_proximity.csv    — columns: source, target, timestamp, rssi
  fb_friends.csv             — columns: source, target  [optional timestamp col]

Layer assignment (1-indexed in net.edges, 0-indexed in code)
-------------------------------------------------------------
  Layer 1 → calls              (context)
  Layer 2 → sms                (context)
  Layer 3 → bluetooth_proximity(context)
  Layer 4 → fb_friends         ← TARGET for prediction

Output (written to <dst_dir>/cns/)
-----------------------------------
  net.edges      — space-separated: layer_id  src  dst
  meta_info.txt  — space-separated header+values: N  E  L
  (l_info.txt is NOT written; load.py's build_p creates full pairings automatically)

Usage
-----
  python cns_preprocess.py --src data/cns_raw --dst data/nets
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# ── layer definitions ──────────────────────────────────────────────────────────
LAYERS = [
    ("calls",               "calls.csv",                1),
    ("sms",                 "sms.csv",                  2),
    ("bluetooth_proximity", "bluetooth_proximity.csv",  3),
    ("fb_friends",          "fb_friends.csv",           4),   # TARGET
]
TARGET_LAYER_FILE_ID = 4   # 1-indexed
EDGE_TYPE = "UNDIRECTED"   # treat all layers as undirected


# ── helpers ────────────────────────────────────────────────────────────────────

def read_edges(path: str) -> np.ndarray:
    """
    Read a CSV edge file and return a (E, 2) int32 array of (src, dst) pairs.

    Handles:
      - optional leading comment line starting with '#'
      - 2, 3, or 4 columns (only first two used)
      - Windows-style line endings
    """
    with open(path, "r") as fh:
        first = fh.readline()

    skip = 1 if first.strip().startswith("#") else 0

    df = pd.read_csv(
        path,
        header=None,
        skiprows=skip,
        sep=r"[\s,]+",
        engine="python",
        usecols=[0, 1],
        names=["src", "dst"],
        dtype={"src": np.int64, "dst": np.int64},
    )

    # Drop any rows with NaN
    df = df.dropna()

    edges = df[["src", "dst"]].to_numpy(dtype=np.int64)
    return edges


def aggregate_edges(edges: np.ndarray, directed: bool = False) -> np.ndarray:
    """
    Collapse repeated interactions into unique (src, dst) pairs.
    For undirected graphs also removes (u, v) / (v, u) duplicates and
    self-loops.
    """
    # Remove self-loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]

    if not directed:
        # Canonical order: smaller ID first
        edges = np.sort(edges, axis=1)

    # Unique pairs
    edges = np.unique(edges, axis=0)
    return edges


def build_remap(all_node_ids: np.ndarray) -> dict:
    """
    Map arbitrary integer node IDs → contiguous 0-based indices.
    Returns a dict {original_id: new_index}.
    """
    unique_ids = np.unique(all_node_ids)
    return {int(uid): idx for idx, uid in enumerate(unique_ids)}


# ── main ───────────────────────────────────────────────────────────────────────

def preprocess(src_dir: str, dst_dir: str, dataset_name: str = "cns"):
    out_dir = os.path.join(dst_dir, dataset_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    directed = EDGE_TYPE == "DIRECTED"

    # ── 1. Read every layer ────────────────────────────────────────────────────
    layer_edges_raw = {}   # layer_name -> (E, 2) int64 array

    for name, fname, layer_id in LAYERS:
        fpath = os.path.join(src_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Layer file not found: {fpath}\n"
                f"Please place the raw CSV files in '{src_dir}/'."
            )
        raw = read_edges(fpath)
        agg = aggregate_edges(raw, directed=directed)
        layer_edges_raw[name] = agg
        print(f"  [{layer_id}] {name:25s}: {len(raw):>10,} interactions "
              f"→ {len(agg):>8,} unique {'edges' if not directed else 'arcs'} "
              f"| node range [{agg.min()}, {agg.max()}]")

    # ── 2. Build unified node ID space ─────────────────────────────────────────
    all_node_ids = np.concatenate(
        [e.ravel() for e in layer_edges_raw.values()]
    )
    remap = build_remap(all_node_ids)
    N = len(remap)
    print(f"\n  Total unique entities: {N}")

    # ── 3. Remap node IDs and write net.edges ──────────────────────────────────
    edges_out_path = os.path.join(out_dir, "net.edges")
    with open(edges_out_path, "w") as fout:
        for name, fname, layer_id in LAYERS:
            edges = layer_edges_raw[name]
            for src_orig, dst_orig in edges:
                src = remap[int(src_orig)]
                dst = remap[int(dst_orig)]
                fout.write(f"{layer_id} {src} {dst}\n")

    total_edges = sum(len(e) for e in layer_edges_raw.values())
    print(f"  Written {total_edges:,} edge records to: {edges_out_path}")

    # ── 4. Write meta_info.txt ─────────────────────────────────────────────────
    meta_path = os.path.join(out_dir, "meta_info.txt")
    n_layers = len(LAYERS)
    with open(meta_path, "w") as fout:
        fout.write("N E L\n")
        fout.write(f"{N} {EDGE_TYPE} {n_layers}\n")
    print(f"  Written meta_info.txt  (N={N}, E={EDGE_TYPE}, L={n_layers})")

    # ── 5. Per-layer summary ───────────────────────────────────────────────────
    print("\nLayer summary:")
    print(f"  {'Layer':<5} {'Name':<25} {'Edges':>8}  Role")
    print("  " + "-" * 55)
    for name, fname, layer_id in LAYERS:
        n_e = len(layer_edges_raw[name])
        role = "TARGET (prediction)" if layer_id == TARGET_LAYER_FILE_ID else "context"
        print(f"  {layer_id:<5} {name:<25} {n_e:>8,}  {role}")

    print(f"\nPreprocessing complete → {out_dir}")
    return out_dir


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess CNS dataset for ML-Link.")
    parser.add_argument("--src", type=str, default="data/cns_raw",
                        help="Directory containing raw layer CSV files.")
    parser.add_argument("--dst", type=str, default="data/nets",
                        help="Root directory where the preprocessed dataset folder is written.")
    parser.add_argument("--name", type=str, default="cns",
                        help="Dataset subfolder name (default: cns).")
    args = parser.parse_args()

    print(f"Preprocessing CNS dataset")
    print(f"  src : {args.src}")
    print(f"  dst : {os.path.join(args.dst, args.name)}\n")
    preprocess(args.src, args.dst, args.name)


if __name__ == "__main__":
    main()
