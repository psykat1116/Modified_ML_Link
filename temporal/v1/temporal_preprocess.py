"""
temporal_preprocess.py  (v3)
=============================
Changes from v2:
  - T = 7 windows (4-day windows instead of daily)
  - Denser features per window → fewer zero vectors

Usage:
    python temporal_preprocess.py --src data/cns_raw --dst data/cns_temporal
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

T          = 7
WINDOW_SEC = 4 * 86400    # 4 days per window
T_START    = 0
T_END      = T * WINDOW_SEC


def _assign_windows(df, ts_col):
    df = df[(df[ts_col] >= T_START) & (df[ts_col] < T_END)].copy()
    df['_win'] = np.clip((df[ts_col] // WINDOW_SEC).astype(int), 0, T - 1)
    return df


def _apply_remap(df, remap):
    df['src'] = df['src'].map(remap)
    df['dst'] = df['dst'].map(remap)
    df = df.dropna(subset=['src', 'dst'])
    df[['src', 'dst']] = df[['src', 'dst']].astype(int)
    return df


def _build_window_matrices(df, n, value_col, sym=True):
    mats = []
    for t in range(T):
        sub = df[df._win == t]
        if len(sub) == 0:
            mats.append(sp.csr_matrix((n, n), dtype=np.float32))
            continue
        src = sub['src'].values.astype(np.int32)
        dst = sub['dst'].values.astype(np.int32)
        val = sub[value_col].values.astype(np.float32)
        if sym:
            rows = np.concatenate([src, dst])
            cols = np.concatenate([dst, src])
            vals = np.concatenate([val, val])
        else:
            rows, cols, vals = src, dst, val
        mat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        mats.append(mat)
    return mats


def _compute_degree(count_mats):
    """Row-sum per window → node activity level."""
    return [np.asarray(mat.sum(axis=1)).flatten().astype(np.float32)
            for mat in count_mats]


def build_snapshots(src_dir: str) -> dict:
    # ── Unified remap ──────────────────────────────────────────────────────────
    all_ids = []
    for fname in ['calls.csv', 'sms.csv', 'bluetooth_proximity.csv', 'fb_friends.csv']:
        fpath = os.path.join(src_dir, fname)
        if os.path.isfile(fpath):
            df_tmp = pd.read_csv(fpath, comment='#', header=None, usecols=[0, 1])
            all_ids.append(df_tmp.values.ravel())
    unique_ids = np.unique(np.concatenate(all_ids))
    remap = {int(uid): idx for idx, uid in enumerate(unique_ids)}
    n = len(remap)
    print(f"  Total unique entities after remap: {n}")
    print(f"  T={T} windows of {WINDOW_SEC//86400} days each")

    snap = {'n_nodes': n, 'T': T, 'window_size': WINDOW_SEC}

    # ── Calls ──────────────────────────────────────────────────────────────────
    calls_path = os.path.join(src_dir, 'calls.csv')
    if os.path.isfile(calls_path):
        df = pd.read_csv(calls_path, comment='#', header=None,
                         names=['src', 'dst', 'timestamp', 'duration'])
        df['duration'] = df['duration'].clip(lower=0)
        df = _apply_remap(df, remap)
        df = _assign_windows(df, 'timestamp')
        df['one'] = 1.0
        count_mats = _build_window_matrices(df, n, 'one')
        snap['calls'] = {
            'count':    count_mats,
            'duration': _build_window_matrices(df, n, 'duration'),
            'degree':   _compute_degree(count_mats),
        }
        print(f"  Calls:      {len(df):>8,} interactions over {T} windows")
    else:
        print(f"  [warn] calls.csv not found")
        empty = [sp.csr_matrix((n, n), dtype=np.float32)] * T
        snap['calls'] = {'count': empty, 'duration': empty,
                         'degree': [np.zeros(n, dtype=np.float32)] * T}

    # ── SMS ────────────────────────────────────────────────────────────────────
    sms_path = os.path.join(src_dir, 'sms.csv')
    if os.path.isfile(sms_path):
        df = pd.read_csv(sms_path, comment='#', header=None,
                         names=['src', 'dst', 'timestamp'])
        df = _apply_remap(df, remap)
        df = _assign_windows(df, 'timestamp')
        df['one'] = 1.0
        count_mats = _build_window_matrices(df, n, 'one')
        snap['sms'] = {
            'count':  count_mats,
            'degree': _compute_degree(count_mats),
        }
        print(f"  SMS:        {len(df):>8,} interactions over {T} windows")
    else:
        print(f"  [warn] sms.csv not found")
        empty = [sp.csr_matrix((n, n), dtype=np.float32)] * T
        snap['sms'] = {'count': empty,
                       'degree': [np.zeros(n, dtype=np.float32)] * T}

    # ── Bluetooth ──────────────────────────────────────────────────────────────
    bt_path = os.path.join(src_dir, 'bluetooth_proximity.csv')
    if os.path.isfile(bt_path):
        chunks = []
        for chunk in pd.read_csv(bt_path, comment='#', header=None,
                                  names=['src', 'dst', 'timestamp', 'rssi'],
                                  chunksize=200_000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df = _apply_remap(df, remap)
        df = _assign_windows(df, 'timestamp')
        df['one'] = 1.0
        count_mats = _build_window_matrices(df, n, 'one')
        snap['bluetooth'] = {
            'count':    count_mats,
            'rssi_sum': _build_window_matrices(df, n, 'rssi'),
            'degree':   _compute_degree(count_mats),
        }
        print(f"  Bluetooth:  {len(df):>8,} interactions over {T} windows")
    else:
        print(f"  [warn] bluetooth_proximity.csv not found")
        empty = [sp.csr_matrix((n, n), dtype=np.float32)] * T
        snap['bluetooth'] = {'count': empty, 'rssi_sum': empty,
                              'degree': [np.zeros(n, dtype=np.float32)] * T}

    # ── Normalisation stats ────────────────────────────────────────────────────
    def _safe_max(mats):
        vals = [m.max() for m in mats if m.nnz > 0]
        return float(max(vals)) if vals else 1.0

    snap['stats'] = {
        'max_calls_count':    _safe_max(snap['calls']['count']),
        'max_calls_duration': _safe_max(snap['calls']['duration']),
        'max_sms_count':      _safe_max(snap['sms']['count']),
        'max_bt_count':       _safe_max(snap['bluetooth']['count']),
    }
    print(f"\n  Normalisation stats: {snap['stats']}")
    return snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/cns_raw')
    parser.add_argument('--dst', type=str, default='data/cns_temporal')
    args = parser.parse_args()

    Path(args.dst).mkdir(parents=True, exist_ok=True)
    print(f"Building temporal snapshots (T=7) from: {args.src}")
    snap = build_snapshots(args.src)

    out_path = os.path.join(args.dst, 'temporal_snapshots.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(snap, f, protocol=4)
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    main()
