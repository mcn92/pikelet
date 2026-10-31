"""Could the decoder fetch wte rows on demand instead of shipping them?
Replays record_candidates.mjs's candidate sets against a shard cache: the
exact rescoring of a step needs the u5 wte rows (576 bytes each) of its K
candidates (plus, on the long text, the input token's row; a generated
token's row was a candidate the step before). Rows live in shards of a
fixed size, stored in token-id order or by how often a row was a candidate
on separate calibration stories; the client keeps an LRU cache of shards,
started empty (cold) or filled with the hottest shards (seeded). Every
missing shard of a step is fetched in one parallel round trip, so a step
costs one round trip if anything is missing.

Per step it reports round trips (the share of steps that wait on the
network), shards requested and KiB fetched; prompts are reported apart
(all their rows are known before any compute, one batched fetch)."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
C = HERE / 'candidates'
V, ROW = 50257, 576
SHARDS_KIB = [4, 16, 64, 256]
BUDGETS_MB = [1, 4, 8, 16]
index = json.loads((C / 'index.json').read_text())
long_ids = json.loads((HERE / 'eval_ref.json').read_text())['long_ids']
counts = np.bincount(np.fromfile(C / 'calib-k256.bin', dtype=np.int32), minlength=V)
ORDERS = {'token id': np.arange(V), 'calib frequency': np.argsort(np.argsort(-counts, kind='stable'))}  # row id -> stored position


def load(K):
    flat = np.fromfile(C / f'test-k{K}.bin', dtype=np.int32).reshape(-1, K)
    events, i = [], 0  # (kind, rows needed); kind 'prompt' or the segment's kind
    for seg in index[str(K)]:
        if seg['prompt']:
            events.append(('prompt', np.array(seg['prompt'])))
        for s in range(seg['steps']):
            rows = flat[i + s]
            if seg['kind'] == 'long':
                rows = np.append(rows, long_ids[s])
            events.append((seg['kind'], rows))
        i += seg['steps']
    assert i == len(flat)
    return events


def replay(events, pos, rows_per_shard, cap, seeded):
    n = (V + rows_per_shard - 1) // rows_per_shard
    last = np.full(n, -np.inf)
    cached = np.zeros(n, bool)
    if seeded:  # hottest shards by calibration candidate count, hottest most recent
        heat = np.bincount(pos // rows_per_shard, weights=counts, minlength=n)
        hot = np.argsort(-heat, kind='stable')[:cap]
        cached[hot] = True
        last[hot] = -np.arange(len(hot), dtype=float) - 1
    out = {}
    for t, (kind, rows) in enumerate(events):
        s = np.unique(pos[rows] // rows_per_shard)
        miss = s[~cached[s]]
        cached[miss] = True
        last[s] = t
        over = cached.sum() - cap
        if over > 0:
            idx = np.flatnonzero(cached)
            cached[idx[np.argpartition(last[idx], over - 1)[:over]]] = False
        o = out.setdefault(kind, [])
        o.append(len(miss))
    return {k: np.array(v) for k, v in out.items()}


def fmt(m, rows_per_shard):
    return f'{(m > 0).mean():5.1%} {m.mean():6.1f} {np.percentile(m, 95):5.0f} {m.mean() * rows_per_shard * ROW / 1024:7.1f}'


print(f'wte at u5: {V * ROW / 2**20:.1f} MB; calibration steps {counts.sum() // 256}, rows ever a candidate there {np.count_nonzero(counts)}')
for K in [128, 256]:
    events = load(K)
    gen = [r for k, r in events if k != 'prompt']
    touched = np.unique(np.concatenate(gen))
    print(f'\n=== K {K}: {len(gen)} steps ({", ".join(f"{k} {sum(1 for e in events if e[0] == k)}" for k in ["long", "greedy", "sampled", "prompt"])}); '
          f'distinct rows touched {len(touched)} ({len(touched) * ROW / 2**20:.2f} MB), never a calibration candidate {np.count_nonzero(counts[touched] == 0)}')
    # ideal row-granular hot set: share of needed rows inside the top-N calibration rows
    hot_rank = ORDERS['calib frequency']
    all_rows = np.concatenate(gen)
    print('  ideal hot set (rows by calibration frequency, no shards): ' + ', '.join(
        f'{mb} MB covers {(hot_rank[all_rows] < mb * 2**20 // ROW).mean():.2%} of row uses, {np.mean([np.all(hot_rank[r] < mb * 2**20 // ROW) for r in gen]):.1%} of steps' for mb in BUDGETS_MB))
    print('  columns per segment kind: steps waiting on network | mean shards fetched | p95 shards | mean KiB fetched')
    for order, pos in ORDERS.items():
        for kib in SHARDS_KIB:
            rps = kib * 1024 // ROW
            for mb in BUDGETS_MB:
                cap = mb * 2**20 // (rps * ROW)
                for seeded in [False, True]:
                    r = replay(events, pos, rps, cap, seeded)
                    gen_m = np.concatenate([r[k] for k in ['long', 'greedy', 'sampled']])
                    print(f'  {order:15s} {kib:3d} KiB {mb:2d} MB {"seeded" if seeded else "cold  "} | all gen {fmt(gen_m, rps)} | sampled {fmt(r["sampled"], rps)} | prompts {fmt(r["prompt"], rps)}', flush=True)
