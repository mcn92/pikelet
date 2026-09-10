#!/usr/bin/env python3.11
"""Score a raw run (query.mjs output) against official BEIR qrels.

Uses pytrec_eval — the same scoring library the official BEIR benchmark
scripts use — so numbers are directly comparable to published BEIR results.
No LLM judging: qrels are the ground truth.

Usage:
    python3.11 bench/beir/score.py scifact A
    -> reads bench/beir/work/scifact/run-A.json
    -> writes bench/beir/results/scifact-A.json
"""
import sys
import os
import json
import pytrec_eval

HERE = os.path.dirname(__file__)


def load_qrels(dataset: str) -> dict:
    path = os.path.join(HERE, "cache", dataset, "qrels", "test.tsv")
    qrels = {}
    with open(path) as f:
        header = f.readline()  # "query-id\tcorpus-id\tscore"
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, docid, score = parts
            qrels.setdefault(qid, {})[docid] = int(score)
    return qrels


def load_run(dataset: str, configuration: str) -> dict:
    path = os.path.join(HERE, "work", dataset, f"run-{configuration}.json")
    with open(path) as f:
        raw = json.load(f)
    # raw["results"]: { queryId: [ {beirId, score}, ... ], ... }
    run = {}
    for qid, hits in raw["results"].items():
        run[qid] = {h["beirId"]: float(h["score"]) for h in hits}
    return run, raw


MEASURES = {"ndcg_cut.10", "recall.10", "recall.100"}


def score(dataset: str, configuration: str) -> dict:
    qrels = load_qrels(dataset)
    run, raw = load_run(dataset, configuration)

    # Only score queries that have both qrels and results (should be identical
    # sets if query.mjs ran every test-split query — mismatches are reported).
    qrel_ids = set(qrels.keys())
    run_ids = set(run.keys())
    missing_from_run = qrel_ids - run_ids
    extra_in_run = run_ids - qrel_ids
    if missing_from_run:
        print(f"WARNING: {len(missing_from_run)} qrels queries missing from run", file=sys.stderr)
    if extra_in_run:
        print(f"WARNING: {len(extra_in_run)} run queries not in qrels test split (ignored)", file=sys.stderr)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, MEASURES)
    per_query = evaluator.evaluate(run)

    n = len(per_query)
    agg = {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "recall_at_100": 0.0}
    for qid, m in per_query.items():
        agg["ndcg_at_10"] += m["ndcg_cut_10"]
        agg["recall_at_10"] += m["recall_10"]
        agg["recall_at_100"] += m["recall_100"]
    for key in agg:
        agg[key] = agg[key] / n if n else 0.0

    result = {
        "benchmark": "BEIR",
        "dataset": dataset,
        "configuration": configuration,
        "num_queries_scored": n,
        "num_queries_missing": len(missing_from_run),
        "metrics": agg,
        "system": raw.get("system", {}),
        "config_snapshot": raw.get("config_snapshot"),
        "pikelet_commit": raw.get("pikelet_commit"),
        "pack_hash": raw.get("pack_hash"),
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    dataset, configuration = sys.argv[1], sys.argv[2]
    result = score(dataset, configuration)

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}-{configuration}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["metrics"], indent=2))
    print(f"-> {out_path}")
