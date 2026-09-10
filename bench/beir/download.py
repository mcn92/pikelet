#!/usr/bin/env python3.11
"""Download an official BEIR dataset (corpus/queries/qrels) into bench/beir/cache/.

Usage:
    python3.11 bench/beir/download.py scifact
    python3.11 bench/beir/download.py nfcorpus fiqa arguana trec-covid

Datasets are fetched from the official BEIR release bucket and cached under
bench/beir/cache/<dataset>/ (gitignored — see bench/beir/README.md). Re-running
is a no-op if the dataset is already present.
"""
import sys
import os

from beir import util

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def download(dataset: str) -> str:
    url = BEIR_URL.format(dataset)
    path = util.download_and_unzip(url, CACHE_DIR)
    print(f"{dataset}: {path}")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for ds in sys.argv[1:]:
        download(ds)
