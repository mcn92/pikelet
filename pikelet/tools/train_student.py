#!/usr/bin/env python3
"""Distill a tiny zero-runtime-dependency query encoder for the Worker demo.

The teacher is used only by this offline script. Runtime inference uses the
quantized PSTU artifact through student-embedder.mjs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import struct
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


MAGIC = b"PSTU"
VERSION = 1
HASH_SEED = 2166136261
DEFAULT_TEACHER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOKEN_RE = re.compile(r"[a-z0-9]+")
ABSTENTION_FEATURES = ["d0", "margin", "pre_norm", "known_word", "known_char", "hidden_probe"]
ABSTENTION_FAR_TARGET = 0.02
ABSTENTION_CALIBRATION_FAR_TARGET = ABSTENTION_FAR_TARGET
ABSTENTION_HARD = 0.05
ABSTENTION_PRENORM_FLOOR = 0.4
ABSTENTION_MIN_FEATURES = 3
ABSTENTION_MIN_AUC = 0.97
ABSTENTION_MAX_TEST_FAR = ABSTENTION_FAR_TARGET + 0.01
ABSTENTION_MIN_COLLISION_CATCH = 0.60
ABSTENTION_MIN_SHUFFLED_CATCH = 0.60
ABSTENTION_MIN_GENERAL_CATCH = 0.80
ABSTENTION_MIN_NONSENSE_NONE = 1.00
ABSTENTION_MAX_NOISE_MEDIAN = ABSTENTION_HARD
GOLDEN_ABSTENTION_FIXTURES_PATH = Path(__file__).with_name("fixtures") / "abstention-golden.json"

CURATED_HELDOUT = [
    "How does an edge isolate recover a saved vector index?",
    "Can I reclaim space after removing lots of records?",
    "How can searches be restricted to one tenant?",
    "What is the RAM cost of storing vectors as eight bit values?",
    "Can a snapshot be opened without repeating its construction settings?",
    "Does every index share the same WebAssembly memory?",
    "What happens when several requests arrive during a cold start?",
    "How do I change search accuracy for only one request?",
]

GENERAL_NEGATIVE_QUERIES = [
    "banana pancake recipe",
    "best restaurants near me",
    "weather tomorrow morning",
    "who won the world cup",
    "cheap flights to tokyo",
    "how to grow tomatoes indoors",
    "symptoms of seasonal allergies",
    "mortgage interest rate forecast",
    "how many calories in an avocado",
    "movie times this weekend",
    "easy chocolate chip cookies",
    "history of the roman empire",
    "convert dollars to euros",
    "how to train for a marathon",
    "best noise cancelling headphones",
    "what is the capital of finland",
    "home workout without equipment",
    "how to file taxes online",
    "coffee brewing temperature",
    "beginner guitar chords",
    "nba playoff schedule",
    "how to remove red wine stains",
    "javascript date formatting",
    "python flask authentication",
    "postgres vacuum analyze",
    "kubernetes ingress controller",
    "react server components",
    "rust borrow checker",
    "linux iptables rules",
    "docker compose networking",
]

FOREIGN_MINI_CORPUS = [
    "SQLite query planner",
    "PostgreSQL write ahead log",
    "React hydration errors",
    "Kubernetes ingress controller",
    "Redis eviction policy",
    "Linux cgroups memory limits",
    "TLS certificate renewal",
    "OAuth redirect URI",
    "Python asyncio event loop",
    "TypeScript generic constraints",
]

CASUAL_POSITIVE_TEMPLATES = [
    "why is {topic} slow",
    "{topic} keeps failing",
    "my {topic} broke",
    "{topic} coming back after restart",
    "how do i fix {topic}",
    "what happened to {topic}",
]

CURATED_CASUAL_POSITIVES = [
    "why is search slow on my phone",
    "deleted stuff coming back after restart",
    "my filtered search broke",
    "snapshots are not loading after restart",
    "why is compaction not removing deleted items",
    "worker restore keeps failing",
]

CURATED_TERSE_POSITIVES = [
    "how does compaction work",
    "compaction after deletes",
    "deleted vectors after restart",
    "filtered search",
    "how does filtered search work",
    "worker snapshot restore",
    "r2 persistence",
    "ef search tuning",
    "quantized memory tradeoffs",
    "export restore snapshot",
]

CURATED_TYPO_POSITIVES = [
    "fitlered saerch",
    "filtred search",
    "filtered saerch",
    "flitered search",
    "filtered serach",
    "fitler search",
    "how does fitlered saerch work",
    "why is filtred search slow",
    "my flitered search broke",
    "fitlered search docs",
    "filtered serach docs",
    "serach filter docs",
    "serach filters",
    "snapshott restore",
    "compactionn deletes",
]

NONSENSE_NEGATIVE_QUERIES = [
    "xkcd zqjv wvvv",
    "qzxv blorpt nym",
    "zzqx vrrp lmnop",
    "flarnq zibble wug",
    "mxxq jjjt vvvv",
    "plorbnar xqzv krrp",
    "qqq zzz xxx",
    "n0nword z9z9 blerp",
    "asdfjkl qwertyui zxcvbn",
    "wvvv zqjv xkcd",
    "trrkl mzzp qqqx",
    "vbnmz lkjasd qpoiu",
    "zqxjv wrrrb nnnm",
    "bloopz frand qix",
    "flarnq zibblar wuggo",
    "fralnq zibble wogg",
    "flermq zabble wugx",
    "zibble wug flarnish",
    "frand zibble wuglet",
    "zxzxzx qqqqq vvvvv",
    "xj9q v2zz n7mn",
    "plugh xyzzy thud",
    "snorfle quux baz",
    "zzzz yyyy xxxx",
    "loremq ipsuz dolorx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--teacher-revision", default=DEFAULT_TEACHER_REVISION)
    parser.add_argument("--buckets", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.008)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--skip-abstention", action="store_true")
    parser.add_argument("--golden-fixtures", type=Path)
    return parser.parse_args()


def fnv1a(text: str, seed: int = HASH_SEED) -> int:
    value = seed & 0xFFFFFFFF
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def extract_feature_records(text: str, buckets: int, max_features: int) -> list[tuple[int, str]]:
    tokens = TOKEN_RE.findall(str(text).lower())
    features: list[tuple[int, str]] = []

    def push(feature: str, family: str) -> None:
        if len(features) < max_features:
            features.append((fnv1a(feature) % buckets, family))

    for token_index, raw_token in enumerate(tokens):
        if len(features) >= max_features:
            break
        token = raw_token[:48]
        push(f"w:{token}", "word")
        padded = f"^{token}$"
        for width in range(3, 6):
            for start in range(0, len(padded) - width + 1):
                push(f"c{width}:{padded[start:start + width]}", "char")
                if len(features) >= max_features:
                    break
            if len(features) >= max_features:
                break
        if token_index > 0:
            push(f"b:{tokens[token_index - 1][:48]}:{token}", "word")
    return features


def extract_features(text: str, buckets: int, max_features: int) -> list[int]:
    return [bucket for bucket, _family in extract_feature_records(text, buckets, max_features)]


def clean_title(value: str) -> str:
    return re.sub(r"\s*[/|>]\s*", " ", str(value or "")).strip()


def first_sentence(value: str, max_words: int = 28) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    return " ".join(sentence.split()[:max_words])


def keywords(value: str, limit: int = 10) -> str:
    stop = {
        "about", "after", "also", "because", "before", "between", "from",
        "have", "into", "only", "pancake", "that", "their", "then", "there",
        "these", "they", "this", "through", "using", "when", "where", "which",
        "while", "with", "your",
    }
    result: list[str] = []
    for token in TOKEN_RE.findall(str(value).lower()):
        if len(token) < 3 or token in stop or token in result:
            continue
        result.append(token)
        if len(result) == limit:
            break
    return " ".join(result)


def dropout_words(text: str, randomizer: random.Random, rate: float = 0.22) -> str:
    words = str(text).split()
    kept = [word for word in words if randomizer.random() >= rate]
    if len(kept) < min(3, len(words)):
        kept = words[: min(len(words), 6)]
    return " ".join(kept)


@dataclass(frozen=True)
class QueryExample:
    text: str
    origin: int | None
    family: str


def make_examples(corpus: list[dict], seed: int) -> tuple[list[QueryExample], list[QueryExample]]:
    randomizer = random.Random(seed)
    train: list[QueryExample] = []
    heldout: list[QueryExample] = []
    seen_train: set[str] = set()
    seen_heldout: set[str] = set()

    def add(target: list[QueryExample], seen: set[str], text: str, origin: int, family: str) -> None:
        normalized = re.sub(r"\s+", " ", text).strip()
        key = normalized.lower()
        if len(normalized) >= 3 and key not in seen:
            target.append(QueryExample(normalized, origin, family))
            seen.add(key)

    for row, chunk in enumerate(corpus):
        title = clean_title(chunk.get("title") or chunk.get("docTitle") or f"section {row}")
        sentence = first_sentence(chunk.get("text") or chunk.get("preview") or "")
        terms = keywords(f"{title} {sentence}")

        training_forms = [
            (title, "title"),
            (f"what is {title}", "what"),
            (f"how does {title} work", "how"),
            (f"explain {title}", "explain"),
            (f"documentation for {title}", "docs"),
            (sentence, "sentence"),
            (terms, "keywords"),
            (dropout_words(f"{title} {sentence}", randomizer), "dropout"),
        ]
        for text, family in training_forms:
            add(train, seen_train, text, row, family)

        heldout_forms = [
            (f"where can i learn about {title}", "where"),
            (f"tell me about {title}", "tell"),
            (f"i need help with {title}", "help"),
        ]
        for text, family in heldout_forms:
            add(heldout, seen_heldout, text, row, family)

    for query in CURATED_HELDOUT:
        key = query.lower()
        if key not in seen_heldout:
            heldout.append(QueryExample(query, None, "curated"))
            seen_heldout.add(key)
    return train, heldout


class TeacherEncoder:
    def __init__(self, name: str, revision: str):
        self.name = name
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int, max_length: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(functional.normalize(pooled, dim=1).cpu())
        return torch.cat(outputs, dim=0)


class StudentEncoder(nn.Module):
    def __init__(self, buckets: int, hidden: int, output: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(buckets, hidden, mode="mean")
        self.projection = nn.Linear(hidden, output)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.035)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, flat_features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        pooled = torch.tanh(self.embedding(flat_features, offsets))
        return functional.normalize(self.projection(pooled), dim=1)


def feature_batch(feature_rows: list[list[int]], indexes: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    flat: list[int] = []
    offsets: list[int] = []
    for index in indexes:
        offsets.append(len(flat))
        row = feature_rows[index]
        flat.extend(row if row else [0])
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.max(np.abs(values), axis=1)
    scales = np.where(maximum > 0, maximum / 127.0, 1.0).astype("<f4")
    quantized = np.clip(np.rint(values / scales[:, None]), -127, 127).astype(np.int8)
    return scales, quantized


def quantized_forward(
    feature_rows: list[list[int]],
    embedding_scales: np.ndarray,
    embedding_quantized: np.ndarray,
    projection_scales: np.ndarray,
    projection_quantized: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    vectors, _pre_norms = quantized_forward_with_norms(
        feature_rows,
        embedding_scales,
        embedding_quantized,
        projection_scales,
        projection_quantized,
        bias,
    )
    return vectors


def quantized_forward_with_norms(
    feature_rows: list[list[int]],
    embedding_scales: np.ndarray,
    embedding_quantized: np.ndarray,
    projection_scales: np.ndarray,
    projection_quantized: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    embedding = embedding_quantized.astype(np.float32) * embedding_scales[:, None]
    projection = projection_quantized.astype(np.float32) * projection_scales[:, None]
    results = np.zeros((len(feature_rows), projection.shape[0]), dtype=np.float32)
    pre_norms = np.zeros(len(feature_rows), dtype=np.float32)
    for row_index, features in enumerate(feature_rows):
        if not features:
            continue
        hidden = np.tanh(embedding[features].mean(axis=0))
        output = projection @ hidden + bias
        norm = np.linalg.norm(output)
        pre_norms[row_index] = norm
        if norm > 0:
            output /= norm
        results[row_index] = output
    return results, pre_norms


def quantized_hidden_rows(
    feature_rows: list[list[int]],
    embedding_scales: np.ndarray,
    embedding_quantized: np.ndarray,
) -> np.ndarray:
    embedding = embedding_quantized.astype(np.float32) * embedding_scales[:, None]
    hidden_rows = np.zeros((len(feature_rows), embedding.shape[1]), dtype=np.float32)
    for row_index, features in enumerate(feature_rows):
        if features:
            hidden_rows[row_index] = np.tanh(embedding[features].mean(axis=0))
    return hidden_rows


def topk_overlap(student: np.ndarray, teacher: np.ndarray, docs: np.ndarray, k: int) -> float:
    student_top = np.argpartition(-(student @ docs.T), min(k, docs.shape[0]) - 1, axis=1)[:, :k]
    teacher_top = np.argpartition(-(teacher @ docs.T), min(k, docs.shape[0]) - 1, axis=1)[:, :k]
    total = 0.0
    for left, right in zip(student_top, teacher_top):
        total += len(set(left.tolist()).intersection(right.tolist())) / k
    return total / len(student)


def origin_recall(vectors: np.ndarray, docs: np.ndarray, examples: list[QueryExample], k: int) -> float | None:
    eligible = [(index, example.origin) for index, example in enumerate(examples) if example.origin is not None]
    if not eligible:
        return None
    top = np.argpartition(-(vectors @ docs.T), min(k, docs.shape[0]) - 1, axis=1)[:, :k]
    hits = sum(1 for index, origin in eligible if origin in top[index])
    return hits / len(eligible)


def corpus_text(row: dict) -> str:
    return f"{row.get('docTitle', '')}. {clean_title(row.get('title', ''))}. {row.get('text', '')}"


def build_activation_tables(corpus: list[dict], buckets: int, max_features: int) -> dict:
    word_df: Counter[int] = Counter()
    char_buckets: set[int] = set()
    for row in corpus:
        records = extract_feature_records(corpus_text(row), buckets, max_features)
        row_word: set[int] = set()
        for bucket, family in records:
            if family == "word":
                row_word.add(bucket)
            elif family == "char":
                char_buckets.add(bucket)
        word_df.update(row_word)

    idfs = {
        bucket: math.log(1 + len(corpus) / df)
        for bucket, df in word_df.items()
        if df > 0
    }
    max_idf = max(idfs.values(), default=1.0)
    word_buckets = [
        [bucket, max(0, min(255, int(round(idf / max_idf * 255))))]
        for bucket, idf in sorted(idfs.items())
    ]
    return {
        "wordBuckets": word_buckets,
        "charBuckets": sorted(char_buckets),
        "_wordWeights": {bucket: weight for bucket, weight in word_buckets},
        "_maxIdfQ": 255,
        "_charSet": char_buckets,
    }


def known_fractions(records: list[tuple[int, str]], activation: dict) -> dict:
    word_total = 0.0
    word_known = 0.0
    char_total = 0
    char_known = 0
    word_weights = activation["_wordWeights"]
    max_idf = activation["_maxIdfQ"]
    char_set = activation["_charSet"]
    for bucket, family in records:
        if family == "word":
            word_total += 1
            if bucket in word_weights:
                word_known += 1
        elif family == "char":
            char_total += 1
            if bucket in char_set:
                char_known += 1
    return {
        "known_word": word_known / word_total if word_total > 0 else 0.0,
        "known_char": char_known / char_total if char_total > 0 else 0.0,
        "n_feats": len(records),
    }


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60, 60)
    return 1.0 / (1.0 + np.exp(-values))


def scalar_sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def logistic_scores(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid_np(features @ weights + bias)


def train_logistic(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales = np.where(scales > 1e-6, scales, 1.0)
    normalized = (features - means) / scales
    weights = rng.normal(0, 0.02, size=features.shape[1]).astype(np.float64)
    bias = float(math.log((labels.mean() + 1e-6) / (1 - labels.mean() + 1e-6)))
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))
    sample_weights = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    learning_rate = 0.08
    l2 = 0.01
    for _step in range(2500):
        predicted = logistic_scores(normalized, weights, bias)
        error = (predicted - labels) * sample_weights
        weights -= learning_rate * ((normalized.T @ error) / len(labels) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    raw_weights = weights / scales
    raw_bias = bias - float(np.sum(weights * means / scales))
    return raw_weights.astype(float), raw_bias


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


def label_from_score(score: float, signals: dict, weak_threshold: float) -> str:
    if (
        score < ABSTENTION_HARD
        or signals["n_feats"] < ABSTENTION_MIN_FEATURES
        or signals["pre_norm"] < ABSTENTION_PRENORM_FLOOR
    ):
        return "none"
    return "strong" if score >= weak_threshold else "weak"


@dataclass(frozen=True)
class AbstentionExample:
    text: str
    label: int
    family: str
    split: str
    origin: int | None = None


def split_for_family_instance(family: str, index: int, seed: int) -> str:
    value = fnv1a(f"{seed}:{family}:{index}") % 10
    if value < 6:
        return "train"
    if value < 8:
        return "validation"
    return "test"


def make_negative_examples(
    corpus: list[dict],
    positives: list[QueryExample],
    activation: dict,
    buckets: int,
    max_features: int,
    seed: int,
) -> list[AbstentionExample]:
    randomizer = random.Random(seed + 17)
    negatives: list[AbstentionExample] = []

    def add(family: str, index: int, text: str) -> None:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) >= 3:
            negatives.append(AbstentionExample(normalized, 0, family, split_for_family_instance(family, index, seed)))

    for index, query in enumerate(GENERAL_NEGATIVE_QUERIES):
        add("general", index, query)
    for index, query in enumerate(NONSENSE_NEGATIVE_QUERIES):
        add("nonsense", index, query)

    foreign_templates = [
        "how does {title} work",
        "documentation for {title}",
        "debug {title}",
        "what is {title}",
    ]
    instance = 0
    for title in FOREIGN_MINI_CORPUS:
        for template in foreign_templates:
            add("foreign", instance, template.format(title=title))
            instance += 1

    general_tokens = [token for query in GENERAL_NEGATIVE_QUERIES for token in TOKEN_RE.findall(query)]
    protected_shuffle_tokens = {
        "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "me",
        "of", "the", "to", "what", "when", "where", "why", "with",
    }
    shuffled_candidates = [
        example for example in positives
        if example.family not in {"sentence", "dropout", "keywords"}
        and 3 <= len(TOKEN_RE.findall(example.text.lower())) <= 14
    ]
    shuffled_examples = shuffled_candidates[: min(len(shuffled_candidates), max(80, len(corpus) // 2))]
    for index, example in enumerate(shuffled_examples):
        tokens = TOKEN_RE.findall(example.text.lower())
        if not tokens:
            continue
        shuffled = tokens[:]
        randomizer.shuffle(shuffled)
        shuffled = shuffled[:12]
        replacement_count = max(1, len(shuffled) // 2)
        content_indexes = [
            token_index for token_index, token in enumerate(shuffled)
            if token not in protected_shuffle_tokens
        ]
        if len(content_indexes) >= replacement_count:
            replace_indexes = set(randomizer.sample(content_indexes, replacement_count))
        else:
            replace_indexes = set(content_indexes)
            remaining = [token_index for token_index in range(len(shuffled)) if token_index not in replace_indexes]
            replace_indexes.update(randomizer.sample(remaining, replacement_count - len(replace_indexes)))
        replaced = [
            randomizer.choice(general_tokens) if token_index in replace_indexes else token
            for token_index, token in enumerate(shuffled)
        ]
        add("shuffled", index, " ".join(replaced))

    high_idf_buckets = {
        bucket
        for bucket, idf_q in activation["wordBuckets"]
        if idf_q >= 230
    }
    corpus_tokens: list[str] = []
    seen_tokens: set[str] = set()
    for row in corpus:
        for token in TOKEN_RE.findall(corpus_text(row).lower()):
            if len(token) < 4 or token in seen_tokens:
                continue
            records = extract_feature_records(token, buckets, max_features)
            if any(bucket in high_idf_buckets and family == "word" for bucket, family in records):
                corpus_tokens.append(token)
                seen_tokens.add(token)
    if not corpus_tokens:
        corpus_tokens = ["pancake", "worker", "snapshot", "compact"]
    collision_bases = [
        "banana pancake recipe",
        "best breakfast recipe",
        "cheap vacation ideas",
        "how to roast vegetables",
        "family dinner menu",
        "weather forecast today",
        "sports highlights tonight",
        "coffee shop playlist",
        "home mortgage calculator",
        "weekend hiking checklist",
    ]
    for index in range(max(40, len(corpus) // 2)):
        base = randomizer.choice(collision_bases)
        token = randomizer.choice(corpus_tokens)
        words = base.split()
        words.insert(randomizer.randrange(0, len(words) + 1), token)
        add("collision", index, " ".join(words))
    return negatives


def typo_variant(text: str, randomizer: random.Random) -> str:
    words = str(text).split()
    candidate_indexes = [index for index, word in enumerate(words) if len(TOKEN_RE.findall(word)) > 0 and len(word) >= 5]
    if not candidate_indexes:
        return text
    edits = min(2, len(candidate_indexes))
    for index in randomizer.sample(candidate_indexes, edits):
        word = words[index]
        if len(word) < 5:
            continue
        position = randomizer.randrange(1, len(word) - 1)
        mode = randomizer.choice(["swap", "delete"])
        if mode == "swap" and position + 1 < len(word):
            chars = list(word)
            chars[position], chars[position + 1] = chars[position + 1], chars[position]
            words[index] = "".join(chars)
        else:
            words[index] = word[:position] + word[position + 1:]
    return " ".join(words)


def topic_from_example(example: QueryExample, corpus: list[dict]) -> str:
    if example.origin is not None and 0 <= example.origin < len(corpus):
        row = corpus[example.origin]
        terms = keywords(f"{clean_title(row.get('title', ''))} {row.get('text', '')}", limit=4)
        if terms:
            return terms
    terms = keywords(example.text, limit=4)
    return terms or clean_title(example.text)


def augmented_positive_examples(
    examples: list[QueryExample],
    corpus: list[dict],
    split: str,
    seed: int,
) -> list[AbstentionExample]:
    randomizer = random.Random(f"{seed}:{split}:positive-augment")
    positives = [
        AbstentionExample(example.text, 1, f"positive:{example.family}", split, example.origin)
        for example in examples
    ]
    if split == "test":
        return positives

    if split == "train":
        for index, example in enumerate(examples):
            if index % 2 == 0:
                positives.append(
                    AbstentionExample(
                        typo_variant(example.text, randomizer),
                        1,
                        f"positive:{example.family}:typo",
                        split,
                        example.origin,
                    )
                )

    if split == "train":
        for index, example in enumerate(examples):
            topic = topic_from_example(example, corpus)
            template = CASUAL_POSITIVE_TEMPLATES[index % len(CASUAL_POSITIVE_TEMPLATES)]
            positives.append(
                AbstentionExample(
                    template.format(topic=topic),
                    1,
                    "positive:casual",
                    split,
                    example.origin,
                )
            )
        for text in CURATED_TYPO_POSITIVES:
            positives.append(AbstentionExample(text, 1, "positive:typo:curated", split, None))

    if split == "test":
        for index, text in enumerate(CURATED_CASUAL_POSITIVES):
            positives.append(AbstentionExample(text, 1, "positive:casual:curated", split, None))
    for index, text in enumerate(CURATED_TERSE_POSITIVES):
        if split == "train" or index % 2 == 0:
            positives.append(AbstentionExample(text, 1, "positive:terse:curated", split, None))

    return positives


def compute_signal_rows(
    examples: list[AbstentionExample],
    docs: np.ndarray,
    activation: dict,
    exported: dict,
    buckets: int,
    max_features: int,
    hidden_probe: tuple[np.ndarray, float] | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    feature_records = [
        extract_feature_records(example.text, buckets, max_features)
        for example in examples
    ]
    feature_rows = [[bucket for bucket, _family in records] for records in feature_records]
    vectors, pre_norms = quantized_forward_with_norms(
        feature_rows,
        exported["embedding_scales"],
        exported["embedding_quantized"],
        exported["projection_scales"],
        exported["projection_quantized"],
        exported["bias"],
    )
    hidden_rows = quantized_hidden_rows(
        feature_rows,
        exported["embedding_scales"],
        exported["embedding_quantized"],
    )
    hidden_scores = np.zeros(len(examples), dtype=np.float64)
    if hidden_probe is not None:
        hidden_weights, hidden_bias = hidden_probe
        hidden_scores = logistic_scores(hidden_rows.astype(np.float64), hidden_weights, hidden_bias)
    if len(examples) == 0:
        return np.zeros((0, len(ABSTENTION_FEATURES)), dtype=np.float64), [], hidden_rows

    similarities = vectors @ docs.T
    top_count = min(5, docs.shape[0])
    top_indexes = np.argpartition(-similarities, top_count - 1, axis=1)[:, :top_count]
    rows: list[dict] = []
    feature_matrix = np.zeros((len(examples), len(ABSTENTION_FEATURES)), dtype=np.float64)
    for row_index, example in enumerate(examples):
        ordered = top_indexes[row_index][np.argsort(-similarities[row_index, top_indexes[row_index]])]
        distances = [float(1.0 - similarities[row_index, doc_index]) for doc_index in ordered]
        d0 = distances[0] if distances else 1.0
        margin_index = min(4, len(distances) - 1)
        margin = distances[margin_index] - d0 if margin_index > 0 else 0.0
        known = known_fractions(feature_records[row_index], activation)
        signals = {
            "d0": d0,
            "margin": margin,
            "pre_norm": float(pre_norms[row_index]),
            "known_word": float(known["known_word"]),
            "known_char": float(known["known_char"]),
            "hidden_probe": float(hidden_scores[row_index]),
            "n_feats": int(known["n_feats"]),
        }
        feature_matrix[row_index] = [signals[name] for name in ABSTENTION_FEATURES]
        rows.append({
            "text": example.text,
            "family": example.family,
            "split": example.split,
            "expected": "in-domain" if example.label == 1 else "negative",
            "origin": example.origin,
            "signals": signals,
        })
    return feature_matrix, rows, hidden_rows


def abstention_split_examples(
    corpus: list[dict],
    train_examples: list[QueryExample],
    validation_examples: list[QueryExample],
    heldout_examples: list[QueryExample],
    negatives: list[AbstentionExample],
    seed: int,
) -> dict[str, list[AbstentionExample]]:
    splits = {
        "train": augmented_positive_examples(train_examples, corpus, "train", seed),
        "validation": augmented_positive_examples(validation_examples, corpus, "validation", seed),
        "test": augmented_positive_examples(heldout_examples, corpus, "test", seed),
    }
    for negative in negatives:
        splits[negative.split].append(negative)
    return splits


def choose_weak_threshold(validation_rows: list[dict], scores: np.ndarray) -> float:
    positive_scores = sorted(
        float(score)
        for row, score in zip(validation_rows, scores)
        if row["expected"] == "in-domain"
    )
    if not positive_scores:
        return 0.5
    allowed_misses = int(math.floor(len(positive_scores) * ABSTENTION_CALIBRATION_FAR_TARGET))
    index = min(allowed_misses, len(positive_scores) - 1)
    return positive_scores[index]


def summarize_abstention(
    rows: list[dict],
    labels: np.ndarray,
    scores: np.ndarray,
    weak_threshold: float,
) -> dict:
    negative_rows = [row for row in rows if row["expected"] == "negative"]
    negative_scores = [float(score) for row, score in zip(rows, scores) if row["expected"] == "negative"]
    positive_scores = [float(score) for row, score in zip(rows, scores) if row["expected"] == "in-domain"]
    false_abstain = sum(1 for score in positive_scores if score < weak_threshold)
    catch_by_family: dict[str, dict] = {}
    for family in sorted({row["family"] for row in negative_rows}):
        family_scores = [
            score
            for row, score in zip(rows, scores)
            if row["expected"] == "negative" and row["family"] == family
        ]
        caught = sum(1 for score in family_scores if score < weak_threshold)
        catch_by_family[family] = {
            "queries": len(family_scores),
            "catchRate": caught / len(family_scores) if family_scores else 0.0,
        }
    return {
        "auc": auc_score(labels, scores),
        "falseAbstainRate": false_abstain / len(positive_scores) if positive_scores else 0.0,
        "catchRate": catch_by_family,
    }


def load_golden_fixtures(path: Path | None) -> list[dict]:
    if path is None:
        return []
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("Golden abstention fixture must be a non-empty JSON array")
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError("Golden abstention fixture rows must be objects")
        if fixture.get("expected") not in {"strong", "weak", "none"}:
            raise ValueError(f"Invalid golden abstention label: {fixture.get('expected')!r}")
        if not isinstance(fixture.get("text"), str) or not fixture["text"].strip():
            raise ValueError("Golden abstention fixture rows require non-empty text")
    return fixtures


def golden_fixture_examples(fixtures: list[dict]) -> list[AbstentionExample]:
    examples: list[AbstentionExample] = []
    for fixture in fixtures:
        label = 1 if fixture["expected"] == "strong" else 0
        examples.append(AbstentionExample(fixture["text"], label, fixture["family"], "golden", None))
    return examples


def expected_label_pass(actual: str, expected: str) -> bool:
    if expected == "strong":
        return actual == "strong"
    if expected == "weak":
        return actual == "weak"
    if expected == "none":
        return actual == "none"
    return actual == expected


def train_abstention(
    corpus: list[dict],
    docs: np.ndarray,
    exported: dict,
    train_examples: list[QueryExample],
    validation_examples: list[QueryExample],
    heldout_examples: list[QueryExample],
    args: argparse.Namespace,
) -> tuple[dict, dict, list[dict]]:
    activation = build_activation_tables(corpus, args.buckets, args.max_features)
    all_positive_queries = train_examples + validation_examples + heldout_examples
    negatives = make_negative_examples(
        corpus,
        all_positive_queries,
        activation,
        args.buckets,
        args.max_features,
        args.seed,
    )
    split_examples = abstention_split_examples(
        corpus,
        train_examples,
        validation_examples,
        heldout_examples,
        negatives,
        args.seed,
    )
    matrices: dict[str, np.ndarray] = {}
    rows_by_split: dict[str, list[dict]] = {}
    labels_by_split: dict[str, np.ndarray] = {}
    hidden_by_split: dict[str, np.ndarray] = {}
    for split, examples in split_examples.items():
        matrix, rows, hidden_rows = compute_signal_rows(
            examples,
            docs,
            activation,
            exported,
            args.buckets,
            args.max_features,
        )
        matrices[split] = matrix
        rows_by_split[split] = rows
        hidden_by_split[split] = hidden_rows.astype(np.float64)
        labels_by_split[split] = np.array([example.label for example in examples], dtype=np.float64)

    hidden_weights, hidden_bias = train_logistic(hidden_by_split["train"], labels_by_split["train"], args.seed + 91)
    hidden_probe = (hidden_weights, hidden_bias)
    for split, examples in split_examples.items():
        matrix, rows, _hidden_rows = compute_signal_rows(
            examples,
            docs,
            activation,
            exported,
            args.buckets,
            args.max_features,
            hidden_probe,
        )
        matrices[split] = matrix
        rows_by_split[split] = rows

    weights, bias = train_logistic(matrices["train"], labels_by_split["train"], args.seed)
    validation_scores = logistic_scores(matrices["validation"], weights, bias)
    weak_threshold = choose_weak_threshold(rows_by_split["validation"], validation_scores)
    test_scores = logistic_scores(matrices["test"], weights, bias)
    train_scores = logistic_scores(matrices["train"], weights, bias)
    golden_fixtures = load_golden_fixtures(args.golden_fixtures)
    golden_matrix, golden_rows, _golden_hidden = compute_signal_rows(
        golden_fixture_examples(golden_fixtures),
        docs,
        activation,
        exported,
        args.buckets,
        args.max_features,
        hidden_probe,
    )
    golden_scores = logistic_scores(golden_matrix, weights, bias)

    for split, scores in [
        ("train", train_scores),
        ("validation", validation_scores),
        ("test", test_scores),
    ]:
        for row, score in zip(rows_by_split[split], scores):
            row["score"] = round(float(score), 6)
            row["label"] = label_from_score(float(score), row["signals"], weak_threshold)
    golden_results = []
    for fixture, row, score in zip(golden_fixtures, golden_rows, golden_scores):
        actual = label_from_score(float(score), row["signals"], weak_threshold)
        golden_results.append({
            "text": fixture["text"],
            "expected": fixture["expected"],
            "actual": actual,
            "score": round(float(score), 6),
            "signals": row["signals"],
        })

    validation_summary = summarize_abstention(
        rows_by_split["validation"],
        labels_by_split["validation"],
        validation_scores,
        weak_threshold,
    )
    test_summary = summarize_abstention(
        rows_by_split["test"],
        labels_by_split["test"],
        test_scores,
        weak_threshold,
    )
    train_summary = summarize_abstention(
        rows_by_split["train"],
        labels_by_split["train"],
        train_scores,
        weak_threshold,
    )
    calibrated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    abstention = {
        "version": 1,
        "features": ABSTENTION_FEATURES,
        "weights": [float(value) for value in weights.tolist()],
        "bias": float(bias),
        "thresholds": {
            "weak": float(weak_threshold),
            "hard": ABSTENTION_HARD,
            "preNormFloor": ABSTENTION_PRENORM_FLOOR,
            "minFeatures": ABSTENTION_MIN_FEATURES,
        },
        "wordBuckets": activation["wordBuckets"],
        "charBuckets": activation["charBuckets"],
        "farTarget": ABSTENTION_FAR_TARGET,
        "calibratedAt": calibrated_at,
        "hiddenProbe": {
            "weights": [float(value) for value in hidden_weights.tolist()],
            "bias": float(hidden_bias),
        },
        "evaluation": {
            "train": train_summary,
            "validation": validation_summary,
            "test": test_summary,
            "golden": golden_results,
        },
    }
    evaluation_summary = {
        "auc": test_summary["auc"],
        "falseAbstainRate": test_summary["falseAbstainRate"],
        "catchRate": test_summary["catchRate"],
        "validationAuc": validation_summary["auc"],
        "validationFalseAbstainRate": validation_summary["falseAbstainRate"],
        "farTarget": ABSTENTION_FAR_TARGET,
        "thresholds": abstention["thresholds"],
        "calibratedAt": calibrated_at,
        "trainQueries": len(rows_by_split["train"]),
        "validationQueries": len(rows_by_split["validation"]),
        "testQueries": len(rows_by_split["test"]),
        "golden": golden_results,
    }
    collision_catch = test_summary["catchRate"].get("collision", {}).get("catchRate", 0.0)
    shuffled_catch = test_summary["catchRate"].get("shuffled", {}).get("catchRate", 0.0)
    general_catch = test_summary["catchRate"].get("general", {}).get("catchRate", 0.0)
    test_nonsense_rows = [
        row for row in rows_by_split["test"]
        if row["expected"] == "negative" and row["family"] == "nonsense"
    ]
    nonsense_none = (
        sum(1 for row in test_nonsense_rows if row.get("label") == "none") / len(test_nonsense_rows)
        if test_nonsense_rows else 1.0
    )
    negative_scores = [
        float(score)
        for row, score in zip(rows_by_split["test"], test_scores)
        if row["expected"] == "negative" and row["family"] in {"general", "foreign", "collision", "nonsense"}
    ]
    negative_median = float(np.median(negative_scores)) if negative_scores else 1.0
    failures = []
    if test_summary["auc"] < ABSTENTION_MIN_AUC:
        failures.append(f"abstention AUC {test_summary['auc']:.3f} < {ABSTENTION_MIN_AUC:.2f}")
    if test_summary["falseAbstainRate"] > ABSTENTION_MAX_TEST_FAR:
        failures.append(
            f"abstention false-abstain {test_summary['falseAbstainRate']:.3f} > {ABSTENTION_MAX_TEST_FAR:.2f}"
        )
    if collision_catch < ABSTENTION_MIN_COLLISION_CATCH:
        failures.append(f"collision catch {collision_catch:.3f} < {ABSTENTION_MIN_COLLISION_CATCH:.2f}")
    if shuffled_catch < ABSTENTION_MIN_SHUFFLED_CATCH:
        failures.append(f"shuffled catch {shuffled_catch:.3f} < {ABSTENTION_MIN_SHUFFLED_CATCH:.2f}")
    if general_catch < ABSTENTION_MIN_GENERAL_CATCH:
        failures.append(f"general catch {general_catch:.3f} < {ABSTENTION_MIN_GENERAL_CATCH:.2f}")
    if nonsense_none < ABSTENTION_MIN_NONSENSE_NONE:
        failures.append(f"nonsense none-rate {nonsense_none:.3f} < {ABSTENTION_MIN_NONSENSE_NONE:.2f}")
    if negative_median > ABSTENTION_MAX_NOISE_MEDIAN:
        failures.append(f"median OOD score {negative_median:.3f} > hard threshold {ABSTENTION_MAX_NOISE_MEDIAN:.2f}")
    for result in golden_results:
        if not expected_label_pass(result["actual"], result["expected"]):
            failures.append(
                f"golden fixture {result['text']!r}: expected {result['expected']}, got {result['actual']} "
                f"at score {result['score']:.3f}"
            )
    if failures:
        raise RuntimeError("Abstention acceptance failed: " + "; ".join(failures))
    all_rows = rows_by_split["train"] + rows_by_split["validation"] + rows_by_split["test"]
    return abstention, evaluation_summary, all_rows


def export_model(path: Path, model: StudentEncoder, args: argparse.Namespace) -> dict:
    embedding = model.embedding.weight.detach().cpu().numpy().astype(np.float32)
    projection = model.projection.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.projection.bias.detach().cpu().numpy().astype("<f4")
    embedding_scales, embedding_quantized = quantize_rows(embedding)
    projection_scales, projection_quantized = quantize_rows(projection)

    payload = bytearray()
    payload.extend(struct.pack(
        "<4s7I",
        MAGIC,
        VERSION,
        args.buckets,
        args.hidden,
        projection.shape[0],
        HASH_SEED,
        args.max_features,
        0,
    ))
    payload.extend(embedding_scales.tobytes())
    payload.extend(embedding_quantized.tobytes())
    while len(payload) % 4:
        payload.append(0)
    payload.extend(projection_scales.tobytes())
    payload.extend(projection_quantized.tobytes())
    payload.extend(bias.tobytes())
    path.write_bytes(payload)
    return {
        "embedding_scales": embedding_scales,
        "embedding_quantized": embedding_quantized,
        "projection_scales": projection_scales,
        "projection_quantized": projection_quantized,
        "bias": bias,
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    args.out.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Corpus must be a non-empty JSON array")
    train_examples, evaluation_examples = make_examples(corpus, args.seed)
    validation_examples = [example for example in evaluation_examples if example.family == "where"]
    heldout_examples = [example for example in evaluation_examples if example.family != "where"]
    document_texts = [corpus_text(row) for row in corpus]

    print(
        f"[data] documents={len(corpus)} train_queries={len(train_examples)} "
        f"validation_queries={len(validation_examples)} heldout_queries={len(heldout_examples)}"
    )
    teacher = TeacherEncoder(args.teacher, args.teacher_revision)
    started = time.perf_counter()
    # Materialize ordinary tensors outside inference mode. PyTorch deliberately
    # prevents inference tensors from being saved for the student's backward pass.
    document_vectors = torch.from_numpy(
        teacher.encode(document_texts, batch_size=16, max_length=256).numpy().copy()
    )
    train_teacher = torch.from_numpy(
        teacher.encode([example.text for example in train_examples], batch_size=64, max_length=96).numpy().copy()
    )
    validation_teacher = torch.from_numpy(
        teacher.encode([example.text for example in validation_examples], batch_size=64, max_length=96).numpy().copy()
    )
    heldout_teacher = torch.from_numpy(
        teacher.encode([example.text for example in heldout_examples], batch_size=64, max_length=96).numpy().copy()
    )
    print(f"[teacher] encoded in {time.perf_counter() - started:.1f}s dim={document_vectors.shape[1]}")

    train_features = [
        extract_features(example.text, args.buckets, args.max_features)
        for example in train_examples
    ]
    validation_features = [
        extract_features(example.text, args.buckets, args.max_features)
        for example in validation_examples
    ]
    heldout_features = [
        extract_features(example.text, args.buckets, args.max_features)
        for example in heldout_examples
    ]
    student = StudentEncoder(args.buckets, args.hidden, document_vectors.shape[1])
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    temperature = 0.055
    best_state = copy.deepcopy(student.state_dict())
    best_epoch = 0
    best_validation = -math.inf
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train_examples), generator=generator).tolist()
        running = 0.0
        student.train()
        for start in range(0, len(permutation), args.batch_size):
            indexes = permutation[start:start + args.batch_size]
            flat, offsets = feature_batch(train_features, indexes)
            targets = train_teacher[indexes]
            predicted = student(flat, offsets)

            cosine_loss = (1 - (predicted * targets).sum(dim=1)).mean()
            with torch.no_grad():
                teacher_distribution = functional.softmax((targets @ document_vectors.T) / temperature, dim=1)
            student_distribution = functional.log_softmax((predicted @ document_vectors.T) / temperature, dim=1)
            ranking_loss = functional.kl_div(student_distribution, teacher_distribution, reduction="batchmean")
            loss = 0.55 * cosine_loss + 0.45 * ranking_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer.step()
            running += loss.item() * len(indexes)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            student.eval()
            with torch.inference_mode():
                flat, offsets = feature_batch(validation_features, range(len(validation_features)))
                prediction = student(flat, offsets)
                cosine = (prediction * validation_teacher).sum(dim=1).mean().item()
                agreement = topk_overlap(
                    prediction.numpy(),
                    validation_teacher.numpy(),
                    document_vectors.numpy(),
                    5,
                )
                validation_score = cosine + agreement
            if validation_score > best_validation + 1e-5:
                best_validation = validation_score
                best_epoch = epoch
                best_state = copy.deepcopy(student.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5 if epoch > 1 else 1
            print(
                f"[train] epoch={epoch:03d} loss={running / len(train_examples):.4f} "
                f"validation_cos={cosine:.4f} teacher_top5_overlap={agreement:.4f}"
            )
            if epochs_without_improvement >= 20:
                print(f"[train] early stop; restoring epoch {best_epoch}")
                break

    student.load_state_dict(best_state)

    model_path = args.out / "student-model.bin"
    exported = export_model(model_path, student, args)
    heldout_quantized = quantized_forward(
        heldout_features,
        exported["embedding_scales"],
        exported["embedding_quantized"],
        exported["projection_scales"],
        exported["projection_quantized"],
        exported["bias"],
    )
    teacher_heldout_np = heldout_teacher.numpy()
    documents_np = document_vectors.numpy().astype("<f4")
    if args.skip_abstention:
        abstention = None
        abstention_evaluation = None
        abstention_rows = []
        abstention_text = None
        abstention_bytes = 0
        abstention_sha256 = None
    else:
        abstention, abstention_evaluation, abstention_rows = train_abstention(
            corpus,
            documents_np,
            exported,
            train_examples,
            validation_examples,
            heldout_examples,
            args,
        )
        abstention_text = json.dumps(abstention, separators=(",", ":")) + "\n"
        abstention_payload = abstention_text.encode("utf-8")
        abstention_bytes = len(abstention_payload)
        abstention_sha256 = hashlib.sha256(abstention_payload).hexdigest()
    cosine_values = np.sum(heldout_quantized * teacher_heldout_np, axis=1)
    evaluation = {
        "heldoutQueries": len(heldout_examples),
        "validationQueries": len(validation_examples),
        "meanTeacherCosine": float(np.mean(cosine_values)),
        "p10TeacherCosine": float(np.quantile(cosine_values, 0.10)),
        "teacherTop1Agreement": topk_overlap(heldout_quantized, teacher_heldout_np, documents_np, 1),
        "teacherTop3Overlap": topk_overlap(heldout_quantized, teacher_heldout_np, documents_np, 3),
        "teacherTop5Overlap": topk_overlap(heldout_quantized, teacher_heldout_np, documents_np, 5),
        "studentOriginRecallAt5": origin_recall(heldout_quantized, documents_np, heldout_examples, 5),
        "teacherOriginRecallAt5": origin_recall(teacher_heldout_np, documents_np, heldout_examples, 5),
        "abstention": abstention_evaluation,
        "queries": [
            {
                "text": example.text,
                "family": example.family,
                "origin": example.origin,
                "teacherTop": int(np.argmax(teacher_heldout_np[index] @ documents_np.T)),
                "studentTop": int(np.argmax(heldout_quantized[index] @ documents_np.T)),
                "teacherCosine": float(cosine_values[index]),
            }
            for index, example in enumerate(heldout_examples)
        ],
        "abstentionQueries": abstention_rows,
    }

    (args.out / "docs-vectors.f32").write_bytes(documents_np.tobytes())
    (args.out / "student-evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    abstention_path = args.out / "student-abstention.json"
    if abstention_text is not None:
        abstention_path.write_text(
            abstention_text,
            encoding="utf-8",
        )
    manifest = {
        "format": "pikelet-distilled-student",
        "version": VERSION,
        "teacher": args.teacher,
        "teacherRevision": args.teacher_revision,
        "architecture": "hashed word/character n-grams -> mean pool -> tanh -> linear -> L2 normalize",
        "bucketCount": args.buckets,
        "hiddenDim": args.hidden,
        "outputDim": int(document_vectors.shape[1]),
        "hashSeed": HASH_SEED,
        "maxFeatures": args.max_features,
        "quantization": "symmetric int8, per row",
        "modelBytes": exported["byte_length"],
        "modelSha256": exported["sha256"],
        "trainQueries": len(train_examples),
        "heldoutQueries": len(heldout_examples),
        "validationQueries": len(validation_examples),
        "maxEpochs": args.epochs,
        "selectedEpoch": best_epoch,
        "seed": args.seed,
        "evaluation": {
            key: value
            for key, value in evaluation.items()
            if key not in {"queries", "abstentionQueries"}
        },
    }
    if abstention is not None:
        manifest["abstention"] = {
            "assetKey": "student-abstention.json",
            "bytes": abstention_bytes,
            "sha256": abstention_sha256,
            "evaluation": abstention_evaluation,
            "calibratedAt": abstention["calibratedAt"],
            "farTarget": ABSTENTION_FAR_TARGET,
        }
    (args.out / "student-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[write] {model_path} ({exported['byte_length'] / 1024:.1f} KiB)")
    if abstention_text is not None:
        print(f"[write] {abstention_path} ({abstention_bytes / 1024:.1f} KiB)")
    print(f"[write] {args.out / 'docs-vectors.f32'}")
    print(f"[eval] {json.dumps(manifest['evaluation'], sort_keys=True)}")


if __name__ == "__main__":
    main()
