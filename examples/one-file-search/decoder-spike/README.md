# Decoder spike (text generation on the inline-encoder kernel)

Can the integer GEMM that embeds queries inside a `.pikelet` file also run a
generative model? This spike says yes, for small models. It is
a spike, not a feature: nothing here is part of either package.

- **Two dependency-free WASM kernels.** `decoder.cpp` runs GPT-Neo
  (TinyStories-33M and TinyStories-Instruct-33M). `llama.cpp` (this
  project's file, unrelated to the llama.cpp project) runs Llama-architecture
  models (SmolLM2-135M and 360M Instruct). Both reuse
  `encoder-spike/encoder.cpp`'s weight format and integer GEMM. The modules
  are 57 KB and 42 KB.
- **Faithful to torch.** Greedy text matches torch running the same
  quantized weights token for token, and the kernel's perplexity matches
  torch's to four decimal places.
- **Bit-identical across engines.** Every logit the GPT-Neo kernel writes
  is the same bit for bit in Node, Chromium, WebKit and Firefox, so a given
  prompt and model produce the same text everywhere.
- **Pack-grounded answers.** SmolLM2-360M answering from the Wikipedia
  pack's top passages gets 54 of 70 answerable questions right, against 2 of
  70 with no passages. A simple check on the question's names catches all 15
  questions about things that don't exist.

All numbers: Ryzen 9 4900HS, single thread, WASM SIMD128, 2026-10-26 to
2026-10-28.

## The GPT-Neo kernel

`decoder.cpp` runs TinyStories-33M (GPT-Neo: 4 layers, width 768, 16 heads,
vocabulary 50,257). It uses the encoder's block-affine weights (u8 in blocks
of 64, or u4 in blocks of 32; plus a u5 format, below) and its integer GEMM
(activations quantized to int16 per token, `i32x4.dot_i16x8_s` per weight
block). What's new for decoding:

- **KV cache:** each call appends K/V at positions `pos0…`. One sequence at a
  time, capped at 1,024 positions.
- **Attention:** causal, unscaled scores (GPT-Neo), SIMD. Layers 1 and 3
  are local: they see only the last 256 positions.
- **Block layout:** pre-LayerNorm blocks and tanh GELU (SIMD).
- **Output:** a final LayerNorm, then a tied LM head (the token-embedding
  table used as a 50,257 × 768 GEMM), optionally through a vocabulary
  shortlist (below).

A GPT-2 byte-level BPE tokenizer (`bpe.mjs`) and a sampling loop
(`tell.mjs`) run in JavaScript.

```bash
pip install torch transformers datasets   # exporters + references only
python3 export_decoder.py                 # tinystories-u8.bin, tokenizer/, refs.json
python3 export_decoder.py 4               # tinystories-u4.bin
python3 export_decoder.py 5               # tinystories-u5.bin (quant_u5.py)
bash build.sh                             # all four modules (checked in; needs emsdk)
node tell.mjs check                       # tokenizer, logits, greedy vs torch-u8
node tell.mjs "Once upon a time," 200 0.8 # prompt, max tokens, temperature (0 = greedy)

mkdir -p data && curl -L -o data/TinyStories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
python3 eval_ref.py 300                   # torch fp32/u8/u4/u5 references
node eval.mjs 8 && node eval.mjs 4 && node eval.mjs 5   # long-context + perplexity, kernel vs torch
node expand_check.mjs                     # u5 expanded at load == u5 in-kernel, and speed
node speed.mjs                            # prompt reading and generation speed per format
```

Weight blobs, tokenizers, data and reference outputs are generated and
git-ignored. `build.sh` is deterministic: rebuilding gives byte-identical
modules.

### Correctness

Kernel against torch running the same dequantized weights:

| Check | u8 | u4 | u5 |
| --- | --- | --- | --- |
| Tokenizer ids, 3 prompts | identical | — | — |
| Greedy, 3 prompts × 60 tokens | 60/60 identical, all 3 | — | — |
| 1,000 teacher-forced tokens: top token, positions 0–255 | 256/256 | 256/256 | 256/256 |
| … positions 256–999 (past the local window) | 744/744 | 744/744 | 744/744 |
| … max logit difference | 0.008 | 0.008 | 0.007 |
| … one token per call vs 200 per call | identical | identical | identical |

Perplexity on the first 300 TinyStories validation stories (63,495 tokens);
the kernel matches torch to four decimal places in every format:

| | Blob | Perplexity | vs fp32 |
| --- | ---: | ---: | ---: |
| fp32 (torch) | — | 3.2895 | — |
| u8 | 77.2 MB | 3.2913 | +0.06% |
| u5 | 51.5 MB | 3.3138 | +0.74% |
| u4 | 51.5 MB | 3.5208 | +7.0% |

### Speed

SIMD attention and GELU took a 1,000-token prompt from 4.18 s to 2.71 s, and
generation from 129 to 145 tokens/s at short context (91 to 123 at 968
tokens), with u8 still matching torch on 1,000 of 1,000 tokens. u4 is no
faster than u8: with one token per call each weight block is widened for a
single use, so generation is bound by compute, not memory bandwidth.

The LM head is the largest cost (54% of short-context generation: 38.6M
multiply-adds per token against 28.3M for all four layers).
`export_shortlist.py` fits a low-rank approximation of the embedding table
to the hidden states the model actually produces. The kernel scores every
token with it, keeps the top K, and rescores those exactly, so the logits it
keeps are bit-identical to the full head's (`shortlist_check.mjs`, 1,000
teacher-forced tokens):

| Shortlist | Speed | Top token kept | Top-40 kept |
| --- | ---: | ---: | ---: |
| rank 192, K 256 | 1.57× | 999/1000 | 98.3% |
| rank 256, K 256 | 1.40× | 1000/1000 | 99.1% |

Fitting to the hidden states matters: a plain SVD of the table keeps only
31–59% of the logit energy at these ranks, against 93–97%.

### u5: u8-like quality at u4's size

`quant_u5.py`: 5-bit codes in blocks of 32 with f16 scale and offset, fitted
by least squares (scale and offset refit given the codes, codes re-rounded,
then re-rounded against the stored f16 values). 24 bytes per block, 6.0 bits
per weight: byte-for-byte u4's size. In simulation (torch, the same 300
stories, plus greedy agreement with fp32 on 200 held-out prompts × 32
tokens):

| | Bits/weight | Perplexity gap | Top-1 = fp32 | Greedy tokens before divergence |
| --- | ---: | ---: | ---: | ---: |
| u4 min/max | 6.0 | +7.03% | 87.7% | 7.5 |
| u5 least-squares, f16 s/o | 6.0 | +0.74% | 94.7% | 14.5 |
| u6 least-squares | 8.0 | +0.52% | 96.8% | 20.3 |
| u8 min/max | 9.0 | +0.06% | 99.2% | 27.4 |

Most of the gain is the fifth bit (u5 min/max: +1.13%); least squares adds
the rest, and f16 scale/offset costs nothing measurable at any width. Near
fp32, perplexity flattens out while the agreement columns keep separating.

Decoding u5 inside the GEMM is slow: with one token per call each block
does about 20 SIMD ops of real work, and unpacking the fifth bit adds about
8 (x86 has no byte-wide SIMD shifts, so V8 synthesizes them).
`decoder_expand_u5` instead expands the file once at load into u8 codes
with f32 scale/offset, with bit-identical logits (measured before the SIMD
attention work):

| Decode, 300 tokens, one per call | File | Memory | Tokens/s |
| --- | ---: | ---: | ---: |
| u8 | 77.2 MB | 77.2 MB | 110 |
| u4 | 51.5 MB | 51.5 MB | 101 |
| u5, decoded in the GEMM | 51.5 MB | 51.5 MB | 57 |
| u5, expanded at load (0.3 s) | 51.5 MB | 85.7 MB | 98 |

## Bit-identical in every engine

`browser_check.mjs` runs `workload.mjs` unchanged in Node and in Chromium,
WebKit and Firefox (Playwright) and hashes every logits row the kernel
writes. All five configurations (u8, u4, u5, u5 expanded, u8 with the r192
K256 shortlist) are bit-identical to Node in every engine: 600 one-per-call
rows, 600 prefill rows and 300 greedy tokens each in Chromium 148 and WebKit
26.4, and 300 + 300 rows and 90 tokens in Firefox 150.

| Greedy tokens/s | Node | Chromium | WebKit |
| --- | ---: | ---: | ---: |
| u8 | 148 | 156 | 154 |
| u4 | 137 | 133 | 121 |
| u5, decoded in the GEMM | 74 | 76 | 72 |
| u5, expanded at load | 137 | 138 | 127 |
| u8 + shortlist | 234 | 239 | 244 |

Playwright's Firefox build runs this WASM 35–80× slower (1–4 tokens/s):
it appears to use only the baseline compiler. Whether stock Firefox does the
same is not yet checked.

## Is the output any good?

`score.mjs`: as a scorer, picking the true next sentence of a story from
four candidates (200 items), the u8 kernel makes the same choice as torch
fp32 on 200/200 items (summed log-likelihood; accuracy 0.805).

`instruct.mjs`: TinyStories-Instruct-33M on 50 "write a story using these
words" and 50 "summarize this story" items. Required words used: kernel u8
82.0%, torch fp32 80.7%. Summaries: ROUGE-1 F1 0.579 against 0.573, with
41 of 50 identical to torch's text. Prompt reading 460 tokens/s, generation
141.

`pragmatic.mjs`: 40 hand-written prompts in five categories, 8 each,
graded blind by an LLM judge (labels shuffled per item; PASS 1, PARTIAL ½).
Out of 8:

| | Grammar | Narrative | Instruction | Factual | Grounded |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyStories-33M | 7.0 | 6.5 | 2.0 | 5.5 | 2.0 |
| TinyStories-Instruct-33M | 7.5 | 6.0 | 6.0 | 3.5 | 3.0 |

Fluent, coherent stories, and the Instruct model follows its formats, but
a 33M model trained on children's stories can't answer questions from a
passage. A second blind pass over u8, u5 and u5 with the shortlist scored
them the same within judge noise (u5 22.0 and 26.5 of 40 against u8 21.5
and 26.0; regrading u8 agreed with the first pass on 35 and 36 of 40 items).

## The Llama kernel: SmolLM2

`llama.cpp` reads its shape from the blob header and adds RMSNorm,
rotate-half RoPE, grouped-query attention and SwiGLU; `bpe.mjs` gained
special tokens and digit splitting. u8 and u5 (expanded at load).

```bash
python3 export_llama.py 8                     # smollm2-135m-u8.bin + llama-tokenizer/
python3 export_llama.py 8 HuggingFaceTB/SmolLM2-360M-Instruct smollm2-360m
python3 llama_ref.py HuggingFaceTB/SmolLM2-135M-Instruct smollm2-135m 8
node llama_check.mjs smollm2-135m 8
```

- Tokenizer: 10/10 test strings and 100/100 chat prompts identical to
  Hugging Face.
- Greedy chat replies identical to torch on every prompt checked.
- 1,000 teacher-forced tokens: the top token agrees on 997–999.
- Perplexity (360M): fp32 4.995, u8 5.007, u5 5.189. u5 costs +3.9% here
  against +0.74% on TinyStories. The largest logit difference is at
  position 0 (0.55 for 360M u8), where per-token int16 activations meet the
  model's outsized first-token activations.
- Speed, 360M, Node, one thread: prompt reading 41 tokens/s, generation 24.
  At about 30 GFLOP/s it is compute-bound like the GPT-Neo kernel, so
  threads are the lever. 135M generates 70–84 tokens/s.

## Answering from a pack

**SQuAD v2** (`grounded_items.py`, `grounded.mjs`; 100 questions, 70
answerable, 30 whose paragraph is on topic but lacks the answer).
SmolLM2-360M, reply contains a gold answer: torch fp32 64.3%, torch u8
65.7%, kernel u8 64.3%, kernel u5 61.4%. It rarely abstains on the
unanswerable ones (6.7% for u8; u5 23.3%, at the cost of 18.6% false
abstentions). Faithful kernel, but a 360M model is short of a useful
grounded-QA bar.

**Pack-native questions** (`pack_sample.mjs`, `pack_rag_retrieve.mjs`,
`rag_ref.py`, `rag_score.mjs`, `kernel_run.mjs`). 100 questions written by
an LLM from 140 passages sampled from the Simple English Wikipedia pack
(`data/pack_qa_items.json`, checked in): 70 answerable from the passage, 15
asking for a detail the passage lacks, 15 about things that don't exist.
For 57 of the 70, the pack's hybrid retrieval ranks a passage containing
the answer first, and for 69 it has one in the top 3. SmolLM2-360M u8:

| Condition | Answerable correct | Declined (of 30) |
| --- | ---: | ---: |
| No passages | 2/70 | 4/30 |
| Top passage | 48/70 | 1/30 |
| Top 3 passages | 54/70 | 0/30 |
| Top 3 + name check | 53/70 | 18/30 (all 15 fictional) |

The model alone answers almost nothing and rarely declines. With the
passages it answers most questions, but it declines almost nothing. The name
check (`rag_score.mjs`, gate `names`) declines when a capitalized word or
number from the question appears in none of the passages, which is the same
signal as a search result's `grounding.uncovered`. It cut confident errors
from 46 to 26 at the cost of one correct answer; checking that every word of
the answer appears in the passages (gate `support`) cut them to 30 but cost
seven. The pack's own `matchQuality` didn't help: this copy of the pack
predates the relevance calibrator and rated all 100 questions "strong".

The kernel gives the same reply as torch on all 100 top-3 items (prompt
reading 38 tokens/s, so a 561-token prompt takes 14.6 s; generation 21
tokens/s).

## Streaming the vocabulary (simulation)

`record_candidates.mjs` and `remote_rows_sim.py` ask whether the exact
embedding rows for the shortlist's rescoring (576 bytes each in u5) could be
range-read from a pack on demand instead of shipped. TinyStories generation
touches 11.7k of the 50,257 rows at K 128 and 15.9k at K 256. Ordered by how
often each row is a candidate (not by token id), with a 16 MB row cache at
K 128 seeded with the hottest rows, fewer than 1% of steps wait on a fetch. In practice that means
shipping a hot prefix and fetching the rare tail, and a fetch must block
rather than fall back to the approximate score, or output would depend on
network timing. TinyStories' vocabulary is narrow, so this is optimistic for
general text. The same idea applies to encoders, where input rows are known
as soon as the text is tokenized.

## Not done

- Threads, and SIMD prompt reading for the Llama kernel.
- A ring-buffer cache for GPT-Neo's local layers.
- GPT-2 124M.
- Firefox outside Playwright.

## Data and licenses

TinyStories data and models (roneneldan/TinyStories, CDLA-Sharing-1.0):
`score_items.json` and `instruct_items.json` hold token ids and text
derived from its validation files. SmolLM2 (HuggingFaceTB, Apache-2.0). SQuAD
v2 (CC BY-SA 4.0) is downloaded by `grounded_items.py`, not checked in.
`data/pack_qa_items.json` quotes answers from Simple English Wikipedia
(CC BY-SA 4.0); the pack is at
`https://pub-6da2384a3bca4a44b2b2fa29a94cc811.r2.dev/wikipedia.pikelet`
(648.5 MiB; save it as `data/wikipedia.pikelet`).
