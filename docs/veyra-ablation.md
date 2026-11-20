# The Veyra ablation: what happens when the answer is removed from the file

Moved verbatim from the README on 2026-09-23. The reproduction commands reference paths in this repository.

## What happens if the answer is changed or removed from the file?

This is different from asking whether search returns sensible documents. To test whether Pikelet's retrieval-quality signal could form a useful evidence boundary, a synthetic corpus called **Station Veyra Registry** was built. One version contained the fact:

```text
The Tovash project is housed in Chamber 17.
```

A second pack was byte-for-byte identical except that the record containing that fact was removed.

```text
full pack:     matchQuality: strong    confidence: 0.915   → Chamber 17
ablated pack:  matchQuality: none      confidence: 0.136   → unsupported
```

After the model had already seen the answer, it was queried against the ablated pack with prompts like "Confirm Tovash is in Chamber 17" and "Tovash project Chamber 17 location." The retrieval result stayed unsupported and the model declined to confirm the location from the pack. A separate, informal session then probed a fresh isolated agent with neutral prompts, leading prompts, authority pressure, invitations to use general knowledge, cross-record distractors, and repeated pressure — six adversarial framings — and it continued distinguishing supported facts from the removed one in every case; that session wasn't captured as a script, so treat it as a described observation rather than a reproducible result.

This does **not** mean Pikelet can prevent an LLM from hallucinating. It means the artifact can expose an explicit evidence boundary that a consuming model can choose to respect — and removing evidence from the artifact changed what that model was able to support from the mounted source.

A second intervention changed only the Tovash location source record from **Chamber 17** to **Chamber 43**, rebuilt the pack, and repeated the same prompt in a fresh session:

```text
chamber43 pack: matchQuality: strong    confidence: 0.916   → Chamber 43
```

The model answered **Chamber 43** and cited the same logical source record. With Veyra not mounted at all, the same prompt produced no chamber number and the model declined to guess. Change the evidence and the grounded answer changes with it; remove the evidence source and the answer disappears.

The test is synthetic and intentionally narrow. The exact packs used for the results above are published as the `veyra-packs-v1` GitHub release assets rather than committed to the repository. Fetch and SHA-256-verify them once, then reproduce the retrieval-side intervention:

```bash
npm run demo:veyra
node examples/one-file-search/web/public/reproduce-ablation.mjs
```

The script reproduces `matchQuality`, confidence, and retrieved evidence for the full, ablated, and Chamber 43 packs. The paired LLM-session results — declining to confirm the removed fact, answering Chamber 43 after the mutation, and declining without Veyra mounted — were run separately and are not scripted here.

## Packs compiled with 0.10 or later

The published packs above were calibrated before 0.10, against on-topic questions the corpus cannot answer, and they keep that behavior: the files are immutable. From 0.10, `matchQuality` measures retrieval relevance instead (see [docs/calibration.md](calibration.md)), because on hand-written questions the earlier fit withheld about half of the answerable ones. Rebuilt with 0.10 from the full and ablated Veyra corpora, the same question gives:

```text
full pack:     matchQuality: strong    grounding.uncovered: []
ablated pack:  matchQuality: strong    grounding.uncovered: ["chamber", "housed"]
```

The ablated pack still covers the Tovash project — its supervision and clearance records are there — so it is on topic and says so. The boundary around the removed fact moves to `grounding.uncovered`: no returned passage mentions a chamber, and the MCP `search` tool tells the model that an asked-for value in that list is not established. After 0.14.0, `pikelet mcp` reports only the grounding, and `compile` fits no verdict ([docs/calibration.md](calibration.md) explains why); the published packs keep theirs, which the `pikelet-wasm` library still returns.

---
