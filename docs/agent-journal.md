# Agent journal: a private, agent-writable pack of development history

**Status: milestone 1 shipped in pikelet 0.11.0 (experimental).** `pikelet journal
init|list|compact`, `pikelet mcp --journal`, and the `write_note` and `latest`
tools work as described under [Implemented](#implemented-milestone-1). The
rest of this document is design; the open questions are still open.

## Summary

A *journal* is a `.pikelet` pack that an agent may write to through MCP. Agents
append self-contained notes as they work (handoffs, decisions, dead ends,
measurements), and any agent, from any vendor, can later mount the journal and
reconstruct what happened and why. The knowledge packs an agent reads stay
read-only; the journal is the one pack it can write, and it lives privately
with the user until the user chooses to share it.

## Motivation

A private pack built from conversation exports already did this once: a
different agent mounted it and reconstructed the development history of
Pikelet itself, across work done by two different assistants. That worked
because the pack is vendor-neutral (any MCP client reads it), searchable at a
scale no prompt holds, and every passage is citable as
`(pack identity, record id)`.

The existing places a project's history lives each miss something:

- **git** records *what* changed. Commit messages carry some *why*, but not the
  alternatives rejected, the experiments that failed, or the state of
  unfinished work.
- **`CLAUDE.md` / `AGENTS.md`** are small and always loaded: rules and
  pointers, not history.
- **Agent memory** is per-vendor and invisible to other agents.
- **Conversation transcripts** hold everything but are long, noisy and
  tool-specific; the history pack had to be built from exports after the fact.

The journal is the searchable, vendor-neutral *why*, written as the work
happens, by the agent that did it, for an agent that was not there.

## Prior art: the history pack

The pack that motivated this was compiled with the ordinary document pipeline
from exported conversations. Its structure, measured over all of its records,
shows what a purpose-built journal should do differently.

**Composition.** 335 Markdown files became 26,086 records in a 92 MB pack
(corpus 46 MB, encoder 26 MB, sketch index 13 MB, keyword index 7 MB):

- 327 exported chat conversations, one file each, named
  `YYYY-MM-DD-slug.md`, with a heading per turn (`T01 USER`,
  `T01 ASSISTANT`, ...): 99.5% of the records.
- 7 curated agent-session transcripts with descriptive titles, numbered
  across files: 68 records.
- 1 reconstructed archive of another assistant's conversations, written as
  dated entries with fixed subsections (topics, discussion, the user's theme,
  recovered analysis): 54 short records, averaging 63 words.

It was built with the standard section chunker (256-token target, 15%
overlap, 25-token minimum) and the pre-0.10 answerability calibrator
(cross-validated AUC 0.865). Records carry only the generic document fields:
title, heading path, source path, text.

**What the structure lost.**

- **Who said it, and when, lived only in headings and file names.** About 40%
  of records sit under a heading that is not a turn marker: a Markdown heading
  inside a reply, or a line of command output or code read as a heading,
  replaced the turn heading, and with it any record of who was speaking or in
  which turn.
- **The user's side mostly disappeared.** The conversations span about 5,000
  turns. Only about 15% of user turns kept a record of their own, against
  about 61% of assistant turns: short questions and instructions fall under
  the chunker's 25-token minimum. Those are the turns that say what was asked
  and decided, the *why* a later reader most needs.
- **Volume, not distillation.** The median conversation produced 39 records
  and the largest 939, much of it long tool output and logs.
- **Time is per file.** Dates come from file names, so a timeline can only
  resolve whole conversations.

**What worked.** The two small curated sources (titled session transcripts,
and dated entries with fixed subsections) are 0.5% of the records but the
easiest to orient by. They are the shape a journal should produce natively.

**Implications for this design.**

- Role, turn, date, author and kind are record fields, never headings.
- Notes bypass the chunker, so a short decision or instruction is kept.
- Transcript import keeps every user turn (attached to the reply it
  prompted), respects code fences, and summarizes long output instead of
  storing it.
- Dated entries with a fixed set of subsections are the template for
  `handoff` and `summary` notes.

## Goals

- An agent can record a note in one tool call and get back a citation for it.
- A different agent, with none of the writer's context, can orient itself:
  where the work stands, what was decided, what was tried and failed.
- Notes are anchored to the repository (commits, files, versions) so the
  journal and git can be read together.
- The journal is append-only and tamper-evident: every write is an immutable
  layer, so what an agent wrote, and when, is preserved.
- The user stays in control: they can review, remove and export notes, and
  nothing leaves their machine unless they move it.

## Non-goals (for now)

- Shared or team journals, journals committed to a repository, and merging
  journals written on different branches.
- Writing to any pack other than the journal.
- Calibrated `matchQuality` on the journal.

## Decisions so far

- **One writable pack, named at server start.** For example
  `pikelet mcp --pack docs.pikelet --journal pikelet` (a bare name is
  `~/.pikelet/journals/pikelet/`).
  No tool takes a path; the model cannot choose what it writes to.
- **Private by default.** The default location is outside any project tree, so
  it cannot be committed by accident. If the journal path is inside a git
  repository, `journal init` warns and offers a `.gitignore` entry. The
  directory is 0700 and its files 0600 whatever the umask; every write
  re-applies those modes, and a filesystem that ignores them gets a warning
  rather than a silent claim of privacy. Sharing is
  an explicit user command (`journal export`), never a tool. The journal can
  hold sensitive material (conversation-derived notes, internal decisions),
  which is why sharing waits until the feature has been proven in private use.
- **Unscored.** A journal is built and appended without calibration and reports
  `matchQuality: 'unscored'`. Notes are personal working records; the reader
  judges them by reading. This also means drift never pushes the journal into a
  refit, so compaction is only about chain length.
- **The model cannot delete.** Removal is a user action through the CLI. The
  model may supersede its own notes (see `update_note`).

## The note

One note is one corpus record. Notes bypass the document chunker: today's
chunker drops text under 25 tokens and splits long documents, and a note must
survive whole. A long note is still embedded in full, through the encoder's
windowed pooling.

Fields:

| Field | Meaning |
|---|---|
| `title` | One line, specific enough to scan in a list. |
| `text` | The body. Self-contained: no "as discussed", no session shorthand. |
| `kind` | `handoff`, `decision`, `dead-end`, `measurement`, `summary`, `todo`, `preference`. |
| `tags` | Free-form labels (`calibration`, `release-0.10`). |
| `created` | ISO timestamp, set by the server, not the model. |
| `authoredBy` | The MCP client, from the request's or session's `clientInfo` (e.g. `claude-code`), set by the server; plus the model name if the client supplies it, marked self-reported otherwise. |
| `session` | An id for the writing session, set by the server. |
| `repo` | Optional anchors: repository, branch, commit SHAs, files, versions or tags. |
| `sources` | Optional pinned citations `(pack identity, record id)` to passages the note relied on, instead of copying them. |

The record format today keeps only a fixed set of fields (`publicChunk` in
`ingest.mjs`: title, text, headingPath, url, anchor, sourcePath, preview), so
note records need their own builder that keeps these fields. Search ranks on
`title` and `text`; the other fields are returned with each result.

The kinds that matter most are the ones git never records: a **decision** with
the alternatives rejected, a **dead-end** ("tried X, it failed because Y", with
the evidence), a **measurement** with its numbers and method, and a
**handoff**. Tool descriptions carry a short template for each, for example a
handoff: goal, current state, done, next steps, open questions, pitfalls,
anchors.

Limits: a maximum note size (on the order of 16 KiB of text) and a per-session
write budget, both enforced by the server.

## Tools

### Writing

- **`write_note(kind, title, text, tags?, repo?, sources?)`** appends one note
  and returns its citation `(journal, record id)`, plus the head identity the
  write produced.
- **`update_note(id, text, title?)`** supersedes a note, for living documents
  such as "the current handoff for this project". Search returns only the
  current version; the superseded one stays reachable by id and through
  citation resolution. Open question: may an agent supersede notes from other
  sessions or other agents?

The descriptions tell the model when to write: at the end of a working
session (a handoff), after a decision, when an experiment concludes either
way, when the user states a durable preference. And what not to write: copies
of retrieved passages (cite them), secrets, or anything the user asked not to
be recorded.

### Reading

- **`orient()`** returns, in one call, what a newcomer needs first: the latest
  handoff, recent decisions, open todos and recent dead ends. "Where are we?" is
  not a semantic search.
- **`latest(kind?, tag?, limit?)`** lists notes newest first. Appends give
  each new note a higher id than every earlier one, so "newest" is "highest
  id", which is cheap; `created` breaks ties after compaction.
- **`timeline(from?, to?)`** lists notes between two dates, versions or commits.
- **`search`** works over the journal like any pack, with filters on kind and
  tag, and benefits from keyword matching (the BM25 half of hybrid
  retrieval, which ships), since notes are full of names, ids and
  identifiers.

Every journal result is labelled as agent-written and unverified (below).

## Trust and labelling

- **Labelled results.** Journal results carry `authoredBy`, `created` and an
  explicit marker that they are agent-written and unverified, so a later
  session does not weigh its own past notes like a published source.
- **Notes are data, never instructions.** If a note summarized a page that
  contained hostile text, the label, and the user's ability to remove the note,
  contain it. Tool descriptions say so.
- **Server-set metadata.** `created`, `session` and the client half of
  `authoredBy` come from the server, not the model, so a note cannot claim
  another client. MCP does not tell the server which model is behind a
  client; a model name the model supplies is recorded as self-reported.
- **Review.** `pikelet journal review` lists notes written since the last
  review, with removal and export from the same place.

## Storage and lifecycle

- **Init.** `pikelet journal init <name>` creates an empty journal. Compile
  needs at least one record today, so init either writes a seed record (the
  journal's own description) or the format gains an empty base.
- **Each write is a layer.** `write_note` appends a small immutable layer
  carrying one record (and `update_note` a supersession). The append embeds the
  note with the encoder inside the base, with no download and no model service.
  After the write the server remounts the chain so the next read sees it.
- **Compaction.** A chain may be at most 8 layers deep (the producer limit
  `append` enforces), so with one note per layer the journal folds itself into
  a new base every 8 notes, before the next write. Compaction reuses the
  stored vectors and records, so it takes seconds; it is synchronous today and
  could move to the background. Citations to notes translate forward through
  compaction, as they already do for any chain.
- **One writer.** A lock file serializes writes, so two local sessions (say,
  Claude and Codex) writing the same journal cannot fork it.
- **Retention.** Optional expiry for `todo` notes; compaction drops removed
  records.

## Command line

- `pikelet journal init <name> [--path …]`
- `pikelet journal list [--kind …] [--since …]`
- `pikelet journal review` (notes since the last review)
- `pikelet journal remove <id>…`
- `pikelet journal export [--format markdown|pikelet] --out …` (the only
  sharing path)
- `pikelet journal compact`
- `pikelet journal import …` (below)

## Bootstrapping a journal

A new journal for an existing project starts empty, which is when it is least
useful. Import, the way the history pack was built:

- agent session transcripts (Claude Code JSONL, Codex logs), summarized into
  notes rather than stored whole;
- commit messages and tags, as `summary` notes anchored to their SHAs;
- the CHANGELOG and design documents, as `summary` and `decision` notes.

Imported notes are labelled with how they were produced.

## What exists and what to build

| Need | Exists | To build |
|---|---|---|
| Append records as an immutable layer | `appendLayer` (CLI `append`), embeds with the base's own encoder | an API taking note records directly instead of `--source` folders |
| Supersede a record | `--supersede id=path`, chain-wide supersession | the same from a record, not a path |
| Remove a record | `--remove` | `journal remove` wrapper |
| Serve a base plus layers | chain mounting in `pikelet mcp` | remount after each write |
| Fold layers back into one file | `compact` | background trigger by layer count or size |
| Unscored packs | `--skip-calibration` | none |
| Grounding on unscored packs | coverage is computed only inside the calibrated scorer | coverage without a fit, useful for every unscored pack |
| Note-shaped records | `publicChunk` whitelists document fields | a note record builder |
| Write tools, orient/latest/timeline | — | new MCP tools and the `--journal` flag |
| Locking, limits, review | — | new |

## Deferred

- **Sharing.** Team journals, journals in a repository, Git LFS for their
  size, and merging two journals that forked on different branches (the
  layered format's `rebase` is the likely tool).
- **Small journals.** Every pack embeds its ~25 MiB encoder. A journal that
  references a pinned encoder by digest instead (the existing external-encoder
  profile) would be megabytes; it matters once journals are shared.
- **Calibration.** Revisit if journals grow large enough for `none` verdicts
  to help.

## Open questions

- One journal per project, or one per user with project tags?
- May an agent supersede notes from other sessions or other agents?
- Do `latest` and `timeline` need filters beyond kind, tag and date (author,
  repository)?
- Is uncalibrated grounding (which query words the note contains) enough of a
  check for journal search, or does journal search need none?
- Should the server ever write on its own (for example, a handoff when a
  session ends without one), or only when the model calls a tool?

## Milestones

1. **Write and read.** `journal init`, `write_note`, `latest`, search over the
   journal, labelling, locking, limits.
2. **Living documents and upkeep.** `update_note`, background compaction,
   `journal review` / `remove` / `export`.
3. **Orientation and import.** `orient`, `timeline`, transcript and git
   import, uncalibrated grounding.

## Implemented (milestone 1)

- **On disk** a journal is a directory: `journal.json` (format
  `pikelet-journal-v1`, the chain's member files base first), the base
  `base-NNNN.pikelet` (NNNN counts compactions), its layers
  `base-NNNN.LLLL.pikelet`, and a `.lock` while a write runs. A bare name
  resolves to `~/.pikelet/journals/<name>/`.
- **`pikelet journal init <name|dir>`** compiles a one-record seed base (the
  journal's own description) without calibration, and warns if the
  directory is inside a git repository.
- **`write_note`** checks the note (kind, one-line title of at most 200
  characters, body of at most 16 KiB, at most 20 tags, bounded `repo` anchors
  and `sources` citations; any other argument, including `created` or
  `authoredBy`, is refused), stamps `created`, `session` and `authoredBy`
  (the client name and version: a modern request's own `_meta` clientInfo,
  or a legacy session's from `initialize`), appends it as a layer with the
  base's own encoder, and remounts. Writes are serialized within a server,
  locked across processes (a writer waits up to 30 s for a live holder; a
  lock left by a dead process is broken under a second `.lock.break` lock, so
  two contenders cannot both take it over), and capped at 100 per session. A
  note that was written but could not be remounted is still reported
  written, with a warning, so the agent does not record it twice.
- **Citations.** A note's record id is stable: notes are never removed, so
  compaction renumbers nothing. The identity in a citation is the head the
  write produced and dates it; compaction replaces that head, and resolving
  a citation by identity is not implemented yet.
- **Freshness.** Before each tool call the server compares `journal.json`'s
  members with the chain it has open and reopens on a difference, so a
  long-running server sees notes and compactions from other sessions.
- **Records** are document-shaped (`title`, `text` beginning with the title,
  `headingPath` [kind]) plus a `note` object holding the metadata; any
  record without `note` (the seed) is not a note.
- **`latest(kind?, tag?, limit?)`** walks ids from the newest down, skipping
  removed records and the seed.
- **Labelling.** Journal results in `search`, `get_record` and `latest` carry
  `agentWritten: true` and the note metadata; a search response that includes
  the journal carries a `journalNote` in place of the generic unscored
  warning; `list_packs` marks the journal `writable`.
- **`pikelet journal list`** prints notes newest first; **`pikelet journal
  compact`** folds the layers by hand; **`pikelet mcp install --journal`**
  writes it into the client config.

Not yet: `update_note`, `orient`, `timeline`, search filters, `journal review`
/ `remove` / `export` / `import`, background compaction, uncalibrated
grounding.

## How we will know it works

Repeat the experiment that motivated it. Let one agent work on a project for
a stretch, writing to a journal. Then give a fresh agent, preferably from a
different vendor, only the journal and the repository, and a fixed set of
questions about the project's history: what was decided and why, what was
tried and abandoned, where the work stands, what to do next. Score the answers
against the record, and compare with the same questions answered from git and
the docs alone.
