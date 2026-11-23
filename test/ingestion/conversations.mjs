// ingestConversations conformance: Claude Code transcripts and Claude.ai and
// ChatGPT exports become one document per exchange, oldest first, carrying
// only what the people in the conversation said: no tool calls, tool output,
// reasoning or harness-injected text, and no credentials or email addresses.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, '..', '..', 'packages', 'pikelet', 'src');
const { ingestConversations, redactSecrets } = await import(pathToFileURL(path.join(src, 'conversation-source.mjs')).href);
const { chunkDocs, publicChunk } = await import(pathToFileURL(path.join(src, 'ingest.mjs')).href);
const { compileArtifact } = await import(pathToFileURL(path.join(src, 'cli.mjs')).href);

const rejects = async (fn, pattern) => {
  try { await fn(); } catch (error) { return pattern.test(error.message) ? true : error.message; }
  return 'did not throw';
};

const LONG = 'The ferry timetable is built from the harbour feed every night, and the build fails closed when the feed is stale.';

// A Claude Code session: a prompt, streamed assistant events (thinking,
// text, a tool call), the tool result, more text, then a second prompt
// with an injected reminder, plus every kind of event that is not the chat.
const claudeCodeSession = [
  { type: 'summary', summary: 'Ferry timetable build', leafUuid: 'x' },
  { type: 'user', sessionId: 'sess-1', cwd: '/home/someone/work/ferry', timestamp: '2026-03-01T10:00:00Z',
    message: { role: 'user', content: 'Why does the timetable build fail on Mondays?' } },
  { type: 'assistant', sessionId: 'sess-1', timestamp: '2026-03-01T10:00:05Z',
    message: { role: 'assistant', content: [{ type: 'thinking', thinking: 'SECRET-REASONING-MARKER' }] } },
  { type: 'assistant', sessionId: 'sess-1', timestamp: '2026-03-01T10:00:06Z',
    message: { role: 'assistant', content: [{ type: 'text', text: 'Let me look at the build script.' }, { type: 'tool_use', id: 't1', name: 'Read', input: { file_path: '/home/someone/work/ferry/build.sh' } }] } },
  { type: 'user', sessionId: 'sess-1', timestamp: '2026-03-01T10:00:07Z',
    message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'TOOL-OUTPUT-MARKER #!/bin/sh' }] } },
  { type: 'assistant', sessionId: 'sess-1', timestamp: '2026-03-01T10:00:09Z',
    message: { role: 'assistant', content: [{ type: 'text', text: `${LONG} On Mondays the feed publishes late, so the build sees Sunday's file.` }] } },
  { type: 'user', sessionId: 'sess-1', isMeta: true, timestamp: '2026-03-01T10:01:00Z',
    message: { role: 'user', content: 'META-MARKER caveat injected by the harness' } },
  { type: 'user', sessionId: 'sess-1', isSidechain: true, timestamp: '2026-03-01T10:01:10Z',
    message: { role: 'user', content: 'SIDECHAIN-MARKER subagent prompt' } },
  { type: 'user', sessionId: 'sess-1', isCompactSummary: true, timestamp: '2026-03-01T10:01:20Z',
    message: { role: 'user', content: 'COMPACT-MARKER This session is being continued' } },
  { type: 'user', sessionId: 'sess-1', timestamp: '2026-03-01T10:02:00Z',
    message: { role: 'user', content: [{ type: 'text', text: 'Can we retry the feed instead? My key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123 and mail me at ops@example.com <system-reminder>REMINDER-MARKER</system-reminder>' }] } },
  { type: 'assistant', sessionId: 'sess-1', timestamp: '2026-03-01T10:02:05Z',
    message: { role: 'assistant', content: [{ type: 'text', text: `${LONG} Yes: retry the feed three times, ten minutes apart, before failing the build.` }] } },
  { type: 'user', sessionId: 'sess-1', timestamp: '2026-03-01T10:03:00Z',
    message: { role: 'user', content: '[Request interrupted by user]' } },
].map((e) => JSON.stringify(e)).join('\n') + '\nnot json, a torn last line';

// A Claude.ai export: content blocks (with a tool use to drop), and an
// older message carrying only the flat text field.
const claudeAiExport = [{
  uuid: 'chat-ai-1', name: 'Harbour feed format', created_at: '2026-02-01T09:00:00Z',
  chat_messages: [
    { sender: 'human', created_at: '2026-02-01T09:00:00Z', text: 'What format is the harbour feed?', content: [] },
    { sender: 'assistant', created_at: '2026-02-01T09:00:10Z', text: '',
      content: [{ type: 'text', text: `${LONG} The harbour feed is GTFS with a custom sailings.txt extension.` }, { type: 'tool_use', name: 'web_search', input: { q: 'TOOL-USE-MARKER' } }] },
  ],
}];

// A ChatGPT export: an edited prompt leaves an abandoned branch; the kept
// branch has a system message, a hidden message, a tool call and its output.
const chatGptExport = [{
  id: 'gpt-1', title: 'Sailing cancellations', current_node: 'n6',
  mapping: {
    root: { id: 'root', message: null, parent: null, children: ['n0'] },
    n0: { id: 'n0', parent: 'root', message: { author: { role: 'system' }, content: { content_type: 'text', parts: ['SYSTEM-MARKER'] } } },
    n1old: { id: 'n1old', parent: 'n0', message: { author: { role: 'user' }, create_time: 1767225600, content: { content_type: 'text', parts: ['ABANDONED-BRANCH-MARKER'] } } },
    n1: { id: 'n1', parent: 'n0', message: { author: { role: 'user' }, create_time: 1767225660, content: { content_type: 'text', parts: ['How should cancelled sailings show in the timetable?'] } } },
    n2: { id: 'n2', parent: 'n1', message: { author: { role: 'assistant' }, recipient: 'python', content: { content_type: 'code', text: 'TOOL-CALL-MARKER' } } },
    n3: { id: 'n3', parent: 'n2', message: { author: { role: 'tool' }, content: { content_type: 'execution_output', text: 'TOOL-OUTPUT-MARKER' } } },
    n4: { id: 'n4', parent: 'n3', message: { author: { role: 'assistant' }, metadata: { is_visually_hidden_from_conversation: true }, content: { content_type: 'text', parts: ['HIDDEN-MARKER'] } } },
    n6: { id: 'n6', parent: 'n4', message: { author: { role: 'assistant' }, recipient: 'all', create_time: 1767225700, content: { content_type: 'text', parts: [`${LONG} Show cancelled sailings struck through, with the reason, rather than removing them.`] } } },
  },
}];

const MARKERS = ['SECRET-REASONING-MARKER', 'TOOL-OUTPUT-MARKER', 'META-MARKER', 'SIDECHAIN-MARKER', 'COMPACT-MARKER',
  'REMINDER-MARKER', 'TOOL-USE-MARKER', 'SYSTEM-MARKER', 'ABANDONED-BRANCH-MARKER', 'TOOL-CALL-MARKER', 'HIDDEN-MARKER',
  'Request interrupted'];

section('redactSecrets: credential shapes and email addresses');
{
  const cases = [
    ['an Anthropic key', 'use sk-ant-api03-abcdefghijklmnopqrstuvwx now', 'use [redacted] now'],
    ['a GitHub token', 'token ghp_abcdefghijklmnopqrstuvwxyz0123456789', 'token [redacted]'],
    ['an AWS access key id', 'AKIAABCDEFGHIJKLMNOP is the id', '[redacted] is the id'],
    ['a bearer token keeps the word Bearer', 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz.123', 'Authorization: Bearer [redacted]'],
    ['password= keeps the key', 'set password=hunter2hunter2 there', 'set password=[redacted] there'],
    ['an email address', 'write to ada@example.com today', 'write to [email] today'],
    ['a private key block', 'key:\n-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\nend', 'key:\n[redacted]\nend'],
  ];
  for (const [label, input, expected] of cases) {
    const { text } = redactSecrets(input);
    check(label, text === expected, JSON.stringify(text));
  }
  check('ordinary prose is untouched', redactSecrets('the secret to good docs is examples').text === 'the secret to good docs is examples');
}

const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'pikelet-ingest-conversations-'));
try {
  await fs.mkdir(path.join(dir, 'claude-code', 'sess-1', 'subagents'), { recursive: true });
  await fs.writeFile(path.join(dir, 'claude-code', 'sess-1.jsonl'), claudeCodeSession);
  await fs.writeFile(path.join(dir, 'claude-code', 'sess-1', 'subagents', 'agent-a.jsonl'),
    JSON.stringify({ type: 'user', sessionId: 'sub', timestamp: '2026-01-01T00:00:00Z', message: { role: 'user', content: 'SUBAGENT-FILE-MARKER' } }));
  await fs.mkdir(path.join(dir, 'claude-ai'));
  await fs.writeFile(path.join(dir, 'claude-ai', 'conversations.json'), JSON.stringify(claudeAiExport));
  await fs.mkdir(path.join(dir, 'chatgpt'));
  await fs.writeFile(path.join(dir, 'chatgpt', 'conversations.json'), JSON.stringify(chatGptExport));

  section('ingestConversations: Claude Code transcript');
  {
    const logs = [];
    const docs = await ingestConversations({ path: path.join(dir, 'claude-code', 'sess-1.jsonl') }, (l) => logs.push(l));
    check('two exchanges; the interrupt and the non-chat events make none', docs.length === 2, JSON.stringify(docs.map((d) => d.text.slice(0, 60))));
    const [first, second] = docs;
    check('the summary is the title', first?.title === 'Ferry timetable build', first?.title);
    check('the prompt and the whole reply are in one exchange',
      first?.text.startsWith('User: Why does the timetable build fail on Mondays?')
      && first.text.includes('Assistant: Let me look at the build script.')
      && first.text.includes("so the build sees Sunday's file"), first?.text);
    check('the exchange names its conversation, project, date and position',
      first?.text.endsWith('From the conversation "Ferry timetable build" in ferry on 2026-03-01, exchange 1 of 2.'), first?.text.slice(-120));
    check('the record carries the conversation',
      first?.conversation?.id === 'sess-1' && first.conversation.exchange === 1 && first.conversation.format === 'claude-code'
      && first.conversation.date === '2026-03-01T10:00:00Z' && first.conversation.project === 'ferry', JSON.stringify(first?.conversation));
    check('sourcePath names the conversation and exchange', second?.sourcePath === 'conversation/sess-1/2', second?.sourcePath);
    check('no URL: a local transcript has nowhere to link', docs.every((d) => d.url === null));
    check('the key and the email are redacted',
      second?.text.includes('My key is [redacted] and mail me at [email]') && !second.text.includes('sk-ant') && !second.text.includes('ops@example.com'), second?.text);
    check('the full working directory never reaches the text', !docs.some((d) => d.text.includes('/home/someone')));
    check('the log counts what it read and redacted',
      logs.some((l) => l.includes('1 conversation(s) with 2 exchange(s)') && l.includes('2 secret(s) or email address(es) redacted')), JSON.stringify(logs));
  }

  section('ingestConversations: Claude.ai and ChatGPT exports');
  {
    const ai = await ingestConversations({ path: path.join(dir, 'claude-ai', 'conversations.json') });
    check('Claude.ai: one exchange with the chat name as title',
      ai.length === 1 && ai[0].title === 'Harbour feed format' && ai[0].conversation.format === 'claude-ai', JSON.stringify(ai.map((d) => d.title)));
    check('Claude.ai: the flat text field is read when content is empty, and blocks otherwise',
      ai[0]?.text.includes('User: What format is the harbour feed?') && ai[0].text.includes('sailings.txt extension'), ai[0]?.text);
    const gpt = await ingestConversations({ path: path.join(dir, 'chatgpt', 'conversations.json') });
    check('ChatGPT: one exchange from the branch the person ended on',
      gpt.length === 1 && gpt[0].text.startsWith('User: How should cancelled sailings show in the timetable?'), JSON.stringify(gpt.map((d) => d.text.slice(0, 80))));
    check('ChatGPT: the date comes from create_time', gpt[0]?.conversation.date === new Date(1767225660 * 1000).toISOString(), gpt[0]?.conversation.date);
    check('ChatGPT: the reply is the visible assistant message', gpt[0]?.text.includes('struck through'), gpt[0]?.text);
  }

  section('ingestConversations: a folder of mixed exports');
  {
    const docs = await ingestConversations({ path: dir });
    check('all three formats are read, subagent transcripts skipped', docs.length === 4, JSON.stringify(docs.map((d) => d.title)));
    check('oldest conversation first, with ids in order',
      docs[0]?.conversation.format === 'chatgpt' && docs[1]?.conversation.format === 'claude-ai' && docs.every((d, i) => d.id === i),
      JSON.stringify(docs.map((d) => [d.id, d.conversation.format, d.conversation.date])));
    const all = JSON.stringify(docs);
    const leaked = [...MARKERS, 'SUBAGENT-FILE-MARKER'].filter((m) => all.includes(m));
    check('no tool call, tool output, reasoning, injected or abandoned text survives', leaked.length === 0, leaked.join(', '));

    const chunks = chunkDocs(docs, {});
    check('chunks keep the conversation', chunks.length > 0 && chunks.every((c) => c.conversation?.id), JSON.stringify(chunks.map((c) => c.conversation)));
    check('the public record keeps the conversation', publicChunk(chunks[0]).conversation?.exchange === chunks[0].conversation.exchange);
  }

  section('ingestConversations: errors');
  {
    await fs.writeFile(path.join(dir, 'other.json'), JSON.stringify([{ hello: 'world' }]));
    check('a JSON array that is no known export is refused',
      await rejects(() => ingestConversations({ path: path.join(dir, 'other.json') }), /not a Claude\.ai or ChatGPT conversation export/) === true);
    await fs.writeFile(path.join(dir, 'export.zip'), 'PK');
    check('a zip is refused with how to fix it',
      await rejects(() => ingestConversations({ path: path.join(dir, 'export.zip') }), /unzip it/) === true);
    check('a missing path is refused',
      await rejects(() => ingestConversations({ path: path.join(dir, 'nope') }), /not found/) === true);
  }
} finally {
  await fs.rm(dir, { recursive: true, force: true });
}

section('compile --conversations: flag combinations');
{
  const out = path.join(os.tmpdir(), `pikelet-conversation-flags-${process.pid}.pikelet`);
  check('--conversations with --git is refused',
    await rejects(() => compileArtifact({ conversations: '.', git: '.', out }), /got --git and --conversations/) === true);
  check('folder filters with --conversations are refused',
    await rejects(() => compileArtifact({ conversations: '.', include: ['*.md'], out }), /not --conversations/) === true);
}
