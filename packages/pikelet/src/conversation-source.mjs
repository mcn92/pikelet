// Conversation history as a source: one document per exchange (a prompt and
// the reply to it), so a pack can answer "what did we decide about X?" and
// "how did I fix that last time?" from past chats, citing the conversation
// and exchange that says so.
//
// Three export formats are read, detected from their shape:
//   - Claude Code session transcripts (*.jsonl, one event per line)
//   - Claude.ai data exports (conversations.json, chat_messages per chat)
//   - ChatGPT data exports (conversations.json, a mapping tree per chat)
//
// Only what the people in the conversation wrote to each other is kept:
// tool calls, tool output, reasoning, and harness-injected text are dropped.
// Tool output is where file dumps, environment listings, and pasted logs
// live, and none of it is the conversation. What is kept still passes
// through redactSecrets, because people paste keys into chats too.

import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import { CliError } from './common.mjs';

const MAX_TITLE_CHARS = 80;
const MAX_FILES = 100_000;

// Pattern-based, so a best effort and not a guarantee: it catches the
// shapes credentials commonly take, not every secret a person can type.
const SECRET_PATTERNS = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
  /\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}/g,
  /\bgh[pousr]_[A-Za-z0-9]{30,}\b/g,
  /\bgithub_pat_[A-Za-z0-9_]{30,}\b/g,
  /\bnpm_[A-Za-z0-9]{36}\b/g,
  /\bxox[abprs]-[A-Za-z0-9-]{10,}/g,
  /\bAKIA[0-9A-Z]{16}\b/g,
  /\bAIza[0-9A-Za-z_-]{35}\b/g,
  /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/g,
  /\b(Bearer\s+)[A-Za-z0-9._~+/-]{20,}=*/gi,
  // key=value and key: value, keeping the key so the sentence still reads.
  /\b((?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)\s*[:=]\s*["']?)[^\s"']{6,}/gi,
];
const EMAIL = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g;

function redactSecrets(text) {
  let count = 0;
  let out = text;
  for (const pattern of SECRET_PATTERNS) {
    out = out.replace(pattern, (match, keep) => {
      count += 1;
      return `${typeof keep === 'string' ? keep : ''}[redacted]`;
    });
  }
  out = out.replace(EMAIL, () => { count += 1; return '[email]'; });
  return { text: out, count };
}

// Text Claude Code puts in the user's turn that the user did not type:
// injected reminders, slash-command plumbing, and command output.
const HARNESS_TAGS = /<(system-reminder|command-name|command-message|command-args|local-command-stdout|local-command-stderr|user-prompt-submit-hook)>[\s\S]*?<\/\1>/g;

function cleanText(text) {
  return String(text || '').replace(HARNESS_TAGS, '').replace(/\n{3,}/g, '\n\n').trim();
}

function textBlocks(content) {
  if (typeof content === 'string') return cleanText(content);
  if (!Array.isArray(content)) return '';
  return content
    .filter((b) => b && b.type === 'text' && typeof b.text === 'string')
    .map((b) => cleanText(b.text))
    .filter(Boolean)
    .join('\n\n');
}

function shortTitle(text) {
  const line = String(text || '').split('\n').find((l) => l.trim()) || '';
  const t = line.trim();
  return t.length > MAX_TITLE_CHARS ? `${t.slice(0, MAX_TITLE_CHARS - 1).trimEnd()}…` : t;
}

// Fold a run of turns into exchanges: each starts at a user turn and takes
// every assistant turn up to the next one. User turns in a row before any
// reply are one prompt sent in parts. Assistant text before any prompt (a
// greeting, a resumed session's tail) has no question to answer and is
// dropped.
function toExchanges(turns) {
  const exchanges = [];
  for (const turn of turns) {
    if (!turn.text) continue;
    const last = exchanges[exchanges.length - 1];
    if (turn.role === 'user' && last && !last.assistant.length) {
      last.user.push(turn.text);
    } else if (turn.role === 'user') {
      exchanges.push({ date: turn.date, user: [turn.text], assistant: [] });
    } else if (exchanges.length) {
      exchanges[exchanges.length - 1].assistant.push(turn.text);
    }
  }
  return exchanges.filter((e) => e.assistant.length);
}

// --- Claude Code: one JSON event per line --------------------------------

function parseClaudeCode(raw, file) {
  const turns = [];
  let title = null;
  let sessionId = null;
  let project = null;
  for (const line of raw.split('\n')) {
    if (!line.trim()) continue;
    let event;
    try { event = JSON.parse(line); } catch { continue; }
    if (!event || typeof event !== 'object') continue;
    if (event.type === 'custom-title' && event.customTitle) title = String(event.customTitle);
    if (event.type === 'summary' && event.summary && !title) title = String(event.summary);
    if (event.type !== 'user' && event.type !== 'assistant') continue;
    // Subagent runs, injected context, compaction summaries and API error
    // stand-ins are not the conversation.
    if (event.isSidechain || event.isMeta || event.isCompactSummary || event.isApiErrorMessage) continue;
    sessionId ||= event.sessionId || null;
    project ||= event.cwd ? path.basename(String(event.cwd)) : null;
    const message = event.message || {};
    const content = message.content;
    // A user event that carries only tool results is the harness reporting
    // back, not the person speaking.
    if (event.type === 'user' && Array.isArray(content) && content.every((b) => b?.type === 'tool_result')) continue;
    const text = textBlocks(content);
    if (!text || /^\[Request interrupted by user/.test(text)) continue;
    turns.push({ role: event.type, text, date: event.timestamp || null });
  }
  const exchanges = toExchanges(turns);
  return [{
    id: sessionId || path.basename(file, '.jsonl'),
    title: title || shortTitle(exchanges[0]?.user[0]) || path.basename(file, '.jsonl'),
    project,
    exchanges,
  }];
}

// --- Claude.ai export: [{ uuid, name, chat_messages: [...] }] -------------

function parseClaudeAi(chats) {
  return chats.map((chat) => {
    const turns = (chat.chat_messages || []).map((m) => ({
      role: m.sender === 'human' ? 'user' : m.sender === 'assistant' ? 'assistant' : null,
      // content blocks when present (they separate text from tool use);
      // the flat text field otherwise, as older exports have.
      text: Array.isArray(m.content) && m.content.length ? textBlocks(m.content) : cleanText(m.text),
      date: m.created_at || null,
    })).filter((t) => t.role);
    const exchanges = toExchanges(turns);
    return {
      id: String(chat.uuid || ''),
      title: (chat.name && String(chat.name).trim()) || shortTitle(exchanges[0]?.user[0]) || 'Untitled conversation',
      project: null,
      exchanges,
    };
  });
}

// --- ChatGPT export: [{ id, title, mapping: { node: {...} }, current_node }]

function parseChatGpt(chats) {
  return chats.map((chat) => {
    const mapping = chat.mapping || {};
    // The mapping is a tree of every edit and regeneration; the branch the
    // person ended on is current_node's ancestry.
    const branch = [];
    const seen = new Set();
    let node = chat.current_node;
    while (node && mapping[node] && !seen.has(node)) {
      seen.add(node);
      branch.push(mapping[node]);
      node = mapping[node].parent;
    }
    branch.reverse();
    const turns = [];
    for (const { message } of branch) {
      if (!message) continue;
      const role = message.author?.role;
      if (role !== 'user' && role !== 'assistant') continue;
      if (message.metadata?.is_visually_hidden_from_conversation) continue;
      // An assistant message addressed to a tool is a tool call.
      if (role === 'assistant' && message.recipient && message.recipient !== 'all') continue;
      const content = message.content || {};
      if (!['text', 'multimodal_text'].includes(content.content_type)) continue;
      const text = cleanText((content.parts || []).filter((p) => typeof p === 'string').join('\n\n'));
      const date = typeof message.create_time === 'number' ? new Date(message.create_time * 1000).toISOString() : null;
      turns.push({ role, text, date });
    }
    const exchanges = toExchanges(turns);
    return {
      id: String(chat.conversation_id || chat.id || ''),
      title: (chat.title && String(chat.title).trim()) || shortTitle(exchanges[0]?.user[0]) || 'Untitled conversation',
      project: null,
      exchanges,
    };
  });
}

function parseJsonExport(raw, file) {
  let data;
  try { data = JSON.parse(raw); } catch (error) {
    throw new CliError(`${file} is not valid JSON: ${error.message}`, 1);
  }
  if (!Array.isArray(data)) throw new CliError(`${file} is not a conversation export (expected a JSON array of conversations)`, 1);
  if (!data.length) return { format: 'empty', conversations: [] };
  if (data.some((c) => c && typeof c === 'object' && c.mapping)) return { format: 'chatgpt', conversations: parseChatGpt(data) };
  if (data.some((c) => c && typeof c === 'object' && Array.isArray(c.chat_messages))) return { format: 'claude-ai', conversations: parseClaudeAi(data) };
  throw new CliError(`${file} is not a Claude.ai or ChatGPT conversation export`, 1);
}

async function listFiles(root) {
  const stat = await fs.stat(root);
  if (stat.isFile()) return [root];
  const out = [];
  const walk = async (dir) => {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        // Claude Code keeps subagent transcripts in a subagents folder
        // beside the session; they are the agent's work, not the chat.
        if (entry.name === 'subagents' || entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
        await walk(full);
      } else if (entry.isFile() && (entry.name.endsWith('.jsonl') || entry.name === 'conversations.json')) {
        out.push(full);
        if (out.length > MAX_FILES) throw new CliError(`More than ${MAX_FILES} transcript files under ${root}`, 1);
      }
    }
  };
  await walk(root);
  // Code-unit order, not localeCompare: file order sets record order, part of the pack identity.
  return out.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

function exchangeText(conversation, exchange, index) {
  const date = exchange.date ? String(exchange.date).slice(0, 10) : null;
  return [
    ...exchange.user.map((t) => `User: ${t}`),
    ...exchange.assistant.map((t) => `Assistant: ${t}`),
    `From the conversation "${conversation.title}"${conversation.project ? ` in ${conversation.project}` : ''}`
      + `${date ? ` on ${date}` : ''}, exchange ${index + 1} of ${conversation.exchanges.length}.`,
  ].join('\n\n');
}

async function ingestConversations(source, log = () => {}) {
  const root = path.resolve(source.path);
  if (!fssync.existsSync(root)) {
    throw new CliError(`Conversation export not found: ${root}\nNext: check --conversations`, 1);
  }
  if (/\.zip$/i.test(root)) {
    throw new CliError(`${root} is a zip archive\nNext: unzip it and pass conversations.json (or the folder it is in) to --conversations`, 1);
  }
  const files = await listFiles(root);
  const conversations = [];
  const formats = new Map();
  for (const file of files) {
    const raw = await fs.readFile(file, 'utf8');
    const parsed = file.endsWith('.jsonl')
      ? { format: 'claude-code', conversations: parseClaudeCode(raw, file) }
      : parseJsonExport(raw, file);
    formats.set(parsed.format, (formats.get(parsed.format) || 0) + parsed.conversations.length);
    for (const c of parsed.conversations) conversations.push({ ...c, format: parsed.format });
  }
  const kept = conversations.filter((c) => c.exchanges.length);
  // Oldest first, so record ids grow with time and a later append of new
  // conversations continues the order.
  kept.sort((a, b) => String(a.exchanges[0].date || '').localeCompare(String(b.exchanges[0].date || '')));
  const docs = [];
  let redacted = 0;
  for (const conversation of kept) {
    conversation.exchanges.forEach((exchange, index) => {
      const { text, count } = redactSecrets(exchangeText(conversation, exchange, index));
      const title = redactSecrets(conversation.title).text;
      redacted += count;
      docs.push({
        id: docs.length,
        sourcePath: `conversation/${conversation.id}/${index + 1}`,
        url: null,
        title,
        slug: null,
        text,
        sections: [],
        conversation: {
          id: conversation.id,
          title,
          exchange: index + 1,
          date: exchange.date || null,
          format: conversation.format,
          ...(conversation.project ? { project: conversation.project } : {}),
        },
      });
    });
  }
  const summary = [...formats].map(([f, n]) => `${n} ${f}`).join(', ') || 'none';
  log(`Read ${kept.length} conversation(s) with ${docs.length} exchange(s) from ${files.length} file(s) (${summary}); `
    + `tool calls, tool output and reasoning excluded; ${redacted} secret(s) or email address(es) redacted`);
  return docs;
}

export { ingestConversations, redactSecrets, parseClaudeCode, parseClaudeAi, parseChatGpt };
