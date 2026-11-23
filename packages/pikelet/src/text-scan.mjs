// Linear-time scanners for text shapes whose obvious regular expression
// backtracks: a pattern like /\/+$/ or /[ \t]*\|/ restarts at every
// character of a long run of slashes or spaces that is not followed by what
// it needs, so a 40 KB run costs about a second and a 400 KB one minutes.
// Crawled pages and documents being ingested are untrusted input. Each
// function here returns exactly what the expression named in its comment
// returns; test/ingestion/linear.mjs checks that against the expression on
// random input and checks the running time on long runs.

// \s in a JavaScript regular expression and String.prototype.trim use the
// same set (WhiteSpace and LineTerminator).
export const isWhitespace = (c) => c !== undefined && c !== '' && c.trim() === '';

// The characters a regular expression's '.' does not match, and where a
// multiline ^ and $ match.
export const isLineTerminator = (c) => c === '\n' || c === '\r' || c === '\u2028' || c === '\u2029';

export const isSpaceOrTab = (c) => c === ' ' || c === '\t';

// s.replace(/^c+|c+$/g, '') for a single character c.
export function trimChar(s, c) {
  let start = 0;
  let end = s.length;
  while (start < end && s[start] === c) start++;
  while (end > start && s[end - 1] === c) end--;
  return s.slice(start, end);
}

// s.replace(/c+$/, '') for a single character c.
export function trimCharEnd(s, c) {
  let end = s.length;
  while (end > 0 && s[end - 1] === c) end--;
  return s.slice(0, end);
}

// s.replace(/<[^>]+>/g, replacement): every '<' up to the next '>', with at
// least one character between them.
export function replaceTags(s, replacement) {
  let out = '';
  let pos = 0;
  let from = 0;
  for (;;) {
    const open = s.indexOf('<', from);
    if (open < 0) break;
    const close = s.indexOf('>', open + 1);
    if (close < 0) break; // no later '<' has a '>' after it either
    if (close === open + 1) { from = open + 1; continue; } // '<>'
    out += s.slice(pos, open) + replacement;
    pos = from = close + 1;
  }
  return out + s.slice(pos);
}

// Markdown links and images, keeping the bracketed text:
//   image:     s.replace(/!\[([^\]]*)]\([^)]*\)/g, '$1')
//   inline:    s.replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
//   reference: s.replace(/\[([^\]]+)]\[[^\]]*]/g, '$1')
const LINK_SHAPES = {
  image: { lead: '![', minText: 0, open: '(', close: ')' },
  inline: { lead: '[', minText: 1, open: '(', close: ')' },
  reference: { lead: '[', minText: 1, open: '[', close: ']' },
};

export function unwrapLinks(s, shape) {
  const { lead, minText, open, close } = LINK_SHAPES[shape];
  let out = '';
  let pos = 0;
  let from = 0;
  for (;;) {
    const start = s.indexOf(lead, from);
    if (start < 0) break;
    const textStart = start + lead.length;
    const textEnd = s.indexOf(']', textStart);
    if (textEnd < 0) break; // no later start has a ']' after it either
    if (textEnd - textStart < minText) { from = start + 1; continue; }
    // Every start before textEnd stops at this same ']', so when the
    // character after it is wrong they all fail: resume past it.
    if (s[textEnd + 1] !== open) { from = textEnd + 1; continue; }
    const end = s.indexOf(close, textEnd + 2);
    if (end < 0) break; // later starts need a closer further on still
    out += s.slice(pos, start) + s.slice(textStart, textEnd);
    pos = from = end + 1;
  }
  return out + s.slice(pos);
}

// Splits on line terminators and keeps them: even indexes are lines, odd
// indexes the terminator after each. Lines are what a multiline ^ and $
// delimit.
export function splitLinesKeep(s) {
  return s.split(/(\r|\n|\u2028|\u2029)/);
}

// s.replace(/[ \t]*\|[ \t]*/g, ' '): each '|' together with the spaces and
// tabs on both sides of it becomes one space.
export function spacePipes(s) {
  let out = '';
  let pos = 0;
  for (let bar = s.indexOf('|'); bar >= 0; bar = s.indexOf('|', pos)) {
    let start = bar;
    while (start > pos && isSpaceOrTab(s[start - 1])) start--;
    let end = bar + 1;
    while (end < s.length && isSpaceOrTab(s[end])) end++;
    out += `${s.slice(pos, start)} `;
    pos = end;
  }
  return out + s.slice(pos);
}

// One line (no terminator) with line.replace(/^[ \t]+|[ \t]+$/g, '').
export function trimSpaceTab(line) {
  let start = 0;
  let end = line.length;
  while (start < end && isSpaceOrTab(line[start])) start++;
  while (end > start && isSpaceOrTab(line[end - 1])) end--;
  return line.slice(start, end);
}

// The url= of a meta refresh tag's content attribute: what
// /content\s*=\s*["']?[^"'>]*?url\s*=\s*([^"'>\s;]+)/i captures, without
// rescanning the same unquoted stretch from every 'content' in it.
export function refreshContentUrl(tag) {
  const contentRe = /content\s*=\s*["']?/gi;
  const urlRe = /url\s*=\s*([^"'>\s;]+)/i;
  let failedUntil = -1;
  for (let m = contentRe.exec(tag); m; m = contentRe.exec(tag)) {
    const start = contentRe.lastIndex;
    if (start <= failedUntil) continue; // a suffix of a stretch with no url=
    let end = start;
    while (end < tag.length && !`"'>`.includes(tag[end])) end++;
    const url = urlRe.exec(tag.slice(start, end));
    if (url) return url[1];
    failedUntil = end;
  }
  return null;
}

// [field, value] as /^([A-Za-z-]+)\s*:\s*(.*)$/ captures them. The value
// starts after the colon's whitespace and may not hold a line terminator.
export function robotsField(line) {
  const field = /^([A-Za-z-]+)\s*:/.exec(line);
  if (!field) return null;
  let start = field[0].length;
  while (start < line.length && isWhitespace(line[start])) start++;
  for (let i = start; i < line.length; i++) if (isLineTerminator(line[i])) return null;
  return [field[1], line.slice(start)];
}

// text.match(/^#\s+(.+)$/m)?.[1]: the first line that starts with '#' and
// whitespace. The whitespace may run across lines; what follows it runs to
// the end of its line. When only whitespace follows, the expression gives
// back the last character of it that is not a line terminator.
export function firstAtxTitle(text) {
  for (let hash = text.indexOf('#'); hash >= 0; hash = text.indexOf('#', hash + 1)) {
    if (hash > 0 && !isLineTerminator(text[hash - 1])) continue;
    let end = hash + 1;
    while (end < text.length && isWhitespace(text[end])) end++;
    if (end === hash + 1) continue;
    if (end < text.length) {
      let stop = end;
      while (stop < text.length && !isLineTerminator(text[stop])) stop++;
      return text.slice(end, stop);
    }
    for (let last = text.length - 1; last >= hash + 2; last--) {
      if (!isLineTerminator(text[last])) return text[last];
    }
    return undefined; // only whitespace remains, so no later '#' either
  }
  return undefined;
}

// An ATX heading line as /^(#{1,6})\s+(.+?)\s*$/ reads it: { depth, text }
// with text trimmed, or null. The text may not hold a line terminator;
// when only whitespace follows the hashes, the expression gives back the
// last character of it that is not a line terminator.
export function atxHeading(line) {
  let depth = 0;
  while (line[depth] === '#') depth++;
  if (depth === 0 || depth > 6 || !isWhitespace(line[depth])) return null;
  let start = depth;
  while (start < line.length && isWhitespace(line[start])) start++;
  if (start === line.length) {
    for (let last = line.length - 1; last >= depth + 1; last--) {
      if (!isLineTerminator(line[last])) return { depth, text: line[last] };
    }
    return null;
  }
  let end = line.length;
  while (isWhitespace(line[end - 1])) end--;
  for (let i = start; i < end; i++) if (isLineTerminator(line[i])) return null;
  return { depth, text: line.slice(start, end) };
}

// heading.replace(/\s+#+\s*$/, ''): a closing run of hashes, with the
// whitespace before it and after it.
export function stripClosingHashes(heading) {
  let end = heading.length;
  while (isWhitespace(heading[end - 1])) end--;
  let hashes = end;
  while (heading[hashes - 1] === '#') hashes--;
  if (hashes === end || !isWhitespace(heading[hashes - 1])) return heading;
  let start = hashes - 1;
  while (isWhitespace(heading[start - 1])) start--;
  return heading.slice(0, start);
}

// heading.match(/\s*\{#([A-Za-z0-9_-]+)\}\s*$/) as { index, id }, or null.
export function customIdSuffix(heading) {
  let end = heading.length;
  while (isWhitespace(heading[end - 1])) end--;
  if (heading[end - 1] !== '}') return null;
  let idStart = end - 1;
  while (idStart > 0 && /[A-Za-z0-9_-]/.test(heading[idStart - 1])) idStart--;
  if (idStart === end - 1 || heading[idStart - 1] !== '#' || heading[idStart - 2] !== '{') return null;
  let index = idStart - 2;
  while (isWhitespace(heading[index - 1])) index--;
  return { index, id: heading.slice(idStart, end - 1) };
}

// title.split(/\s*[|·–—-]\s+/)[0]: the title up to the first separator
// that has whitespace after it ("Install · Astro Docs" gives "Install").
export function titleBeforeSiteName(title) {
  for (let i = 0; i + 1 < title.length; i++) {
    if ('|·–—-'.includes(title[i]) && isWhitespace(title[i + 1])) {
      let start = i;
      while (isWhitespace(title[start - 1])) start--;
      return title.slice(0, start);
    }
  }
  return title;
}

// Applies fn to each line (the text between line terminators).
export function mapLines(text, fn) {
  const parts = splitLinesKeep(text);
  for (let i = 0; i < parts.length; i += 2) parts[i] = fn(parts[i]);
  return parts.join('');
}

// line.replace(/[ \t]+#+[ \t]*$/, ''): an ATX heading's closing hashes.
export function stripClosingHashLine(line) {
  let end = line.length;
  while (isSpaceOrTab(line[end - 1])) end--;
  let hashes = end;
  while (line[hashes - 1] === '#') hashes--;
  if (hashes === end || !isSpaceOrTab(line[hashes - 1])) return line;
  let start = hashes - 1;
  while (isSpaceOrTab(line[start - 1])) start--;
  return line.slice(0, start);
}

// text.replace(/^[ \t]*\|?(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*:?-*:?[ \t]*\|?[ \t]*\n/gm, ''):
// drops each table separator row (|---|:--:|) that ends in '\n'.
export function dropTableSeparatorRows(text) {
  const parts = splitLinesKeep(text);
  let out = '';
  for (let i = 0; i < parts.length; i += 2) {
    const after = parts[i + 1] ?? '';
    if (after === '\n' && isTableSeparatorRow(parts[i])) continue;
    out += parts[i] + after;
  }
  return out;
}

// A table cell that /^[ \t]*:?-+:?[ \t]*$/ matches (minDashes 1) or
// /^[ \t]*:?-*:?[ \t]*$/ (minDashes 0): an optional colon on each side of
// a run of dashes, spaces and tabs around.
function isDashCell(cell, minDashes) {
  let s = trimSpaceTab(cell);
  if (s[0] === ':') s = s.slice(1);
  if (s.at(-1) === ':') s = s.slice(0, -1);
  if (s.length < minDashes) return false;
  for (const c of s) if (c !== '-') return false;
  return true;
}

// Cells of dashes, each closed by '|', after an optional leading '|'; then
// an optional last cell (its dashes optional) and an optional closing '|'.
export function isTableSeparatorRow(line) {
  let row = trimSpaceTab(line);
  if (row[0] === '|') row = row.slice(1);
  const cells = row.split('|');
  if (cells.length < 2) return false;
  const tail = cells.pop();
  if (cells.every((cell) => isDashCell(cell, 1))) return isDashCell(tail, 0);
  // The closing '|' ends the optional last cell rather than a dash cell.
  return tail === '' && cells.length >= 2
    && cells.slice(0, -1).every((cell) => isDashCell(cell, 1))
    && isDashCell(cells.at(-1), 0);
}
