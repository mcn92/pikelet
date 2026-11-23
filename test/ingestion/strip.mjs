// stripMarkdown conformance: markup is removed by position, content is
// never damaged. Regression for record text that shipped with hyphens,
// minus signs, underscores, and operators deleted (uint8_float_hnsw.hpp
// became "uint8 float hnsw.hpp"; "range = vmax - vmin" became "range =
// vmax vmin"). The stored record is evidence; it must read as the source.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { check, section } from './harness.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const { stripMarkdown } = await import(pathToFileURL(path.resolve(here, '..', '..', 'packages', 'pikelet', 'src', 'ingest.mjs')).href);

section('stripMarkdown: content survives');
const keep = [
  ['snake_case identifier', 'call uint8_float_hnsw.hpp:213 and my_var', 'call uint8_float_hnsw.hpp:213 and my_var'],
  ['hyphenated words', 'Row-wise affine, degenerate-vector guard', 'Row-wise affine, degenerate-vector guard'],
  ['minus sign and exponent', 'range = vmax - vmin; if (range < 1e-30)', 'range = vmax - vmin; if (range < 1e-30)'],
  ['operators', 'x = a * b - c; if (a > b && q[0..D-1])', 'x = a * b - c; if (a > b && q[0..D-1])'],
  ['intra-word underscore is not emphasis', 'not_this_one stays', 'not_this_one stays'],
  ['fenced code verbatim, indentation kept', '```python\ndef f():\n    if x:\n        return a_b - 1  # not a heading\n```', 'def f():\n    if x:\n        return a_b - 1  # not a heading'],
  ['inline code contents kept', 'use `offset + scale * q[d]` here', 'use offset + scale * q[d] here'],
];
for (const [label, input, want] of keep) {
  const got = stripMarkdown(input);
  check(label, got === want, `got ${JSON.stringify(got)}`);
}

section('stripMarkdown: markup is removed');
const strip = [
  ['ATX heading hashes', '## Title ##\nbody', 'Title\nbody'],
  ['blockquote prefixes, nested', '> a\n>> b', 'a\nb'],
  ['list bullets', '- one\n* two\n+ three', 'one\ntwo\nthree'],
  ['horizontal rule', 'a\n\n---\n\nb', 'a\n\nb'],
  ['boundary emphasis', '**bold** and _em_ and ~~gone~~', 'bold and em and gone'],
  ['links keep text, drop url', 'see [the docs](http://x) and ![alt](i.png)', 'see the docs and alt'],
  ['table scaffolding', '| a | b |\n|---|---|\n| 1 | 2 |', 'a b\n1 2'],
  ['backslash escapes', 'a \\* b \\_ c', 'a * b _ c'],
];
for (const [label, input, want] of strip) {
  const got = stripMarkdown(input);
  check(label, got === want, `got ${JSON.stringify(got)}`);
}

section('stripMarkdown: code protection is linear and CommonMark-shaped');
const fences = [
  // A fenced block is restored as its own paragraph (blank line after it).
  ['closing fence may be longer than the opener', '````\nx **y**\n`````\nafter **z**', 'x **y**\n\nafter z'],
  ['a shorter fence line is body, not a closer', '````\n```\n# still code\n````\n# heading', '```\n# still code\n\nheading'],
  ['unclosed fence runs to the end of the document', 'a\n```\n# code\n**more**', 'a\n\n# code\n**more**'],
  ['tilde and backtick fences do not close each other', '~~~\n```\n# code\n~~~\n# heading', '```\n# code\n\nheading'],
  ['double-tick span quotes a single backtick', 'use ``a ` b`` now', 'use a ` b now'],
  ['span body may contain runs of other lengths', 'run `` a `b` c `` now', 'run a `b` c now'],
  ['a longer closing run does not close a shorter opener', 'x `a``` y', 'x `a``` y'],
  ['a shorter closing run does not close a longer opener', 'x ```a` y', 'x ```a` y'],
  ['a span does not cross a line break', 'a ` b\n` c', 'a ` b\n` c'],
  ['prose numbers between spaces are not stash markers', 'take 7 of `x` and 0 more', 'take 7 of x and 0 more'],
  ['a NUL-bracketed number in the input is not a stash marker', 'a `x` b \u00000\u0000 c', 'a x b 0 c'],
  ['NULs in the input are dropped', 'one\u0000two', 'onetwo'],
];
for (const [label, input, want] of fences) {
  const got = stripMarkdown(input);
  check(label, got === want, `got ${JSON.stringify(got)}`);
}
{
  // Regression for the quadratic regexes: these inputs took seconds to
  // minutes before; the scanners finish them in milliseconds.
  const hostile = [
    ['40k backticks then 40k letters (inline span backtracking)', '`'.repeat(40000) + 'a'.repeat(40000)],
    ['40k-backtick fence line with no closer (fence backtracking)', '`'.repeat(40000) + '\n' + 'a\n'.repeat(20000)],
    ['20k unclosed opener lines', '```x\n'.repeat(20000)],
  ];
  for (const [label, input] of hostile) {
    const t0 = performance.now();
    stripMarkdown(input);
    const ms = performance.now() - t0;
    check(`${label} finishes under 500 ms`, ms < 500, `${ms.toFixed(0)} ms`);
  }
}
