// The Node releases pikelet supports, mirrored by package.json `engines`
// (npm only warns on an engines mismatch, and an existing install never
// re-checks it, so the CLI says so itself on every start).
//
// Lines below these patch releases ship a fetch whose response decoder runs
// on any body a server labels with a Content-Encoding. The CLI's own reads
// no longer go through it (pikelet-wasm/complete/node-fetch.mjs), but a
// dependency or plugin loaded into the same process still might, and 18.x,
// 20.x, 23.x and 25.x are end-of-life with no fix.

const SUPPORTED = [
  { major: 22, min: [22, 23, 4] },
  { major: 24, min: [24, 22, 1] },
  { major: 26, min: [26, 12, 1], andLater: true },
];

export const SUPPORTED_NODE_RANGE = '>=22.23.4 <23 || >=24.22.1 <25 || >=26.12.1';

function atLeast(version, min) {
  for (let i = 0; i < 3; i++) {
    if (version[i] !== min[i]) return version[i] > min[i];
  }
  return true;
}

export function isSupportedNode(versionString) {
  const version = String(versionString).replace(/^v/, '').split('.').map((n) => Number.parseInt(n, 10) || 0);
  return SUPPORTED.some((line) => (line.andLater ? version[0] >= line.major : version[0] === line.major)
    && atLeast(version, line.min));
}

export function nodeSupportWarning(versionString = process.versions.node) {
  if (isSupportedNode(versionString)) return null;
  // One line: it prints on every run. The reason is in the README and above.
  return `warning: Node ${versionString} is unsupported; use ${SUPPORTED_NODE_RANGE} (PIKELET_NO_NODE_WARNING=1 hides this)`;
}
