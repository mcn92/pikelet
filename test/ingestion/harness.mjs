// Shared check/summary state for the ingestion conformance suite. Each test
// module here imports check() and section(); run.mjs imports the modules by
// name for their side effects and exits on the shared counters.
let passed = 0;
let failed = 0;

export function section(title) {
  console.log(`\n${title}`);
}

export function check(label, cond, detail = '') {
  if (cond) {
    passed++;
    console.log(`  ok: ${label}`);
  } else {
    failed++;
    console.log(`  FAIL: ${label}${detail ? ` — ${detail}` : ''}`);
  }
}

export function counts() {
  return { passed, failed };
}
