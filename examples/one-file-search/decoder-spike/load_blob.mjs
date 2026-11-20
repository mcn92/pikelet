// Reads a weight blob straight into a module's memory, in 64 MB chunks, so
// no JavaScript copy of the file exists alongside it (a 1.9 GB blob read with
// readFileSync and then copied in held both). Returns the blob's address
// (unsigned: past 2 GB, pointers must be read with >>> 0).
import { openSync, fstatSync, readSync, closeSync } from 'node:fs';

const CHUNK = 64 << 20;

export function loadBlob(mod, file) {
  const fd = openSync(file, 'r');
  try {
    const size = fstatSync(fd).size;
    const ptr = mod._malloc(size) >>> 0;
    if (!ptr) throw new Error(`malloc(${size}) failed for ${file}`);
    // HEAPU8 is replaced when memory grows, so it is read after the malloc.
    const heap = mod.HEAPU8;
    for (let off = 0; off < size;) {
      const n = readSync(fd, heap, ptr + off, Math.min(CHUNK, size - off), off);
      if (n <= 0) throw new Error(`short read of ${file} at ${off}`);
      off += n;
    }
    return { ptr, size };
  } finally {
    closeSync(fd);
  }
}
