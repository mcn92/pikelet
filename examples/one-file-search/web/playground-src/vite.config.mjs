import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// The answer page's model runs the decoder spike's Llama kernel in a worker.
// Those files are emitted to llm/ as they are, not bundled: the threaded
// build starts its threads from new URL('llama-mt.web.mjs', import.meta.url)
// and fetches its .wasm beside itself, so the files must keep their names
// and sit together.
const SPIKE = resolve(__dirname, '..', '..', 'decoder-spike');
const LLM_FILES = [
    [resolve(__dirname, 'llm-worker.mjs'), 'worker.mjs'],
    // Model sizes and SHA-256s (decoder-spike/model_manifest.mjs).
    [resolve(__dirname, 'llm-models.json'), 'models.json'],
    [resolve(SPIKE, 'bpe.mjs'), 'bpe.mjs'],
    ...['llama-mt.web.mjs', 'llama-mt.web.wasm', 'llama.web.mjs', 'llama.web.wasm']
        .map((f) => [resolve(SPIKE, f), f]),
];

function llmFiles() {
    return {
        name: 'llm-files',
        generateBundle() {
            for (const [from, name] of LLM_FILES) {
                this.emitFile({ type: 'asset', fileName: `llm/${name}`, source: readFileSync(from) });
            }
        },
    };
}

export default defineConfig({
    root: resolve(__dirname),
    base: './',
    plugins: [llmFiles()],
    build: {
        target: 'es2022',
        outDir: resolve(__dirname, '..', 'playground-dist'),
        emptyOutDir: true,
        rollupOptions: {
            input: {
                index: resolve(__dirname, 'index.html'),
                answer: resolve(__dirname, 'answer.html'),
            },
        },
    },
});
