import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
    base: './',
    build: {
        // getBigUint64 and top-level await need a modern floor; every
        // browser with WASM SIMD (the engine's own floor) clears it.
        target: 'es2022',
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                ablation: resolve(__dirname, 'ablation.html'),
            },
        },
    },
});
