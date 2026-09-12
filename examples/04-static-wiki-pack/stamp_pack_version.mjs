// Stamp pack-manifest.json with packVersion: a content hash over every pack
// asset served under the versioned /pack/vXXXX/ URL (see DEPLOY.md §3). The
// hash must change whenever any of those bytes change, or immutable-cached
// clients would keep stale assets forever — which is why calibrate_abstention
// re-stamps after writing the abstention assets.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const VERSIONED_ASSETS = [
    'wiki.pikelet-sketch',
    'corpus.bin',
    'corpus-offsets.u32',
    'wiki-abstention.json',
    'wiki-vocab.bloom',
];

export function stampPackVersion(dataDir) {
    const manifestPath = path.join(dataDir, 'pack-manifest.json');
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const hash = createHash('sha256');
    const hashed = [];
    for (const name of VERSIONED_ASSETS) {
        const assetPath = path.join(dataDir, name);
        if (!fs.existsSync(assetPath)) continue;
        hash.update(name);
        hash.update(fs.readFileSync(assetPath));
        hashed.push(name);
    }
    const packVersion = `v${hash.digest('hex').slice(0, 12)}`;
    manifest.packVersion = packVersion;
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
    return { packVersion, hashed };
}
