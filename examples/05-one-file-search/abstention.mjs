// Abstention scoring, extracted from 03's worker.js and parameterized on
// the calibration model. Pure JS, no runtime dependencies — shared by the
// Node spike facade and the browser one-file reader. The scoreQuery import
// is the only coupling to the encoder module (they form one
// query-interpretation unit; see spec/COMPLETE_PROFILE.md section 3.6).

import { scoreQuery } from '../legacy/03-edge-docs-search/student-embedder.mjs';

export function buildKnownBucketTables(model) {
    const word = new Set();
    let maxIdf = 0;
    for (const entry of model?.wordBuckets || []) {
        if (!Array.isArray(entry) || entry.length < 2) continue;
        const bucket = Number(entry[0]);
        const idf = Number(entry[1]);
        if (Number.isInteger(bucket) && bucket >= 0) {
            word.add(bucket);
            if (idf > maxIdf) maxIdf = idf;
        }
    }
    const char = new Set();
    for (const value of model?.charBuckets || []) {
        const bucket = Number(value);
        if (Number.isInteger(bucket) && bucket >= 0) char.add(bucket);
    }
    return { word, char, maxIdf: maxIdf || 255 };
}

export function computeKnownFractions(features, model) {
    if (!model) return { known_word: 0, known_char: 0, n_feats: features.length };
    const tables = model._knownBucketTables || (model._knownBucketTables = buildKnownBucketTables(model));
    let wordKnown = 0;
    let wordTotal = 0;
    let charKnown = 0;
    let charTotal = 0;
    for (const feature of features) {
        if (feature.family === 'word') {
            wordTotal += 1;
            if (tables.word.has(feature.bucket)) wordKnown += 1;
        } else if (feature.family === 'char') {
            charTotal += 1;
            if (tables.char.has(feature.bucket)) charKnown += 1;
        }
    }
    return {
        known_word: wordTotal > 0 ? wordKnown / wordTotal : 0,
        known_char: charTotal > 0 ? charKnown / charTotal : 0,
        n_feats: features.length,
    };
}

export function computeHiddenProbe(embedded, model) {
    const probe = model?.hiddenProbe;
    if (!probe || !Array.isArray(probe.weights) || !embedded.hidden) return 0;
    let logit = Number(probe.bias) || 0;
    const limit = Math.min(probe.weights.length, embedded.hidden.length);
    for (let index = 0; index < limit; index++) {
        logit += (Number(probe.weights[index]) || 0) * embedded.hidden[index];
    }
    if (logit >= 0) {
        const z = Math.exp(-logit);
        return 1 / (1 + z);
    }
    const z = Math.exp(logit);
    return z / (1 + z);
}

export function computeMatchQuality(hits, embedded, model) {
    if (!model) return { match_quality: 'unscored' };
    const d0 = hits.length > 0 ? hits[0].distance : 1;
    const marginIndex = Math.min(4, hits.length - 1);
    const margin = marginIndex > 0 ? hits[marginIndex].distance - d0 : 0;
    const known = computeKnownFractions(embedded.features, model);
    const signals = {
        d0,
        margin,
        pre_norm: embedded.preNorm,
        known_word: known.known_word,
        known_char: known.known_char,
        hidden_probe: computeHiddenProbe(embedded, model),
        n_feats: known.n_feats,
    };
    return scoreQuery(signals, model);
}

export function computePreSearchAbstention(embedded, model) {
    if (!model) return null;
    const thresholds = model.thresholds || {};
    const minFeatures = Number.isInteger(thresholds.minFeatures) ? thresholds.minFeatures : 3;
    const preNormFloor = Number.isFinite(thresholds.preNormFloor) ? thresholds.preNormFloor : 0.4;
    if (embedded.features.length >= minFeatures && embedded.preNorm >= preNormFloor) return null;
    const known = computeKnownFractions(embedded.features, model);
    return scoreQuery({
        d0: 1,
        margin: 0,
        pre_norm: embedded.preNorm,
        known_word: known.known_word,
        known_char: known.known_char,
        hidden_probe: computeHiddenProbe(embedded, model),
        n_feats: known.n_feats,
    }, model);
}
