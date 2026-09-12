'use strict';

const PIKELET_ERROR_CODES = Object.freeze({
    INVALID_ARGUMENT: 'INVALID_ARGUMENT',
    DIMENSION_MISMATCH: 'DIMENSION_MISMATCH',
    INVALID_VECTOR: 'INVALID_VECTOR',
    INDEX_FULL: 'INDEX_FULL',
    INDEX_DISPOSED: 'INDEX_DISPOSED',
    COMPACTION_REQUIRED: 'COMPACTION_REQUIRED',
    SNAPSHOT_INVALID: 'SNAPSHOT_INVALID',
    SNAPSHOT_CONFIG_MISMATCH: 'SNAPSHOT_CONFIG_MISMATCH',
    SNAPSHOT_CAPACITY_EXCEEDED: 'SNAPSHOT_CAPACITY_EXCEEDED',
    WASM_LOAD_FAILED: 'WASM_LOAD_FAILED',
    WASM_ALLOCATION_FAILED: 'WASM_ALLOCATION_FAILED',
    FILE_IO_FAILED: 'FILE_IO_FAILED',
    PARSE_FAILED: 'PARSE_FAILED',
    INTERNAL_INVARIANT: 'INTERNAL_INVARIANT',
    // Reserved for the deferred shared-runtime API.
    INDEX_LIMIT: 'INDEX_LIMIT',
});

class PikeletError extends Error {
    constructor(code, message, details, cause) {
        super(message);
        this.name = 'PikeletError';
        this.code = code;
        if (details !== undefined) this.details = details;
        if (cause !== undefined) this.cause = cause;
    }
}

function pikeletError(code, message, details, cause) {
    return new PikeletError(code, message, details, cause);
}

module.exports = {
    PikeletError,
    PIKELET_ERROR_CODES,
    pikeletError,
};
