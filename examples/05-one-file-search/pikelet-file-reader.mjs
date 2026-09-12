// The one-file reader now lives in the published package — this example
// imports it the way any consumer does:
//
//   import { openPikeletFile } from 'pikelet-wasm/complete';
//
// (Relative here because the example lives inside the pikelet repo itself.)
export * from '../../complete/index.mjs';
