// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Josh Hughes
// Local Model Access (LMA) - created by artist Josh Hughes. The core, as plain ES modules with no dependencies and no build step:
// copy the lma/ folder into a site and import from it. See SPEC.md.
export { readHeader, readModel, sniff, tensorMap, FormatError, GGML_TYPES, ST_DTYPES } from './formats.js';
export { inspectFile, inspectFiles, candidates, partsOf, verifyModel, verifyFile, fileSha256, hashFile, fingerprint, forget, cacheKey, verifyChunk } from './verify.js';
export { Loader } from './loader.js';
export { makePicker, storageSnapshot, describeFile } from './picker.js';
export { Sha256, sha256hex } from './sha256.js';
export * as pocketflux from './ext/pocketflux.js';   // the pocketflux.pack/1 extension: reference dequantisers (docs/QUANT-q4g32a.md)
export const LMA_SPEC_VERSION = '1.0';
