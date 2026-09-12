// Query encoder: corpus-distilled PSTU student model, bundled with the Worker.
// Queries embed in-process with zero outbound calls, so this Worker needs no
// AI binding and `wrangler dev` runs fully local. The bundled abstention
// scorer gates low-confidence matches when it was calibrated at build time.

import { loadStudentModel, embedTextWithStudent, computeMatchQuality } from './student-embedder.mjs';
import STUDENT_MODEL_ASSET from './assets/student-model.bin';
import STUDENT_ABSTENTION from './assets/student-abstention.json';

let studentModel = null;

function studentModelBytes() {
  const asset = STUDENT_MODEL_ASSET;
  if (asset instanceof ArrayBuffer) return new Uint8Array(asset);
  if (ArrayBuffer.isView(asset)) return new Uint8Array(asset.buffer, asset.byteOffset, asset.byteLength);
  throw Object.assign(new Error('Student model was not bundled as binary data'), { code: 'MANIFEST_MISMATCH' });
}

function getStudentModel() {
  if (!studentModel) {
    try {
      studentModel = loadStudentModel(studentModelBytes());
    } catch (error) {
      throw Object.assign(new Error(`Student model failed to load: ${error.message}`), { code: 'MANIFEST_MISMATCH' });
    }
  }
  return studentModel;
}

export function assertEncoderManifest(manifest) {
  if (manifest.encoder?.mode !== 'student' || manifest.encoder?.format !== 'pstu') {
    throw Object.assign(new Error('Manifest encoder mode mismatch (expected student/pstu)'), { code: 'MANIFEST_MISMATCH' });
  }
  const model = getStudentModel();
  if (model.outputDim !== (manifest.dims || manifest.dim)) {
    throw Object.assign(new Error(`Student model dimension mismatch (${model.outputDim} !== ${manifest.dims})`), { code: 'MANIFEST_MISMATCH' });
  }
}

export async function embedQuery(query, manifest) {
  const prefixed = `${manifest.prefixPolicy.query}${query}`;
  const embedded = embedTextWithStudent(prefixed, getStudentModel());
  return { vector: embedded.vector, embedded };
}

export function scoreHits(hits, embedded) {
  if (!embedded) return null;
  return computeMatchQuality(hits, embedded, STUDENT_ABSTENTION || null);
}

export function encoderInfo(env, manifest) {
  return {
    model: manifest?.model || 'pikelet-distilled-student',
    encoder_mode: 'student',
    abstention_calibrated: !!STUDENT_ABSTENTION,
  };
}
