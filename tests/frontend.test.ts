import test from 'node:test';
import assert from 'node:assert/strict';
import { infer, encodeNpy, BATCH_EPOCHS, listModels, checkServer, type ModelSpec } from '../frontend/src/inference.ts';
import { parseNpy, parseResults, makeReport } from '../frontend/src/data.ts';

const cnn: ModelSpec = { key: 'cnn-source', id: 'cnn:v1', name: 'CNN', description: 'CNN', note: '', engine: 'onnxruntime-cpu', max_batch_epochs: 64, smoke_test_only: false, available: true };
const dino: ModelSpec = { ...cnn, key: 'slarda-dinov3-quick', id: 'dino:v1', name: 'DINOv3 QuickRun', engine: 'onnxruntime-cpu', max_batch_epochs: 4, smoke_test_only: true };
function reply(model: ModelSpec, epochs: number) {
  return { schema_version: 1, model_id: model.id, model_key: model.key, model: model.name,
    engine: model.engine, smoke_test_only: model.smoke_test_only, epoch_seconds: 30,
    sampling_rate_hz: 100, channel: 'EEG Fpz-Cz', epochs, flat: 0,
    predictions: Array(epochs).fill(2), confidences: Array(epochs).fill(0.7) };
}

test('encoded upload preserves raw EEG values and stays below 1 MiB', () => {
  const samples = new Float32Array(BATCH_EPOCHS * 3000);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i) * 1e-5;
  const body = encodeNpy(samples);
  assert.ok(body.byteLength < 1024 * 1024);
  assert.deepEqual(parseNpy(body), samples);
  assert.throws(() => encodeNpy(new Float32Array(3001)));
});

for (const model of [cnn, dino]) {
  test(`${model.key}: route, chunk size, order, progress and result identity`, async () => {
    const count = model === cnn ? 400 : 11;
    const samples = Float32Array.from({ length: count * 3000 }, (_, i) => Math.sin(i) * 1e-5);
    if (model === cnn) assert.ok(samples.byteLength > 4_500_000);
    const original = globalThis.fetch;
    let processed = 0, calls = 0;
    globalThis.fetch = async (url, options) => {
      assert.equal(url, `/api/analyze?model=${model.key}`);
      assert.equal(options?.method, 'POST');
      const body = options?.body as ArrayBuffer;
      assert.ok(body.byteLength < 1024 * 1024);
      const chunk = parseNpy(body), epochs = chunk.length / 3000;
      assert.ok(epochs <= model.max_batch_epochs);
      assert.deepEqual(chunk, samples.slice(processed * 3000, (processed + epochs) * 3000));
      processed += epochs; calls++;
      return Response.json(reply(model, epochs));
    };
    try {
      const progress: number[] = [];
      const result = await infer(samples, model, n => progress.push(n));
      assert.equal(calls, Math.ceil(count / model.max_batch_epochs));
      assert.equal(processed, count);
      assert.equal(result.predictions.length, count);
      assert.equal(result.model_id, model.id);
      assert.equal(result.smoke_test_only, model.smoke_test_only);
      assert.equal(progress.at(-1), 100);
    } finally { globalThis.fetch = original; }
  });
}

test('rejects a different model, changed version or smoke flag mid-session', async () => {
  const original = globalThis.fetch;
  try {
    for (const wrong of [reply(cnn, 1), { ...reply(dino, 1), model_id: 'dino:v2' }, { ...reply(dino, 1), smoke_test_only: false }]) {
      globalThis.fetch = async () => Response.json(wrong);
      await assert.rejects(infer(new Float32Array(3000), dino, () => {}), /phiên bản model/);
    }
    let calls = 0;
    globalThis.fetch = async () => Response.json(++calls === 1 ? reply(dino, 4) : reply(cnn, 1));
    await assert.rejects(infer(new Float32Array(5 * 3000), dino, () => {}), /phiên bản model/);
    const controller = new AbortController(); controller.abort();
    await assert.rejects(infer(new Float32Array(3000), dino, () => {}, controller.signal), { name: 'AbortError' });
  } finally { globalThis.fetch = original; }
});

test('loads the server model catalog and checks the selected health endpoint', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async url => {
    if (url === '/api/models') return Response.json({ default_model: dino.key, models: [dino, cnn] });
    assert.equal(url, `/api/health?model=${dino.key}`);
    return Response.json({ status: 'ok', model_id: dino.id });
  };
  try {
    const catalog = await listModels();
    assert.equal(catalog.default_model, dino.key);
    await checkServer(catalog.models[0]);
    globalThis.fetch = async () => Response.json({ default_model: dino.key, models: [{ ...dino, max_batch_epochs: 999 }] });
    await assert.rejects(listModels(), /không hợp lệ/);
  } finally { globalThis.fetch = original; }
});

test('report export and history import retain the chosen model and QuickRun warning', () => {
  const results = parseResults(reply(dino, 2));
  const report = makeReport({ id: 'test', name: 'Demo', date: '', note: '', epochs: 2, results });
  const imported = parseResults(JSON.parse(JSON.stringify(report)));
  assert.equal(imported.model_id, dino.id);
  assert.equal(imported.model_key, dino.key);
  assert.equal(imported.engine, dino.engine);
  assert.equal(imported.smoke_test_only, true);
});
