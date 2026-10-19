import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { infer, encodeNpy, BATCH_EPOCHS, MODEL_ID } from '../frontend/src/inference.ts';
import { parseNpy } from '../frontend/src/data.ts';

test('encoded client upload preserves raw EEG values and stays below 1 MiB', () => {
  const samples = new Float32Array(BATCH_EPOCHS * 3000);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i) * 1e-5;
  const body = encodeNpy(samples);
  assert.ok(body.byteLength < 1024 * 1024);
  assert.deepEqual(parseNpy(body), samples);
  assert.throws(() => encodeNpy(new Float32Array(3001)));
});

test('file larger than Vercel request cap is split in order and progress completes', async () => {
  const bytes = readFileSync(new URL('../frontend/public/samples/eeg-demo.npy', import.meta.url));
  const demo = parseNpy(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const samples = new Float32Array(400 * 3000);
  for (let i = 0; i < samples.length; i++) samples[i] = demo[i % demo.length];
  assert.ok(samples.byteLength > 4_500_000);
  const originalFetch = globalThis.fetch;
  let processed = 0, calls = 0;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/analyze');
    assert.equal(options?.method, 'POST');
    const body = options?.body as ArrayBuffer;
    assert.ok(body.byteLength < 1024 * 1024);
    const chunk = parseNpy(body), epochs = chunk.length / 3000;
    assert.deepEqual(chunk, samples.slice(processed * 3000, (processed + epochs) * 3000));
    processed += epochs; calls++;
    return Response.json({
      schema_version: 1, model_id: MODEL_ID, model: 'test-server', epoch_seconds: 30,
      epochs, flat: 0, predictions: Array(epochs).fill(2), confidences: Array(epochs).fill(0.7),
    });
  };
  try {
    const progress: number[] = [];
    const result = await infer(samples, n => progress.push(n));
    assert.equal(calls, 7);
    assert.equal(processed, 400);
    assert.equal(result.predictions.length, 400);
    assert.equal(progress.at(-1), 100);
  } finally { globalThis.fetch = originalFetch; }
});

test('rejects unexpected model versions and supports cancellation', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ schema_version: 1, model_id: 'different-model' });
  try {
    await assert.rejects(infer(new Float32Array(3000), () => {}), /phiên bản model/);
    const controller = new AbortController(); controller.abort();
    await assert.rejects(infer(new Float32Array(3000), () => {}, controller.signal), { name: 'AbortError' });
  } finally { globalThis.fetch = originalFetch; }
});
