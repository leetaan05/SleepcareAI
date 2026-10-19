import type { Results } from './data';

export const MODEL_ID = 'sleep-cnn-source:b52c50097fe9';
export const BATCH_EPOCHS = 64;
const SAMPLES_PER_EPOCH = 3000;
const REQUEST_TIMEOUT_MS = 65_000;

// The browser only packages raw EEG. Normalization and AI inference run in Python.
export function encodeNpy(samples: Float32Array): ArrayBuffer {
  const epochs = samples.length / SAMPLES_PER_EPOCH;
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > BATCH_EPOCHS) {
    throw new Error('Mỗi phần EEG phải có từ 1 đến 64 epoch.');
  }
  if (!samples.every(Number.isFinite)) throw new Error('EEG chứa NaN/Inf.');
  const rawHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + epochs + ", 1, 3000), }";
  const padding = (64 - (10 + rawHeader.length + 1) % 64) % 64;
  const header = new TextEncoder().encode(rawHeader + ' '.repeat(padding) + '\n');
  const buffer = new ArrayBuffer(10 + header.length + samples.length * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  const view = new DataView(buffer);
  view.setUint16(8, header.length, true);
  bytes.set(header, 10);
  for (let i = 0; i < samples.length; i++) view.setFloat32(10 + header.length + i * 4, samples[i], true);
  return buffer;
}

type BatchResult = Results & { schema_version: number; model_id: string; flat: number; epochs: number };

function validateBatch(value: unknown, epochs: number): BatchResult {
  const r = value as Partial<BatchResult> | null;
  if (!r || r.schema_version !== 1 || r.model_id !== MODEL_ID || r.epoch_seconds !== 30 ||
      typeof r.model !== 'string' || r.epochs !== epochs ||
      !Number.isInteger(r.flat) || r.flat! < 0 || r.flat! > epochs ||
      !Array.isArray(r.predictions) || r.predictions.length !== epochs ||
      !r.predictions.every(n => Number.isInteger(n) && n >= 0 && n < 5) ||
      !Array.isArray(r.confidences) || r.confidences.length !== epochs ||
      !r.confidences.every(n => Number.isFinite(n) && n >= 0 && n <= 1)) {
    throw new Error('Kết quả API không hợp lệ hoặc phiên bản model đã thay đổi. Hãy tải lại trang.');
  }
  return r as BatchResult;
}

class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}

async function requestJson(url: string, options: RequestInit, signal?: AbortSignal): Promise<unknown> {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener('abort', cancel, { once: true });
  if (signal?.aborted) controller.abort();
  const timeout = setTimeout(cancel, REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal, cache: 'no-store' });
    const payload = await response.json().catch(() => null) as { detail?: unknown } | null;
    if (!response.ok) {
      const fallback = response.status === 429
        ? 'API đã đạt giới hạn lượt xử lý. Hãy thử lại sau.'
        : 'Không kết nối được máy chủ AI. Hãy thử lại sau.';
      throw new ApiError(typeof payload?.detail === 'string' ? payload.detail : fallback, response.status);
    }
    if (payload === null) throw new Error('API không trả về JSON. Kiểm tra cả frontend và backend đã được deploy.');
    return payload;
  } catch (error) {
    if (signal?.aborted) throw new DOMException('Đã dừng phân tích.', 'AbortError');
    if (controller.signal.aborted) throw new Error('Máy chủ phản hồi quá lâu. Hãy thử phân tích lại.');
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener('abort', cancel);
  }
}

export async function checkServer(signal?: AbortSignal): Promise<void> {
  const health = await requestJson('/api/health', { headers: { Accept: 'application/json' } }, signal) as
    { status?: unknown; model_id?: unknown };
  if (health?.status !== 'ok' || health.model_id !== MODEL_ID) {
    throw new Error('Máy chủ AI chưa sẵn sàng hoặc model không khớp với giao diện.');
  }
}

export async function infer(
  samples: Float32Array,
  progress: (n: number) => void,
  signal?: AbortSignal,
): Promise<Results & { flat: number }> {
  const epochs = samples.length / SAMPLES_PER_EPOCH;
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 2000) {
    throw new Error('EEG cần có từ 1 đến 2.000 epoch, mỗi epoch 3.000 mẫu.');
  }
  const predictions: number[] = [], confidences: number[] = [];
  let flat = 0, model = '';
  progress(0);
  for (let start = 0; start < epochs; start += BATCH_EPOCHS) {
    if (signal?.aborted) throw new DOMException('Đã dừng phân tích.', 'AbortError');
    const size = Math.min(BATCH_EPOCHS, epochs - start);
    const body = encodeNpy(samples.subarray(start * SAMPLES_PER_EPOCH, (start + size) * SAMPLES_PER_EPOCH));
    let payload: unknown;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        payload = await requestJson('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream', Accept: 'application/json' },
          body,
        }, signal);
        break;
      } catch (error) {
        const retry = error instanceof ApiError && [502, 503, 504].includes(error.status);
        if (attempt === 1 || !retry || signal?.aborted) throw error;
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    const result = validateBatch(payload, size);
    if (model && model !== result.model) throw new Error('Model đã thay đổi giữa phiên. Hãy tải lại trang.');
    predictions.push(...result.predictions);
    confidences.push(...result.confidences);
    flat += result.flat;
    model = result.model;
    progress(Math.round((start + size) / epochs * 100));
  }
  return { predictions, confidences, flat, model, epoch_seconds: 30 };
}
