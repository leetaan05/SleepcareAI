import type { Results } from './data';

export const BATCH_EPOCHS = 64;
const SAMPLES_PER_EPOCH = 3000;
const REQUEST_TIMEOUT_MS = 125_000;

export type ModelSpec = {
  key: string; id: string; name: string; description: string; note: string;
  engine: string; max_batch_epochs: number; smoke_test_only: boolean; available: boolean;
};

export async function listModels(signal?: AbortSignal): Promise<{ default_model: string; models: ModelSpec[] }> {
  const value = await requestJson('/api/models', { headers: { Accept: 'application/json' } }, signal) as
    { default_model?: unknown; models?: unknown };
  if (!value || typeof value.default_model !== 'string' || !Array.isArray(value.models) || !value.models.length ||
      !value.models.every((m: Partial<ModelSpec>) => m && typeof m.key === 'string' && /^[a-z0-9-]+$/.test(m.key) &&
        typeof m.id === 'string' && typeof m.name === 'string' && typeof m.description === 'string' &&
        typeof m.note === 'string' && typeof m.engine === 'string' && typeof m.available === 'boolean' &&
        typeof m.smoke_test_only === 'boolean' && Number.isInteger(m.max_batch_epochs) &&
        m.max_batch_epochs! >= 1 && m.max_batch_epochs! <= BATCH_EPOCHS) ||
      !value.models.some(m => m.key === value.default_model)) {
    throw new Error('Danh sách model từ máy chủ không hợp lệ. Hãy thử kết nối lại.');
  }
  return value as { default_model: string; models: ModelSpec[] };
}

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

function validateBatch(value: unknown, epochs: number, model: ModelSpec): BatchResult {
  const r = value as Partial<BatchResult> | null;
  if (!r || r.schema_version !== 1 || r.model_id !== model.id || r.model_key !== model.key ||
      r.engine !== model.engine || r.smoke_test_only !== model.smoke_test_only || r.epoch_seconds !== 30 ||
      r.model !== model.name || r.epochs !== epochs ||
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

export async function checkServer(model: ModelSpec, signal?: AbortSignal): Promise<void> {
  const health = await requestJson(`/api/health?model=${encodeURIComponent(model.key)}`, { headers: { Accept: 'application/json' } }, signal) as
    { status?: unknown; model_id?: unknown };
  if (health?.status !== 'ok' || health.model_id !== model.id) {
    throw new Error('Máy chủ AI chưa sẵn sàng hoặc model không khớp với giao diện.');
  }
}

export async function infer(
  samples: Float32Array,
  selectedModel: ModelSpec,
  progress: (n: number) => void,
  signal?: AbortSignal,
): Promise<Results & { flat: number }> {
  const epochs = samples.length / SAMPLES_PER_EPOCH;
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 2000) {
    throw new Error('EEG cần có từ 1 đến 2.000 epoch, mỗi epoch 3.000 mẫu.');
  }
  const predictions: number[] = [], confidences: number[] = [];
  if (!selectedModel.available) throw new Error('Model này chưa có trên máy chủ.');
  const batchSize = selectedModel.max_batch_epochs;
  if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > BATCH_EPOCHS) throw new Error('Giới hạn model không hợp lệ.');
  let flat = 0;
  progress(0);
  for (let start = 0; start < epochs; start += batchSize) {
    if (signal?.aborted) throw new DOMException('Đã dừng phân tích.', 'AbortError');
    const size = Math.min(batchSize, epochs - start);
    const body = encodeNpy(samples.subarray(start * SAMPLES_PER_EPOCH, (start + size) * SAMPLES_PER_EPOCH));
    let payload: unknown;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        payload = await requestJson(`/api/analyze?model=${encodeURIComponent(selectedModel.key)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream', Accept: 'application/json' },
          body,
        }, signal);
        break;
      } catch (error) {
        const retry = error instanceof ApiError && [429, 502, 503, 504].includes(error.status);
        if (attempt === 2 || !retry || signal?.aborted) throw error;
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    const result = validateBatch(payload, size, selectedModel);
    predictions.push(...result.predictions);
    confidences.push(...result.confidences);
    flat += result.flat;
    progress(Math.round((start + size) / epochs * 100));
  }
  return { predictions, confidences, flat, model: selectedModel.name, model_key: selectedModel.key,
    model_id: selectedModel.id, engine: selectedModel.engine,
    smoke_test_only: selectedModel.smoke_test_only, epoch_seconds: 30 };
}
