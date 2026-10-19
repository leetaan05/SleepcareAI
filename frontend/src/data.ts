export const STAGES = ['W', 'N1', 'N2', 'N3', 'REM'] as const;
export const STAGE_NAMES = ['Thức', 'Ngủ N1', 'Ngủ N2', 'Ngủ N3', 'REM'];
export const COLORS = ['#f3b65e', '#a2b3d4', '#5686ed', '#8e74d8', '#4db6a5'];
export type Results = { predictions: number[]; confidences: number[]; model: string; epoch_seconds: number; smoke_test_only?: boolean; model_key?: string; model_id?: string; engine?: string };
export type Session = { id: string; name: string; date: string; note: string; epochs: number; samples?: Float32Array; results?: Results };

export function parseNpy(buffer: ArrayBuffer): Float32Array {
  const view = new DataView(buffer);
  if (buffer.byteLength < 12 || new TextDecoder().decode(new Uint8Array(buffer, 1, 5)) !== 'NUMPY' || view.getUint8(0) !== 147) throw new Error('File không phải NumPy .npy hợp lệ.');
  const version = view.getUint8(6);
  if (![1, 2, 3].includes(version)) throw new Error('Phiên bản .npy chưa được hỗ trợ.');
  const headerStart = version === 1 ? 10 : 12;
  const headerLength = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const offset = headerStart + headerLength;
  if (offset > buffer.byteLength) throw new Error('Header .npy bị thiếu.');
  const header = new TextDecoder().decode(new Uint8Array(buffer, headerStart, headerLength));
  const dtype = header.match(/['"]descr['"]\s*:\s*['"]([^'"]+)/)?.[1];
  if (dtype !== '<f4' && dtype !== '<f8') throw new Error('Chỉ nhận float32/float64 little-endian. Hãy lưu bằng np.save(x.astype(np.float32)).');
  if (!/['"]fortran_order['"]\s*:\s*False/.test(header)) throw new Error('Cần mảng C-contiguous, không dùng Fortran order.');
  const shape = header.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/)?.[1].split(',').map(s => s.trim()).filter(Boolean).map(Number);
  if (!shape || !shape.every(n => Number.isSafeInteger(n) && n > 0) || !(
    (shape.length === 1 && shape[0] === 3000) ||
    (shape.length === 2 && shape[1] === 3000) ||
    (shape.length === 3 && shape[1] === 1 && shape[2] === 3000))) throw new Error('Shape cần là [3000], [N,3000] hoặc [N,1,3000].');
  const count = shape.reduce((a, b) => a * b, 1), bytes = dtype === '<f4' ? 4 : 8;
  if (count / 3000 > 2000 || offset + count * bytes !== buffer.byteLength) throw new Error('File bị thiếu dữ liệu hoặc vượt giới hạn 2.000 epoch.');
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const v = bytes === 4 ? view.getFloat32(offset + i * 4, true) : view.getFloat64(offset + i * 8, true);
    values[i] = v;
    if (!Number.isFinite(values[i])) throw new Error('EEG chứa NaN/Inf hoặc vượt miền float32.');
  }
  return values;
}

export function normalizeEpochs(samples: Float32Array): { values: Float32Array; flat: number } {
  if (!samples.length || samples.length % 3000) throw new Error('Mỗi epoch cần đúng 3.000 mẫu.');
  const output = new Float32Array(samples.length);
  let flat = 0;
  for (let offset = 0; offset < samples.length; offset += 3000) {
    let mean = 0, min = Infinity, max = -Infinity;
    for (let i = 0; i < 3000; i++) {
      const x = samples[offset + i];
      if (!Number.isFinite(x)) throw new Error('EEG chứa NaN/Inf.');
      mean += x; min = Math.min(min, x); max = Math.max(max, x);
    }
    mean /= 3000;
    let variance = 0;
    for (let i = 0; i < 3000; i++) variance += (samples[offset + i] - mean) ** 2;
    const std = Math.sqrt(variance / 3000);
    if (max === min) { flat++; continue; }
    const scale = Math.max(std, 1e-8);
    for (let i = 0; i < 3000; i++) output[offset + i] = (samples[offset + i] - mean) / scale;
  }
  return { values: output, flat };
}

export function parseResults(value: unknown): Results {
  const p = value as Record<string, unknown>;
  if (!p || p.schema_version !== 1 || p.epoch_seconds !== 30 || p.sampling_rate_hz !== 100 || p.channel !== 'EEG Fpz-Cz' ||
      !Array.isArray(p.predictions) || !p.predictions.length || p.predictions.length > 2000 ||
      !p.predictions.every(v => Number.isInteger(v) && v >= 0 && v <= 4) ||
      !Array.isArray(p.confidences) || p.confidences.length !== p.predictions.length ||
      !p.confidences.every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1) ||
      typeof p.model !== 'string') throw new Error('JSON kết quả không đúng schema xuất từ notebook.');
  return { predictions: p.predictions, confidences: p.confidences, epoch_seconds: 30,
    model: p.model.slice(0, 150), smoke_test_only: p.smoke_test_only === true,
    model_key: typeof p.model_key === 'string' ? p.model_key.slice(0, 150) : undefined,
    model_id: typeof p.model_id === 'string' ? p.model_id.slice(0, 150) : undefined,
    engine: typeof p.engine === 'string' ? p.engine.slice(0, 80) : undefined };
}

export function duration(epochs: number): string {
  const minutes = epochs / 2, hours = Math.floor(minutes / 60), rest = minutes % 60;
  return hours ? `${hours} giờ ${rest.toLocaleString('vi-VN')} phút` : `${rest.toLocaleString('vi-VN')} phút`;
}

export function makeReport(session: Session) {
  if (!session.results) throw new Error('Phiên chưa có kết quả.');
  return { schema_version: 1, session: session.name, exported_at: new Date().toISOString(),
    channel: 'EEG Fpz-Cz', sampling_rate_hz: 100, ...session.results,
    class_names: STAGES, stage_minutes: STAGES.map((_, i) => session.results!.predictions.filter(v => v === i).length / 2),
    note: session.note, research_only: true };
}
