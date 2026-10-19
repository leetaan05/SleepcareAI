import { useEffect, useRef, useState } from 'react';
import { Activity, ArrowRight, BarChart3, BookOpen, Check, ChevronLeft, ChevronRight, CircleHelp, Clock3, Download, FileJson, History, House, List, LoaderCircle, Moon, Play, Plus, Search, ShieldCheck, Sparkles, Upload, X } from 'lucide-react';
import { COLORS, STAGES, STAGE_NAMES, duration, makeReport, normalizeEpochs, parseNpy, parseResults, type Session } from './data';
import { checkServer, infer } from './inference';

type Page = 'home' | 'sessions' | 'analysis' | 'results' | 'history' | 'guide';
const NAV = [{ id: 'home', label: 'Trang chủ', icon: House }, { id: 'sessions', label: 'Phiên ngủ của tôi', icon: List }, { id: 'analysis', label: 'Phân tích bằng AI', icon: Sparkles }, { id: 'results', label: 'Kết quả giấc ngủ', icon: BarChart3 }, { id: 'history', label: 'Lịch sử báo cáo', icon: History }] as const;
const TITLES: Record<Page, [string, string]> = {
  home: ['Một góc nhìn rõ hơn về giấc ngủ', 'Theo dõi tín hiệu EEG, khám phá nhịp điệu giấc ngủ của bạn.'],
  sessions: ['Phiên ngủ của tôi', 'Chọn bản ghi để bắt đầu hành trình phân tích.'],
  analysis: ['Phân tích bằng AI', 'Từ tín hiệu EEG đến năm giai đoạn giấc ngủ.'],
  results: ['Kết quả giấc ngủ', 'Các giai đoạn được mô hình dự đoán theo thời gian.'],
  history: ['Lịch sử báo cáo', 'Những kết quả bạn đã lưu trên trình duyệt này.'],
  guide: ['Bắt đầu cùng SleepCare', 'Chuẩn bị dữ liệu và hiểu kết quả nghiên cứu.'],
};

function download(name: string, value: string, type = 'application/json') {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function readHistory(): Session[] {
  try {
    const items = JSON.parse(localStorage.getItem('sleepcare-server-reports-v1') || '[]');
    if (!Array.isArray(items)) return [];
    return items.slice(0, 20).flatMap(p => {
      try {
        const results = parseResults(p);
        return [{ id: String(p.id), name: String(p.session), date: String(p.exported_at), note: String(p.note), epochs: results.predictions.length, results }];
      } catch { return []; }
    });
  } catch { return []; }
}

function Waveform({ samples, epoch }: { samples?: Float32Array; epoch: number }) {
  if (!samples) return <div className="empty small"><Activity /><p>Phiên này chỉ có kết quả đã nhập.<br />Hãy tải EEG để xem tín hiệu.</p></div>;
  const slice = samples.slice(epoch * 3000, (epoch + 1) * 3000);
  const normalized = normalizeEpochs(slice).values;
  const points = Array.from({ length: 600 }, (_, i) => `${i * 720 / 599},${110 - Math.max(-4, Math.min(4, normalized[i * 5])) * 20}`).join(' ');
  return <div className="wave"><svg viewBox="0 0 720 220" role="img" aria-label={`Tín hiệu EEG epoch ${epoch + 1}, cửa sổ 30 giây`}>
    <defs><pattern id="grid" width="120" height="44" patternUnits="userSpaceOnUse"><path d="M 120 0 L 0 0 0 44" fill="none" stroke="#e5ebf3" strokeWidth="1" /></pattern></defs>
    <rect width="720" height="220" fill="url(#grid)" /><rect x="240" width="120" height="220" fill="#edf3ff" />
    <polyline points={points} fill="none" stroke="#377ceb" strokeWidth="1.8" strokeLinejoin="round" />
  </svg><div className="axis">{[0, 5, 10, 15, 20, 25, 30].map(n => <span key={n}>{n}s</span>)}</div></div>;
}

function Hypnogram({ session }: { session: Session }) {
  const predictions = session.results!.predictions, order = [0, 4, 1, 2, 3];
  const y = (stage: number) => 28 + order.indexOf(stage) * 42;
  const n = predictions.length, w = 660 / n;
  return <div className="hypnogram"><svg viewBox="0 0 720 250" role="img" aria-label="Biểu đồ giai đoạn giấc ngủ theo thời gian">
    {order.map(stage => <g key={stage}><text x="0" y={y(stage) + 4}>{STAGES[stage]}</text><line x1="45" x2="708" y1={y(stage)} y2={y(stage)} stroke="#e6ebf2" /></g>)}
    {predictions.map((stage, i) => <g key={i}>{i > 0 && <line x1={45 + i * w} x2={45 + i * w} y1={y(predictions[i - 1])} y2={y(stage)} stroke="#b6c5dd" strokeWidth="1.5" />}<line x1={45 + i * w} x2={45 + (i + 1) * w} y1={y(stage)} y2={y(stage)} stroke={COLORS[stage]} strokeWidth="4" /></g>)}
    {[0, 1, 2, 3, 4].map(i => <text key={i} x={45 + i * 165} y="236" textAnchor={i === 4 ? 'end' : 'start'}>{(n / 2 * i / 4).toLocaleString('vi-VN', { maximumFractionDigits: 1 })} phút</text>)}
  </svg><div className="legend">{STAGES.map((s, i) => <span key={s}><i style={{ background: COLORS[i] }} />{s}</span>)}</div></div>;
}

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedId, setSelectedId] = useState('demo');
  const [saved, setSaved] = useState<Session[]>(readHistory);
  const [search, setSearch] = useState('');
  const [epoch, setEpoch] = useState(0);
  const [busy, setBusy] = useState(false), [progress, setProgress] = useState(0);
  const [message, setMessage] = useState(''), [error, setError] = useState('');
  const [server, setServer] = useState<'checking' | 'online' | 'offline'>('checking');
  const analysisAbort = useRef<AbortController | null>(null);
  const upload = useRef<HTMLInputElement>(null), importFile = useRef<HTMLInputElement>(null);
  const selected = sessions.find(s => s.id === selectedId);
  const result = selected?.results;
  const counts = STAGES.map((_, i) => result?.predictions.filter(s => s === i).length || 0);
  const longest = counts.indexOf(Math.max(...counts));

  useEffect(() => {
    let active = true;
    fetch('/samples/eeg-demo.npy').then(r => { if (!r.ok) throw new Error('Không tải được EEG mô phỏng.'); return r.arrayBuffer(); })
      .then(b => { if (active) setSessions([{ id: 'demo', name: 'Phiên EEG mô phỏng', date: new Date().toISOString(), note: 'Tín hiệu tổng hợp để thử giao diện; không đại diện giấc ngủ thực.', epochs: 24, samples: parseNpy(b) }]); })
      .catch(e => { if (active) setError(String(e.message)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    checkServer(controller.signal).then(() => setServer('online')).catch(() => {
      if (!controller.signal.aborted) setServer('offline');
    });
    return () => { controller.abort(); analysisAbort.current?.abort(); };
  }, []);

  async function reconnect() {
    setServer('checking');
    try { await checkServer(); setServer('online'); }
    catch (e) { setServer('offline'); setError((e as Error).message); }
  }

  function navigate(next: Page) { setPage(next); setMessage(''); setError(''); }
  function select(session: Session) { setSelectedId(session.id); setEpoch(0); }
  async function load(file?: File, resultsOnly = false) {
    if (!file) return;
    setError(''); setMessage('');
    try {
      if (file.size > 64 * 1024 * 1024) throw new Error('Giới hạn file 64 MB. Hãy chia bản ghi thành các đoạn nhỏ hơn.');
      let session: Session;
      if (resultsOnly) {
        const p = parseResults(JSON.parse(await file.text()));
        session = { id: crypto.randomUUID(), name: file.name, date: new Date().toISOString(), note: 'Kết quả nhập từ notebook / công cụ bên ngoài.', epochs: p.predictions.length, results: p };
      } else {
        const values = parseNpy(await file.arrayBuffer());
        session = { id: crypto.randomUUID(), name: file.name, date: new Date().toISOString(), note: 'EEG do bạn cung cấp · gửi đến máy chủ khi bấm phân tích; server không lưu file.', epochs: values.length / 3000, samples: values };
      }
      setSessions(prev => [...prev, session]); select(session); setPage(resultsOnly ? 'results' : 'analysis');
    } catch (e) { setError((e as Error).message); }
  }

  async function analyze() {
    if (!selected?.samples || busy) return;
    const current = selected;
    const controller = new AbortController();
    analysisAbort.current = controller;
    setBusy(true); setProgress(0); setError(''); setMessage('');
    try {
      const values = await infer(current.samples!, setProgress, controller.signal);
      setServer('online');
      setSessions(prev => prev.map(s => s.id === current.id ? { ...s, results: values } : s));
      if (values.flat) setMessage(`${values.flat} epoch tín hiệu hằng đã được giữ lại; cần xem xét chất lượng bản ghi.`);
      setPage('results');
    } catch (e) {
      if (controller.signal.aborted) setMessage('Đã dừng phân tích. Kết quả chưa hoàn thành không được lưu.');
      else { setServer('offline'); setError(`Không chạy được mô hình: ${(e as Error).message}`); }
    }
    finally { analysisAbort.current = null; setBusy(false); }
  }

  function saveReport() {
    if (!selected?.results) return;
    try {
      const report = makeReport(selected), updated = [{ ...selected, samples: undefined }, ...saved.filter(s => s.id !== selected.id)].slice(0, 20);
      localStorage.setItem('sleepcare-server-reports-v1', JSON.stringify(updated.map(s => ({ id: s.id, ...makeReport(s) }))));
      setSaved(updated); setMessage('Đã lưu báo cáo trên trình duyệt này.');
      download(`sleepcare-${selected.id}.json`, JSON.stringify(report, null, 2));
    } catch { setError('Không lưu được trên trình duyệt. Hãy dùng Xuất JSON để tải báo cáo.'); }
  }
  function exportCsv() {
    if (!result) return;
    const rows = ['epoch,start_seconds,stage,confidence', ...result.predictions.map((s, i) => `${i + 1},${i * 30},${STAGES[s]},${result.confidences[i].toFixed(6)}`)];
    download('sleepcare-predictions.csv', rows.join('\n'), 'text/csv;charset=utf-8');
  }
  function openSaved(s: Session) { setSessions(prev => prev.some(p => p.id === s.id) ? prev : [...prev, s]); select(s); navigate('results'); }

  return <div className="app">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); navigate('home'); }}><span className="brand-symbol"><Moon size={25} fill="currentColor" /></span><span>SleepCare<span className="brand-ai">AI · EEG LAB</span></span></a>
      <div className="nav-label">KHÔNG GIAN CỦA BẠN</div>
      <nav aria-label="Điều hướng chính">{NAV.map(({ id, label, icon: Icon }) => <button key={id} className={page === id ? 'nav-item active' : 'nav-item'} onClick={() => navigate(id)} aria-current={page === id ? 'page' : undefined}><Icon size={20} /><span>{label}</span>{page === id && <i />}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-note"><ShieldCheck size={19} /><div>Phân tích bằng AI trên server<span>EEG được gửi khi bấm phân tích</span></div></div><button className="nav-item" onClick={() => navigate('guide')}><CircleHelp size={20} />Hướng dẫn sử dụng</button><span className="version">SleepCare · Nghiên cứu giấc ngủ</span></div>
    </aside>
    <main>
      <header className="topbar"><span><i className="status-dot" />PHÒNG PHÂN TÍCH GIẤC NGỦ</span><span className="top-right"><button className={`server-status status-${server}`} onClick={() => void reconnect()} disabled={server === 'checking'} title="Kiểm tra kết nối máy chủ AI"><i />{server === 'online' ? 'AI sẵn sàng' : server === 'checking' ? 'Đang kết nối AI…' : 'Kiểm tra kết nối'}</button><span className="model-tag">EEG · 100 Hz</span><span className="avatar">SC</span></span></header>
      <div className="page-heading"><div><span className="eyebrow">SLEEPCARE AI</span><h1>{TITLES[page][0]}</h1><p>{TITLES[page][1]}</p></div><button className="button secondary compact" onClick={() => importFile.current?.click()} disabled={busy}><FileJson size={16} />Nhập kết quả</button></div>
      <input ref={upload} type="file" accept=".npy" hidden onChange={e => { void load(e.target.files?.[0]); e.target.value = ''; }} />
      <input ref={importFile} type="file" accept=".json" hidden onChange={e => { void load(e.target.files?.[0], true); e.target.value = ''; }} />
      {error && <div role="alert" className="notice error">{error}<button aria-label="Đóng lỗi" onClick={() => setError('')}><X size={16} /></button></div>}
      {message && <div role="status" className="notice"><Check size={17} />{message}</div>}

      {page === 'home' && <>
        <div className="home-grid"><section className="hero"><span className="pill"><Moon size={13} />CHẠM VÀO NHỊP ĐIỆU GIẤC NGỦ</span><h2>Mỗi giấc ngủ<br />kể một câu chuyện.</h2><p>Khám phá từng giai đoạn qua tín hiệu EEG,<br className="desktop" /> trong một không gian đơn giản và trực quan.</p><button className="button" onClick={() => navigate('sessions')}>Khám phá phiên EEG<ArrowRight size={18} /></button><div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="crescent" /><i className="star s1" /><i className="star s2" /><i className="star s3" /><svg viewBox="0 0 350 100"><path d="M0 65 Q25 65 32 45 T58 60 T85 40 T110 60 T135 30 T160 60 T195 55 T230 62 T265 45 T300 60 T350 50" fill="none" stroke="#a6c6fc" strokeWidth="2" /></svg></div></section>
        <section className="card recent"><div className="section-label">PHIÊN ĐANG CHỌN <span className="live-dot" /></div><span className="recent-moon"><Moon size={25} /></span><h3>{selected?.name || 'Đang tải phiên mẫu…'}</h3><p>{selected ? duration(selected.epochs) : 'Chuẩn bị dữ liệu'}</p><span className={`badge ${result ? 'green' : 'amber'}`}>{result ? 'Đã phân tích' : 'Sẵn sàng khám phá'}</span><button className="text-button" onClick={() => navigate('analysis')}>Xem tín hiệu EEG<ArrowRight size={16} /></button></section></div>
        <div className="stats"><Stat icon={<List />} value={String(sessions.length).padStart(2, '0')} label="Phiên EEG trong bộ nhớ" /><Stat icon={<Clock3 />} value={String(saved.length).padStart(2, '0')} label="Báo cáo đã lưu" tone="teal" /><Stat icon={<BarChart3 />} value="05" label="Giai đoạn giấc ngủ" tone="purple" /></div>
        <div className="lower-grid"><section className="card introduction"><div className="section-label">TỪ TÍN HIỆU ĐẾN GÓC NHÌN MỚI</div><h3>Hiểu hơn về các giai đoạn ngủ</h3><p>Tín hiệu EEG được chia thành các cửa sổ 30 giây. Mô hình phân loại từng cửa sổ thành một trong năm giai đoạn.</p><div className="stage-cards">{STAGES.map((s, i) => <span key={s}><i style={{ background: COLORS[i] }} />{s}<small>{STAGE_NAMES[i]}</small></span>)}</div><button className="text-button" onClick={() => navigate('guide')}>Tìm hiểu cách đọc kết quả<ArrowRight size={16} /></button></section><section className="card caution"><span className="round-icon amber-icon"><BookOpen /></span><h3>Một công cụ nghiên cứu</h3><p>SleepCare mô tả các giai đoạn do mô hình dự đoán từ EEG. Kết quả không dùng để chẩn đoán bệnh hay chấm điểm chất lượng giấc ngủ.</p><span className="subtle">Phiên có sẵn dùng tín hiệu mô phỏng.</span></section></div>
      </>}

      {page === 'sessions' && <><div className="toolbar"><label className="search"><Search size={19} /><input placeholder="Tìm theo tên phiên EEG…" value={search} onChange={e => setSearch(e.target.value)} /></label><button className="button" disabled={busy} onClick={() => upload.current?.click()}><Upload size={17} />Tải EEG lên</button></div><div className="sessions-grid"><section className="card"><div className="card-heading"><div><h2>Danh sách phiên ngủ</h2><p>{sessions.length} phiên trong bộ nhớ của tab này</p></div><span className="count-pill">{sessions.length}</span></div><div className="session-list">{sessions.filter(s => s.name.toLowerCase().includes(search.toLowerCase())).map(s => <button disabled={busy} key={s.id} onClick={() => select(s)} className={`session-row ${selectedId === s.id ? 'selected' : ''}`}><span className="session-icon"><Moon size={22} /></span><span className="session-title">{s.name}<small>{s.id === 'demo' ? 'EEG mô phỏng · 24 epoch' : 'EEG do bạn tải / kết quả đã nhập'}</small></span><span className="session-info">{duration(s.epochs)}<small className={`badge ${s.results ? 'green' : 'amber'}`}>{s.results ? 'Đã phân tích' : 'Chưa phân tích'}</small></span></button>)}{!sessions.filter(s => s.name.toLowerCase().includes(search.toLowerCase())).length && <p className="empty">Không có phiên khớp từ khóa.</p>}</div><div className="upload-hint"><Plus size={18} /><span>Nhận .npy float32/float64 · EEG Fpz-Cz · 100 Hz<br /><small>Chọn file để xem trước. Bấm phân tích để gửi EEG đến server.</small></span></div></section><section className="card session-details"><h2>Thông tin phiên</h2>{selected ? <><div className="selected-label"><span className="eyebrow">ĐANG CHỌN</span><h3>{selected.name}</h3></div><Info label="Kênh tín hiệu" value="EEG Fpz-Cz · 1 kênh" /><Info label="Tần số lấy mẫu" value="100 Hz" /><Info label="Thời lượng" value={duration(selected.epochs)} /><Info label="Số epoch" value={`${selected.epochs} × 30 giây`} /><p className="subtle">{selected.note}</p><button className="button" onClick={() => navigate('analysis')}>Xem EEG và phân tích<ArrowRight size={17} /></button></> : <p>Chưa có phiên EEG.</p>}</section></div></>}

      {page === 'analysis' && <><div className="steps"><span><i className="done">1</i>Chọn phiên</span><ArrowRight /><span className="current"><i>2</i>Xem tín hiệu EEG</span><ArrowRight /><span><i>3</i>Nhận dự đoán</span></div><div className="analysis-grid"><section className="card waveform-card"><div className="card-heading"><div><h2>EEG — {selected?.name || 'Chưa chọn phiên'}</h2><p>Cửa sổ 30 giây · tín hiệu chuẩn hóa z-score</p></div><span className="badge blue">1 KÊNH</span></div><Waveform samples={selected?.samples} epoch={epoch} />{selected?.samples && <div className="epoch-control"><button aria-label="Epoch trước" disabled={epoch === 0} onClick={() => setEpoch(e => e - 1)}><ChevronLeft size={18} /></button><label>Epoch {epoch + 1} / {selected.epochs}<input aria-label="Chọn epoch EEG" type="range" min="0" max={selected.epochs - 1} value={epoch} onChange={e => setEpoch(Number(e.target.value))} /></label><button aria-label="Epoch sau" disabled={epoch >= selected.epochs - 1} onClick={() => setEpoch(e => e + 1)}><ChevronRight size={18} /></button></div>}<p className="subtle">Biên độ hiển thị theo độ lệch chuẩn. Trục ngang là thời gian trong một epoch.</p></section><section className="card model-card"><span className="round-icon"><Sparkles /></span><h2>{busy ? 'Đang phân tích…' : server === 'online' ? 'Máy chủ AI sẵn sàng' : 'Phân tích trên máy chủ'}</h2><span className="badge purple">EEG_M_SL · SOURCE-ONLY</span><p>EEG → CNN encoder<br />→ Bộ phân loại 5 giai đoạn</p><Info label="Đơn vị phân loại" value="Epoch 30 giây" /><Info label="Chạy trên" value="Máy chủ · CPU" /><Info label="Dữ liệu gửi máy chủ" value="Có · khi phân tích" /><p className="server-note">EEG được gửi đến server để phân tích. Server không lưu file; báo cáo được lưu trên trình duyệt.</p><button className="button" disabled={busy || !selected?.samples} onClick={() => void analyze()}>{busy ? <LoaderCircle className="spin" size={18} /> : <Play size={17} />}{busy ? `Đang xử lý ${progress}%` : 'Phân tích trên server'}{!busy && <ArrowRight size={17} />}</button>{busy && <><progress value={progress} max="100" aria-label="Tiến trình phân tích" /><button className="button secondary" onClick={() => analysisAbort.current?.abort()}>Dừng phân tích</button></>}<small>Chỉ suy luận · không huấn luyện lại</small>{!selected?.samples && <button className="text-button" onClick={() => upload.current?.click()}>Tải dữ liệu EEG<Upload size={16} /></button>}</section></div><div className="notice muted"><ShieldCheck size={19} /><span>Baseline source-only đang được dùng cho web. Kết quả DINOv3 đã train có thể nhập từ notebook bằng nút “Nhập kết quả”.</span></div></>}

      {page === 'results' && (!selected || !result ? <section className="card empty"><BarChart3 size={38} /><h2>Chưa có kết quả phân tích</h2><p>Chọn một phiên EEG và chạy mô hình để xem hypnogram.</p><button className="button" onClick={() => navigate('analysis')}>Đến phân tích AI<ArrowRight size={17} /></button></section> : <><div className="result-context"><span>{selected.name}</span><span className="badge blue">{result.model}</span></div>{result.smoke_test_only && <div className="notice error">Model này chỉ chạy smoke test; chưa hoàn thành huấn luyện.</div>}<div className="stats result-stats"><Stat icon={<Clock3 />} value={duration(selected.epochs)} label="Tổng thời lượng bản ghi" /><Stat icon={<Activity />} value={`${selected.epochs} epoch`} label="Số cửa sổ đã phân loại" tone="teal" /><Stat icon={<BarChart3 />} value={`${STAGES[longest]} · ${duration(counts[longest])}`} label="Giai đoạn dự đoán dài nhất" tone="purple" /></div><div className="results-grid"><section className="card"><h2>Hypnogram</h2><p>Các giai đoạn AI dự đoán theo thời gian</p><Hypnogram session={selected} /></section><section className="card stage-duration"><h2>Thời lượng giai đoạn</h2>{STAGES.map((s, i) => <div className="stage-duration-row" key={s}><div><span><i style={{ background: COLORS[i] }} />{STAGE_NAMES[i]}</span><span>{(counts[i] / 2).toLocaleString('vi-VN')} phút</span></div><div className="bar-track"><div style={{ width: `${counts[i] / selected.epochs * 100}%`, background: COLORS[i] }} /></div></div>)}</section></div><div className="report-actions"><div className="notice muted"><CircleHelp size={19} />{selected.note}</div><button className="button" onClick={saveReport}><Download size={17} />Lưu báo cáo JSON</button><button className="button secondary" onClick={exportCsv}>Xuất CSV</button><button className="button secondary" onClick={() => window.print()}>In / PDF</button></div><p className="subtle">Thời lượng = số epoch × 30 giây. Đây là nhãn dự đoán, không phải đánh giá chất lượng hay chẩn đoán giấc ngủ.</p></>)}

      {page === 'history' && <section className="card"><div className="card-heading"><div><h2>Báo cáo đã lưu</h2><p>Lưu tối đa 20 báo cáo trên trình duyệt này; không lưu tín hiệu EEG gốc.</p></div><span className="count-pill">{saved.length}</span></div>{!saved.length ? <div className="empty"><History size={36} /><h3>Chưa có báo cáo đã lưu</h3><p>Sau khi phân tích, chọn “Lưu báo cáo JSON”.</p></div> : saved.map(s => <div className="history-row" key={s.id}><span className="session-icon"><BarChart3 size={20} /></span><div><strong>{s.name}</strong><small>{duration(s.epochs)} · {s.results?.model}</small></div><button className="text-button" onClick={() => openSaved(s)}>Xem<ArrowRight size={16} /></button><button className="icon-button" aria-label={`Xóa báo cáo ${s.name}`} onClick={() => { try { const next = saved.filter(p => p.id !== s.id); localStorage.setItem('sleepcare-server-reports-v1', JSON.stringify(next.map(p => ({ id: p.id, ...makeReport(p) })))); setSaved(next); } catch { setError('Không cập nhật được bộ nhớ trình duyệt.'); } }}><X size={16} /></button></div>)}</section>}

      {page === 'guide' && <div className="guide-grid"><section className="card guide"><span className="round-icon"><Upload /></span><h2>Chuẩn bị EEG của bạn</h2><p>Nhận file NumPy .npy, float32 hoặc float64, little-endian, C-contiguous. Shape hỗ trợ: <code>[3000]</code>, <code>[N,3000]</code> hoặc <code>[N,1,3000]</code>.</p><p>Dùng kênh EEG Fpz-Cz ở 100 Hz; mỗi epoch 30 giây. Tín hiệu thô theo volt hoặc EEG đã z-score từng epoch. Server chuẩn hóa từng epoch trước khi dự đoán. Định dạng .npy không lưu tần số hay tên kênh nên bạn cần xác nhận hai thông tin này trước khi tải.</p><p>Giới hạn 2.000 epoch / file, tối đa 64 MB. File EDF cần được xử lý bằng pipeline Python trước.</p><button className="button secondary" onClick={() => downloadSample()}>Tải EEG mô phỏng<Download size={16} /></button></section><section className="card guide"><span className="round-icon purple-icon"><BarChart3 /></span><h2>Đọc kết quả có ngữ cảnh</h2><p>W: thức; N1: ngủ nông; N2: ngủ giai đoạn 2; N3: ngủ sâu; REM: ngủ chuyển động mắt nhanh. Hypnogram thể hiện một nhãn cho mỗi epoch.</p><p>Web sử dụng CNN source-only từ thử nghiệm SC→ST. Báo cáo đã lưu của project ghi accuracy 69,93%, macro-F1 0,478 trên 2.025 epoch ST test. Target sau thích ứng chưa được chọn cho web vì kết quả thấp hơn.</p><p>Confidence trong CSV là xác suất softmax của mô hình, chưa được hiệu chuẩn và không thể hiện độ chắc chắn y khoa.</p></section><section className="card guide"><span className="round-icon teal-icon"><FileJson /></span><h2>Kết nối kết quả từ Colab</h2><p>Notebook train CNN/DINOv3 xuất JSON chứa các nhãn dự đoán. Chọn “Nhập kết quả” để xem hypnogram của model đó. Báo cáo chỉ có nhãn sẽ không có waveform.</p><p>Huấn luyện và kiểm thử thực hiện trên Colab; web dùng để suy luận baseline hoặc xem kết quả đã xuất. Khi bấm phân tích, EEG được gửi theo từng phần nhỏ đến máy chủ để chạy mô hình ONNX. Server xử lý trong bộ nhớ, không lưu file EEG.</p></section><section className="card guide"><span className="round-icon amber-icon"><ShieldCheck /></span><h2>Lưu và chia sẻ</h2><p>“Lưu báo cáo JSON” tải file và lưu một bản trên trình duyệt. CSV chứa nhãn, thời điểm và confidence từng epoch. Chọn “In / PDF” để lưu trang kết quả.</p><p>Dữ liệu EEG chỉ tồn tại trong bộ nhớ tab. Reload sẽ xóa những phiên chưa lưu; báo cáo đã lưu còn ở mục Lịch sử. Mẫu có sẵn là tín hiệu tổng hợp, không chứa dữ liệu người thật.</p></section></div>}
      <footer><span><Moon size={13} />SleepCare AI</span><span>Khám phá giấc ngủ qua dữ liệu · Công cụ nghiên cứu</span></footer>
    </main>
  </div>;
}

function downloadSample() { const a = document.createElement('a'); a.href = '/samples/eeg-demo.npy'; a.download = 'eeg-demo.npy'; a.click(); }
function Stat({ icon, value, label, tone = 'blue' }: { icon: React.ReactNode; value: string; label: string; tone?: string }) { return <div className="card stat"><span className={`round-icon ${tone}-icon`}>{icon}</span><div><strong>{value}</strong><p>{label}</p></div></div>; }
function Info({ label, value }: { label: string; value: string }) { return <div className="info-row"><span>{label}</span><strong>{value}</strong></div>; }
