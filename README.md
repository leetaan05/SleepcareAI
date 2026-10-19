# SleepcareWeb — chọn model và phân tích EEG trên server

React/Vite + FastAPI. Website chạy suy luận thật trên CPU của backend và cho chọn:

| Lựa chọn | Runtime | File đóng gói | Batch tối đa |
| --- | --- | --- | --- |
| **SLARDA + DINOv3 · QuickRun A→B** (mặc định) | PyTorch CPU | `backend/models/slarda-dinov3-quick.pt` (~82,67 MiB) | 4 epoch/request |
| EEG_M_SL · CNN source-only | ONNX Runtime CPU | `backend/models/sleep-cnn-source.onnx` (~385 KiB) | 64 epoch/request |

Bundle DINOv3 được sao chép nguyên trạng từ
`../SLARDA_DINOv3/QuickRun/outputs/deployable_a_to_b.pt`.
SHA-256: `706e9c1486b18ad772b5e89a8575f6823a90f427c49a8b26746d105363f0d4e5`.

**Bản này thực sự chạy bundle DINOv3 trên server.** Bundle chứa đủ 191 tensor
(21.650.693 tham số), gồm backbone, projector và classifier. Loader mới dựng
đúng kiến trúc rồi `load_state_dict(..., strict=True)`; dùng
`torch.load(..., weights_only=True)`. Không cần `model.safetensors`, thư mục
training, đường dẫn ổ G:, GPU hay token Hugging Face ở runtime. Đường dẫn
`dinov3_checkpoint` cũ bên trong bundle không được sử dụng. SHA-256 phải khớp
trước khi nạp; thiếu/hỏng model sẽ báo lỗi, không tự đổi model hoặc dùng trọng số ngẫu nhiên.

## Giới hạn của bundle tạm

QuickRun dùng 80 epoch tổng cộng trong train/validation/test, warm-up source-only
2 epoch và SLARDA A→B 2 epoch; backbone đóng băng. Test target chỉ có 10 mẫu:
accuracy 0,50, macro-F1 0,46095. Đây là checkpoint thử tích hợp, không phải model
huấn luyện đầy đủ. Giao diện và báo cáo JSON giữ cờ `smoke_test_only: true`.
Không so sánh trực tiếp metric này với CNN vì tập đánh giá khác nhau.

## Chạy tại máy Windows

Cần Node.js >=22.12 và Python 3.11 hoặc 3.12. Vercel dùng Python 3.12 từ
`.python-version`. Mở PowerShell và chạy từng lệnh:

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
node --version
python --version
npm.cmd ci
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` ghim PyTorch **CPU**; lần đầu cần tải khoảng vài trăm MB.
Không cài CUDA hoặc toàn bộ requirements của thư mục training vào web.
Nếu `python` đang là bản khác, dùng `py -3.12 -m venv .venv` thay lệnh tạo môi trường.

PowerShell thứ nhất — chạy backend và giữ cửa sổ mở:

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

PowerShell thứ hai — chạy frontend và giữ cửa sổ mở:

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
npm.cmd run dev
```

Mở `http://localhost:5173` → **Phân tích bằng AI** → **Chọn model AI** → chọn
SLARDA + DINOv3 hoặc CNN → **Phân tích trên server**. Dropdown bị khóa khi đang
phân tích. Đổi model rồi phân tích lại sẽ cập nhật kết quả của phiên đang chọn;
hãy tải báo cáo trước nếu muốn giữ kết quả của cả hai model.

## Dữ liệu và tiền xử lý

- EEG Fpz-Cz, 100 Hz, epoch 30 giây = 3.000 mẫu.
- `.npy` float32/float64 little-endian, C-contiguous; shape `[3000]`, `[N,3000]`
  hoặc `[N,1,3000]`; tối đa 2.000 epoch và 64 MiB/file được chọn trên trình duyệt.
- Nhận tín hiệu thô theo volt hoặc dữ liệu đã z-score từng epoch. DINOv3 dùng
  population standard deviation với floor **1e-6**, theo pipeline
  `raw/prepare_sleep_edf.py`; CNN giữ floor **1e-8** của bản export hiện có.
- DINOv3: z-score → STFT 256/hop 64 → log amplitude → resize phổ 224×224 →
  chuẩn hóa ảnh RGB → ViT-S/16 → projector 384→128 → 22 bước → 5 logits.
- Frontend chỉ đóng gói EEG theo batch của model đã chọn. CPU server chạy model
  và trả nhãn/confidence; biểu đồ z-score ở browser chỉ để xem tín hiệu.
- Không nhận `.pt` do khách tải lên và không chạy pickle từ người dùng. EDF cần
  tiền xử lý trước. Mẫu demo là EEG tổng hợp, không phải dữ liệu người thật.

## API và cấu trúc

| Endpoint | Chức năng |
| --- | --- |
| `GET /api/models` | Danh sách model, ID, batch size, giới hạn QuickRun, file có mặt hay không |
| `GET /api/health?model=slarda-dinov3-quick` | Nạp/kiểm tra model DINOv3 |
| `GET /api/health?model=cnn-source` | Nạp/kiểm tra CNN |
| `POST /api/analyze?model=<key>` | Body `.npy`, `Content-Type: application/octet-stream` |

Nếu bỏ `model`, API dùng DINOv3 QuickRun. Model không hợp lệ/batch quá lớn: 422;
model thiếu hoặc không nạp được: 503; CPU đang bận yêu cầu khác: 429.
Frontend thử lại có giới hạn khi gặp 429/502/503/504. Không trộn kết quả từ các
model hoặc phiên bản trọng số khác nhau trong một lần phân tích.

`model_key`, `model_id`, `engine`, `smoke_test_only` đi cùng kết quả, JSON và lịch sử.
`available` trong catalog chỉ xác nhận file có mặt; `/api/health` mới kiểm tra
nạp trọng số/runtime. Backend xử lý EEG trong RAM, không ghi file hoặc lưu database.
Lịch sử tối đa 20 báo cáo nằm trong localStorage của từng trình duyệt.

```text
backend/catalog.py           # Danh sách model và SHA-256 được phép
backend/dinov3_model.py      # Kiến trúc DINOv3 độc lập dùng để suy luận
backend/inference.py         # Kiểm tra NPY, chuẩn hóa, điều phối hai model
backend/app.py               # API dùng chung ở local và Vercel
backend/models/              # Hai model và metadata
api/index.py                # Một entrypoint FastAPI cho toàn bộ /api/*
frontend/src/               # Dropdown model, client chia batch, báo cáo
tests/                      # Kiểm thử API/model và frontend
docs/DEPLOYMENT.md           # Hướng dẫn deploy từng bước cho người mới
docs/VERIFICATION.md         # Kết quả kiểm thử thực tế và giới hạn
```

## Kiểm thử và build

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
npm.cmd run test:frontend
npm.cmd run build
```

Fixture DINOv3 được tính từ encoder QuickRun gốc không sửa, khởi tạo bằng
safetensors gốc rồi nạp bundle, dùng 24 epoch tổng hợp. Kiểm thử so sánh logits,
nhãn và confidence của backend độc lập với fixture này. Xem chi tiết trong
[VERIFICATION.md](docs/VERIFICATION.md).

## Deploy miễn phí

Đọc [DEPLOYMENT.md](docs/DEPLOYMENT.md). Vẫn dùng frontend Vite + Python Functions
trên Vercel Hobby, nhưng **PyTorch làm bundle server lớn hơn giới hạn mặc định**.
Project dùng một entrypoint `api/index.py` để tránh đóng gói lặp backend ba lần.
Vẫn phải bật Fluid Compute và Large Functions bằng biến
`VERCEL_SUPPORT_LARGE_FUNCTIONS=1` trong Vercel trước khi redeploy. Nếu vẫn thấy
`maximum function size (500 MB)`, biến chưa áp dụng đúng môi trường hoặc project
chưa đủ điều kiện Large Functions. Tính năng này đang beta; code local chạy được
không thay thế kiểm tra build và suy luận trên tài khoản Vercel thực tế.

Model DINOv3 có [giấy phép riêng của Meta](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
Giữ nguồn gốc/metadata; hướng dẫn dùng repository GitHub cá nhân Private để
đóng gói model cho backend. Website là công cụ nghiên cứu, không dùng chẩn đoán.
