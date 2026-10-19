# SleepcareWeb

Ứng dụng React + backend FastAPI để giảng viên truy cập bằng URL công khai,
tải EEG lên và chạy AI **trên máy chủ**. Giao diện dựa trên thư mục `web/`.

## Cấu trúc

```text
SleepcareWeb/
├── frontend/
│   ├── src/               # Giao diện, đọc file xem trước, gọi API
│   ├── public/samples/    # EEG mô phỏng để giảng viên thử
│   └── index.html
├── backend/
│   ├── app.py             # FastAPI: /api/health, /api/analyze
│   ├── inference.py       # Kiểm tra dữ liệu, z-score, suy luận ONNX CPU
│   └── models/            # CNN source-only và metadata, chỉ dùng phía server
├── api/                   # Entry point Vercel cho từng endpoint
├── tests/                 # Parity model, API và gửi EEG theo từng phần
├── docs/DEPLOYMENT.md     # Phân tích phương án và hướng dẫn deploy miễn phí
├── requirements.txt
├── package.json
└── vercel.json
```

## Luồng xử lý

1. Người dùng chọn file EEG `.npy`; trình duyệt kiểm tra định dạng và vẽ tín hiệu.
2. Khi bấm **Phân tích trên server**, frontend gửi từng phần tối đa 64 epoch
   dạng file NumPy float32 đến `POST /api/analyze`.
3. Backend kiểm tra shape/dtype/kích thước, z-score từng epoch, chạy
   `sleep-cnn-source.onnx` bằng ONNX Runtime CPU.
4. Backend trả nhãn W/N1/N2/N3/REM và confidence; frontend ghép theo đúng thứ tự,
   hiển thị hypnogram, xuất JSON/CSV và in PDF.

Mỗi phần float32 tối đa khoảng 0,77 MB, thấp hơn 1 MiB của API này và giới hạn
4,5 MB/request của Vercel. Có thể chọn file gốc lớn hơn 4,5 MB mà không cần Blob:
ứng dụng gửi nhiều phần độc lập, không giữ trạng thái trên server.
Không chuẩn hóa hay chạy AI trong trình duyệt; z-score dùng vẽ waveform chỉ để hiển thị.
Nếu API lỗi, ứng dụng báo lỗi và cho thử lại; không thay bằng dự đoán giả.

## Dữ liệu và model

- EEG Fpz-Cz, 100 Hz, epoch 30 giây (3.000 mẫu).
- `.npy` float32/float64 little-endian, C-contiguous; shape `[3000]`,
  `[N,3000]` hoặc `[N,1,3000]`; tối đa 2.000 epoch và 64 MiB/file gốc.
- EDF phải được tiền xử lý trước. API không train và không tự chuyển EDF.
- Model là **CNN source-only EEG_M_SL**, cùng bản ONNX của `web/`.
  Không tuyên bố đang chạy DINOv3. Có thể nhập JSON kết quả từ notebook.
- Backend không ghi file EEG ra disk/database và không log nội dung EEG.
  Dữ liệu được gửi qua HTTPS khi deploy và chỉ được xử lý trong bộ nhớ.
- Lịch sử báo cáo nằm ở `localStorage` của từng trình duyệt, tối đa 20 báo cáo.
  Không có tài khoản hay đồng bộ lịch sử giữa các máy.
- Mẫu demo là tín hiệu tổng hợp. Giới hạn nghiên cứu và thông tin model nằm trong
  `backend/models/metadata.json`; đây không phải công cụ chẩn đoán.

## Chạy ở máy để phát triển

Cần Node.js 22.12 trở lên và Python 3.12 hoặc 3.13.
Mở PowerShell trong **SleepcareWeb**:

```powershell
npm ci
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
npm run dev
```

Mở `http://localhost:5173`. Vite tự chuyển `/api` tới backend cổng 8000.
Sau khi deploy Vercel, không cần chạy các lệnh này hay bật máy cá nhân.

## Kiểm tra

```powershell
npm run build
npm run test:frontend
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Backend test so với kết quả ONNX/PyTorch đã xuất, kiểm tra file hỏng và so sánh
400 epoch gửi nhiều phần với suy luận toàn bộ bản ghi.
Xem [hướng dẫn deploy](docs/DEPLOYMENT.md) để đưa lên Vercel Hobby miễn phí.
