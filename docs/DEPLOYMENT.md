# Deploy SleepCare có SLARDA + DINOv3 QuickRun và CNN

Hướng dẫn Windows PowerShell, cập nhật 19/10/2026. Làm lần lượt; nếu lệnh báo lỗi,
giữ lại thông báo để xử lý trước khi sang bước kế tiếp.

## Phương án và chi phí

Dùng Vercel Hobby cho giao diện Vite và Python API cùng domain. Máy cá nhân có
thể tắt sau khi deploy. Cả hai model chạy trên CPU server. DINOv3 mặc định dùng
bundle QuickRun đã đóng gói, không cần train lại hoặc gọi API AI trả phí.

Tài liệu chính thức đã đối chiếu:

- [Hobby](https://vercel.com/docs/plans/hobby): miễn phí cho dự án cá nhân/phi thương mại, có quota.
- [Functions limits](https://vercel.com/docs/functions/limitations): Hobby có 2 GB RAM;
  Python bundle thông thường 500 MB; **Large Functions beta tối đa 5 GB**, cần
  Fluid Compute với Active CPU. Chọn gói Hobby, không chọn Pro trial/add-on.
- [Large Functions](https://vercel.com/kb/guide/troubleshooting-function-250mb-limit):
  biến `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` bật hỗ trợ package lớn cho project đủ điều kiện.
- [Python runtime](https://vercel.com/docs/functions/runtimes/python): hỗ trợ FastAPI,
  đọc `.python-version` và đóng gói model qua cấu hình `functions`.

PyTorch/Transformers làm package lớn; chỉ xem dung lượng file .pt (~83 MiB) là
chưa đủ. `vercel.json` bật Fluid Compute và giới hạn mỗi request 120 giây.
DINOv3 chỉ nhận 4 epoch/request; CNN nhận 64. Tổng file nguồn web dưới 100 MiB,
nhưng thư viện cài trên server lớn hơn. Cần kiểm tra Build Logs và runtime thật.
Không cam kết host miễn phí luôn hoạt động hoặc không hết quota; không dùng ping
liên tục để giữ host hoạt động.

## 1. Chuẩn bị tài khoản và phần mềm

Tạo tài khoản GitHub cá nhân và Vercel, chọn Hobby. Mở PowerShell:

```powershell
git --version
node --version
python --version
```

Cần Node >=22.12 và Python 3.11/3.12. Nếu thiếu, cài Git từ
https://git-scm.com/downloads/win, Node LTS từ https://nodejs.org/en/download,
Python 3.12 từ https://www.python.org/downloads/windows/; bật Add Python to PATH,
đóng rồi mở lại PowerShell. Python local dùng để kiểm thử; Vercel tự cài Python 3.12.

## 2. Kiểm tra đủ hai model

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
Get-ChildItem .\backend\models
Get-FileHash .\backend\models\slarda-dinov3-quick.pt -Algorithm SHA256
```

Phải có `slarda-dinov3-quick.pt`, `dinov3-metadata.json`,
`sleep-cnn-source.onnx`, `metadata.json`. SHA-256 DINOv3 phải là:

```text
706E9C1486B18AD772B5E89A8575F6823A90F427C49A8B26746D105363F0D4E5
```

File .pt này đã được chuẩn bị trong SleepcareWeb. Nếu bạn chỉ sao chép mã nguồn
và làm mất file, lấy lại từ bundle gốc bằng:

```powershell
Copy-Item "..\SLARDA_DINOv3\QuickRun\outputs\deployable_a_to_b.pt" ".\backend\models\slarda-dinov3-quick.pt"
```

Không cần copy `model.safetensors`, dataset hoặc thư mục training. Loader web
nạp toàn bộ state từ bundle và bỏ qua đường dẫn local cũ trong metadata.

## 3. Cài thư viện và chạy kiểm thử

```powershell
npm.cmd ci
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
npm.cmd run test:frontend
npm.cmd run build
```

Nếu `python` chưa phải 3.11/3.12, dùng `py -3.12 -m venv .venv` cho bước tạo môi trường.
Lần đầu pip tải PyTorch CPU khoảng vài trăm MB. Không dùng lệnh cài CUDA từ
README training. Giữ `requirements.txt` để máy chủ dùng cùng runtime đã ghim.

Để thử bằng trình duyệt trước khi upload, chạy backend trong PowerShell thứ nhất:

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Chạy frontend trong PowerShell thứ hai:

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
npm.cmd run dev
```

Mở http://localhost:5173, chọn Phân tích bằng AI → Chọn model AI → DINOv3 →
Phân tích trên server. Thử lại với CNN. Đây mới là local; bước sau tạo URL công khai.

## 4. Tạo repository GitHub cá nhân Private

Trên GitHub chọn New repository → tên `sleepcareweb` → Private → không thêm
README/.gitignore/license vì đã có file ở local → Create repository. Copy URL HTTPS.
Chỉ upload nội dung thư mục SleepcareWeb; không upload thư mục SleepCare cha.

Lần đầu thư mục chưa có Git:

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
git init -b main
git add .
git status --short
```

Danh sách phải có hai file model, mã backend/frontend và cấu hình; không có
`.venv`, `node_modules`, dữ liệu EEG thật hoặc bí mật. Nếu Git yêu cầu danh tính:

```powershell
git config user.name "Tên của bạn"
git config user.email "Email GitHub của bạn"
```

Commit và đẩy lên repository vừa tạo:

```powershell
git commit -m "Add SLARDA DINOv3 model selection"
$repo = Read-Host "Dán URL HTTPS repository GitHub vừa tạo"
git remote add origin $repo
git push -u origin main
```

Nếu đã làm hướng dẫn cũ và có repository, dùng `git add .`, `git commit ...`,
`git push`; bỏ `git init`/`git remote add`. Git có thể mở browser yêu cầu đăng nhập.
File bundle nhỏ hơn 100 MiB nên không cần Git LFS cho bản này. Nếu sau này thay
checkpoint lớn hơn, cần đánh giá lại cách lưu model. Model DINOv3 đi theo
[license Meta](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).

## 5. Import vào Vercel và bật package lớn

1. Đăng nhập Vercel → Add New → Project → Import repository `sleepcareweb`.
2. Root Directory là `.` vì repo chứa trực tiếp `package.json`. Nếu repo chứa
   thư mục SleepcareWeb bên trong, chọn đúng thư mục đó.
3. Framework Preset **Vite**; Build Command `npm run build`; Output Directory `dist`.
4. Trong Environment Variables, thêm tên `VERCEL_SUPPORT_LARGE_FUNCTIONS`,
   giá trị `1`, áp dụng **Production và Preview**. Đây là biến cấu hình, không phải token.
5. Vào Settings → Functions, bật Fluid Compute nếu project có công tắc này.
   Sau đó Redeploy (không chỉ bấm Retry trên bản deploy cũ). Biến môi trường chỉ
   có hiệu lực từ lần build mới.

Vercel cài CPU Torch/Transformers/ONNX Runtime từ requirements và đóng gói
`backend/models` một lần qua `api/index.py`. Không chọn GPU hoặc dịch vụ trả phí.
Large Functions hiện cho phép tối đa 5 GB trên project đủ điều kiện; bundle hiện
tại đã giảm việc lặp function nhưng vẫn cần bật biến trên. Nếu log vẫn báo
`Total bundle size ... exceeds ... 500 MB`, hãy dừng và kiểm tra biến đã có ở
đúng project và đúng Production/Preview, rồi tạo một deployment mới. Nếu sau đó
log báo bundle vượt 5 GB, Vercel Hobby không đủ cho PyTorch; khi đó cần chuyển
backend AI sang một host Python/container khác và giữ Vercel cho frontend.

## 6. Kiểm tra model thật trên production

Copy domain production `.vercel.app` từ Vercel và chạy:

```powershell
$url = Read-Host "Dán URL production, ví dụ https://sleepcareweb.vercel.app"
$url = $url.TrimEnd('/')
Invoke-RestMethod "$url/api/models" | ConvertTo-Json -Depth 6
Invoke-RestMethod "$url/api/health?model=slarda-dinov3-quick" | ConvertTo-Json
Invoke-RestMethod "$url/api/health?model=cnn-source" | ConvertTo-Json
```

Cả hai health phải trả `status: ok`; DINOv3 trả
`model_id: slarda-dinov3-quick:706e9c1486b1` và `engine: pytorch-cpu`;
CNN trả `model_id: sleep-cnn-source:b52c50097fe9` và `engine: onnxruntime-cpu`.

Trong browser:

1. Mở production URL → Phân tích bằng AI → DINOv3 QuickRun → Phân tích trên server.
2. Mẫu tổng hợp 24 epoch tạo 6 request DINOv3; kết quả ghi đúng tên DINOv3 và thông tin QuickRun.
3. Lưu JSON; mở file kiểm tra `model_key`, `model_id`, `engine`, `smoke_test_only`.
4. Quay lại phân tích → chọn CNN → chạy lại; kết quả phải ghi CNN và `smoke_test_only: false`.
5. Thử cửa sổ ẩn danh/điện thoại. Nếu Vercel yêu cầu đăng nhập, kiểm tra Project
   Settings → Deployment Protection của production rồi thử lại.

Chỉ gửi giảng viên production URL khi các bước trên chạy được. Có thể tắt dev
server và máy cá nhân sau khi xác nhận; không gửi localhost hoặc đường dẫn ổ G:.

## 7. Cập nhật lần sau

```powershell
cd "G:\My Drive\SleepCare\SleepcareWeb"
git add .
git commit -m "Update SleepCare"
git push
```

Vercel tự build lại. Nếu thay trọng số, phải cập nhật SHA-256, model ID, metadata
và kết quả tham chiếu/kiểm thử; không chỉ đổi tên file checkpoint.

## Xử lý lỗi

| Lỗi | Cách xử lý |
| --- | --- |
| `/api/models` 404 | Deploy cả project, kiểm tra `api/index.py`, Root Directory và preset Vite |
| Bundle vượt 500 MB | Kiểm tra Large Functions, biến môi trường `1`, Fluid Compute/Active CPU, rồi Redeploy |
| Tổng bundle khoảng 6 GB | Bản cũ có ba entrypoint cùng import PyTorch; phải dùng `api/index.py` duy nhất rồi redeploy |
| `No matching distribution` hoặc `uv pip install` thất bại | Dùng commit mới nhất; requirements đã chọn wheel riêng cho Python 3.12/3.13 và 3.14, có CPU index cho Torch. Kiểm tra Vercel đang dùng Python 3.12–3.14; đừng thay sang Torch CUDA |
| DINOv3 503 | Xem Function Logs: đủ .pt, checksum đúng, CPU Torch/Transformers đã cài; không tự chọn CNN thay |
| 422 | Kiểm tra model key, .npy đúng định dạng; DINOv3 tối đa 4 epoch/request |
| 429 | Máy chủ đang bận hoặc nền tảng giới hạn; client đã thử lại có giới hạn, chờ rồi chạy lại |
| 504/hết RAM | Kiểm tra logs và cold start; kết quả local không chứng minh tài nguyên cloud đủ cho mọi bản ghi |
| Model/version không khớp | Reload trang sau deployment; kiểm tra frontend/backend cùng bản |

Chưa có production URL được xác nhận chỉ từ việc sửa file và test local.
Thử nghiệm kỹ thuật không xác nhận độ chính xác y khoa của bundle QuickRun.
