# Phương án deploy SleepcareWeb

## Quyết định

Dùng **Vercel Hobby** cho frontend React/Vite và hai Python Functions.
Đây là website có xử lý động phía server: `POST /api/analyze` thực sự nạp
model ONNX và dự đoán EEG bằng CPU của Vercel.
Frontend và API cùng domain, không cần server local, CORS hay dịch vụ AI trả phí.

Sau khi deploy thành công, giảng viên mở URL `https://<ten-project>.vercel.app`.
Máy cá nhân có thể tắt. Serverless functions được Vercel khởi tạo khi có request;
không cần tự chạy tiến trình server hay dùng dịch vụ ping để giữ máy thức.

**Không dịch vụ miễn phí nào bảo đảm website không bao giờ lỗi hoặc ngừng hoạt động.**
Vercel có thể có cold start, sự cố nền tảng và giới hạn tài nguyên.
Yêu cầu thực tế mà thiết kế này đáp ứng là có hosting và backend công khai,
độc lập máy cá nhân, dùng miễn phí trong hạn mức cho bài tập phi thương mại.

## Vì sao không đưa nguyên project nghiên cứu lên host?

Thư mục tổng có raw EEG, dataset PyTorch, hai nhánh training và checkpoint
`model.safetensors`. Chúng phục vụ nghiên cứu, không phải server web.
Ứng dụng hiện có sẵn CNN xuất ONNX khoảng 394 KB, đủ để suy luận CPU.
Project mới đóng gói riêng model này, giao diện và API; không phụ thuộc đường dẫn
`G:\My Drive\SleepCare` hay các thư mục đồng cấp sau khi deploy.

Giao diện cũ suy luận bằng ONNX WebAssembly trong trình duyệt.
Trong project mới, phần này đã đổi thành client gọi API; browser không tải
runtime WebAssembly/model để phân tích nữa. Backend dùng đúng bản model được
kiểm tra SHA-256 để tránh trộn model giữa các phần của một phiên.

## Giới hạn miễn phí cần biết

Tài liệu chính thức được đối chiếu ngày 19/10/2026:

- [Hobby](https://vercel.com/docs/plans/hobby): dành cho cá nhân/phi thương mại;
  có hạn mức tài nguyên. Không chọn Pro trial hay add-on trả phí cho bài này.
- [Functions limits](https://vercel.com/docs/functions/limitations):
  request/response tối đa 4,5 MB. Python function có giới hạn bundle.
  API này nhận tối đa 1 MiB và 64 epoch/request để giữ tải nhỏ.
- [Fluid Compute](https://vercel.com/docs/fluid-compute): hỗ trợ Python và
  cold start/scaling do Vercel quản lý. Project bật Fluid Compute, đặt giới hạn
  60 giây/request; mỗi request chỉ suy luận một batch nhỏ.
- [Usage](https://vercel.com/docs/functions/usage-and-pricing): Hobby có hạn mức
  CPU, memory và invocation. Vượt hạn mức có thể phải đợi kỳ reset; không cam kết
  phục vụ số người không giới hạn.
- [Python /api](https://vercel.com/docs/functions/runtimes/python/api-directory):
  hỗ trợ Python Functions cùng ứng dụng frontend. Cấu hình framework được
  giữ là **Vite**, không chuyển toàn project sang preset FastAPI.

Không cần database/Blob cho luồng chọn file → phân tích → tải báo cáo.
Nếu cần tài khoản, lưu EEG lâu dài hoặc đồng bộ lịch sử thì phải bổ sung storage,
database và kiểm soát truy cập; phần đó chưa thuộc luồng hiện tại.

## Deploy bằng GitHub

1. Tạo repository chứa **nội dung SleepcareWeb**. Giữ file
   `backend/models/sleep-cnn-source.onnx` và `metadata.json`.
   Không đưa `node_modules`, `.venv`, dataset hay checkpoint training vào repo.
2. Đăng nhập Vercel bằng tài khoản cá nhân, chọn gói Hobby, chọn **Add New Project**
   và import repository.
3. **Root Directory**:
   - Repo chỉ chứa project này: giữ `.`.
   - Repo chứa cả SleepCare: chọn `SleepcareWeb`.
4. Giữ **Framework Preset: Vite**, build command `npm run build`,
   output directory `dist`. Dùng Node.js 22.x hoặc mới hơn tương thích;
   Python được khai báo 3.12 trong `.python-version`.
5. Chọn **Deploy**. Vercel cài dependencies Node/Python, build frontend và
   đóng gói `api/analyze.py`, `api/health.py` kèm `backend/`.
6. Để giảng viên truy cập mà không phải đăng nhập Vercel, gửi production domain
   `.vercel.app`; kiểm tra Deployment Protection của production phù hợp mục đích
   chia sẻ công khai. Không gửi đường dẫn file local hay localhost.

Nếu dùng CLI thay GitHub, mở terminal tại SleepcareWeb:

```powershell
npx vercel@latest
npx vercel@latest --prod
```

CLI yêu cầu đăng nhập tài khoản Vercel của chủ project.
Không thêm token/tài khoản vào source code.

## Kiểm tra sau deploy

1. Mở `https://<production-domain>/api/health`: phải trả JSON
   `status: "ok"`, `model_id: "sleep-cnn-source:b52c50097fe9"`.
2. Mở website; trạng thái góc trên báo **AI sẵn sàng**.
3. Chọn phiên EEG mô phỏng → **Phân tích trên server**. Nhận 24 epoch kết quả.
4. Dùng một file EEG hợp lệ lớn hơn 4,5 MB; Network phải có nhiều request nhỏ
   tới `/api/analyze`, không request nào vượt 1 MiB.
5. Kiểm tra xem tín hiệu, hypnogram, lưu JSON, xuất CSV và in PDF.
6. Dùng thiết bị khác hoặc cửa sổ riêng tư để xác nhận URL không phụ thuộc
   phiên đăng nhập hay máy đang chạy dev server.

## Xử lý lỗi deploy

- Trang có giao diện nhưng API 404: kiểm tra Root Directory và preset **Vite**;
  phải deploy cả project chứ không chỉ upload `dist/`.
- API 503: xem Function Logs; kiểm tra model được đóng gói cùng backend,
  dependency ONNX Runtime và SHA-256 trong metadata.
- API 413: frontend và backend phải cùng phiên bản; client mới gửi tối đa 64
  epoch float32/request. Không POST nguyên file lớn vào Function.
- Model ID không khớp: tải lại trang sau deploy và kiểm tra đủ frontend/backend
  trong cùng một deployment.
- API hết quota hoặc lỗi nền tảng: kiểm tra Usage/Status trên Vercel, thử lại
  khi tài nguyên được phục hồi. Gói miễn phí không có bảo đảm luôn sẵn sàng.

## Phương án dự phòng

Backend FastAPI này cũng chạy được trên một Python web service thông thường
bằng `uvicorn backend.app:app --host 0.0.0.0 --port <PORT>`.
Nếu sau này cần xử lý file lớn nguyên bản/EDF hay model DINOv3 nặng, nên đánh
giá lại compute và timeout trước khi chuyển; không mặc định serverless miễn phí
đủ cho mọi model. Render Free là một lựa chọn demo nhưng có sleep sau thời gian
không hoạt động: https://render.com/docs/free.
