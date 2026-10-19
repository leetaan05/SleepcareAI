# Kiểm thử bản chọn CNN / SLARDA + DINOv3 QuickRun

Ngày 19/10/2026, môi trường Windows. Bản sao website được dùng để cài PyTorch CPU
trong môi trường ảo tách khỏi thư mục Google Drive; mã và model cuối cùng được
sao chép về `SleepcareWeb` sau khi kiểm thử. Python local 3.11 và ONNX Runtime
1.22.1. PyTorch/Transformers chỉ dùng trong môi trường export kiểm chứng,
không nằm trong runtime production.

| Kiểm tra | Kết quả |
| --- | --- |
| SHA-256 bundle DINOv3 đã sao chép | `706e9c1486b18ad772b5e89a8575f6823a90f427c49a8b26746d105363f0d4e5` |
| SHA-256 model ONNX production | `4502866c503ad9b1a543e58f7b13d3a0fca146eeccdc0a60c88059293d8b921b` |
| Xuất ONNX từ bundle và kiểm tra bằng ONNX Runtime | Đạt; sai số logits tối đa khoảng `1.70e-5` trên fixture, không cần PyTorch/Transformers ở runtime |
| So sánh bản web với QuickRun gốc trên 24 epoch EEG tổng hợp | Nhãn giống nhau; logits và confidence trong sai số 1e-5 |
| Backend/API unit tests | 10/10 đạt; gồm hai model, danh sách model, giới hạn batch, model thiếu, 429 |
| Frontend tests | 6/6 đạt; gồm chọn đúng endpoint, 4/64 epoch mỗi batch, version, báo cáo |
| TypeScript + Vite production build | Đạt |
| npm audit sau cập nhật Vite 7.3.7 | 0 vulnerability được báo |

Vercel entrypoint đã gom từ ba file API thành `api/index.py`. Production dùng
ONNX Runtime và loại file `.pt` khỏi Function; không còn phụ thuộc PyTorch nên
không cần Large Functions beta.

Sau khi chép 21 file đã sửa (gồm bundle) về đúng `SleepcareWeb` trên ổ G:,
SHA-256 của từng file đều khớp bản đã kiểm thử. Chạy lại kiểm thử trực tiếp
trong thư mục đích: backend **10/10 đạt**, frontend **6/6 đạt**.

Fixture `tests/fixtures/dinov3-parity.json` được tạo độc lập từ
`SLARDA_DINOv3/QuickRun/dinov3_eeg.py` gốc, nạp checkpoint raw Meta rồi nạp
toàn bộ state trong bundle. Nó dùng mẫu EEG tổng hợp `eeg-demo.npy`, chuẩn hóa
population standard deviation với floor 1e-6. Không có dữ liệu người thật.

Trong lượt này chưa xác nhận deploy production trên tài khoản Vercel; còn phải
kiểm tra Build Logs, `/api/health` và phân tích mẫu trên URL production.
Vercel có thể dùng Python 3.14; requirements đã có nhánh wheel tương thích cho
3.14, trong khi Python 3.12/3.13 giữ bản đã kiểm thử. Runtime Vercel và hạn mức
RAM của Function chưa được kiểm thử trực tiếp.
Kiểm tra giao diện bằng tab trình duyệt riêng bị hệ thống duyệt tự động từ chối
do giới hạn sử dụng; các tương tác qua UI vì vậy chưa được xác nhận hoàn chỉnh.

QuickRun là checkpoint thử tích hợp trên 80 epoch, test target 10 mẫu. Các kiểm
thử trên chứng minh code nạp và chạy đúng bundle, không chứng minh độ chính xác
y khoa hoặc chất lượng trên tập Sleep-EDF đầy đủ.
