# Kết quả kiểm tra

Ngày 19/10/2026, trên Windows, Node.js 22.16.0 và Python 3.13.9.
Thư viện cài từ package-lock.json và requirements-dev.txt của project.

| Kiểm tra | Kết quả |
| --- | --- |
| TypeScript + Vite production build | Đạt |
| Frontend tests | 3/3 đạt |
| Backend/model/API tests | 5/5 đạt |
| Model CNN thật và bộ dự đoán tham chiếu 24 epoch | Cùng nhãn; confidence trong sai số 1e-5 |
| Chia 400 epoch (file .npy 4.800.128 byte) thành nhiều phần | 7 request, mỗi request nhỏ hơn 1 MiB |
| Client thực → Vite proxy → FastAPI → ONNX CPU | Đạt, trả đúng 400 nhãn tham chiếu |
| Trình duyệt: chọn file .npy → phân tích trên server → hypnogram | Đạt với file EEG mô phỏng 24 epoch |
| vercel.json đối chiếu schema chính thức | Đạt |
| Wheel Linux x86_64 Python 3.12 của numpy 2.2.6 và onnxruntime 1.22.1 | Có trên registry |

Preview kết quả ở [preview.png](preview.png).

Chưa tạo deployment trên tài khoản Vercel, chưa có production URL được xác nhận.
Kiểm tra schema và wheel không thay thế một lượt build/run thật trên Vercel.
Sau khi import project, cần kiểm tra /api/health và phân tích mẫu trên production
theo [DEPLOYMENT.md](DEPLOYMENT.md).

Các phép thử dùng EEG tổng hợp và kết quả model đã xuất; không chứng minh độ chính
xác trên bệnh nhân hay cam kết uptime của hạ tầng miễn phí.
