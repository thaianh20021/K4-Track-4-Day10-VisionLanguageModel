Họ tên: Đặng Thái Anh
`dataset_version`: d10-21e4070301

Trạng thái: đã xác minh evidence E0 và E1 từ lần chạy Colab. E2–E4 chưa có bằng chứng.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | không áp dụng | không áp dụng | 0.0814 | 0.0563–0.1126 | 0.8553 | 0.1419 | 0.0253 | 0.1490 |
| E1 | LoRA LLM | 33,030,144 | 8.70 | 39.94 | 0.2524 | 0.2163–0.2894 | 1.3087 | 0.1393 | 0.1791 | 0.5418 |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Chưa hoàn tất E2/E3. E1 chỉ cập nhật LoRA trên LLM; E2 mở thêm projector, E3 thêm LoRA ViT (theo cấu hình notebook). Chưa đo tham số từng khối trên lần chạy này.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate = 0.8907`. F1 lớp crack chỉ 0.0253, với 170 FP và 293 FN; cần xem lại output thô để xác định và minh hoạ hai kiểu lỗi.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nhãn và prompt dùng thang 0–1000. Chưa thực hiện thí nghiệm chấm sai thang để báo F1.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Loss chỉ tính trên token trả lời của assistant; prompt và token ảnh là ngữ cảnh đầu vào, không phải mục tiêu sinh.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Projector được giữ ngoài lượng tử hoá và mở huấn luyện float32 ở E2/E3; các setup này chưa được chạy và xác minh.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Split theo phiên quay để các frame gần nhau không lọt qua train/test. Train: 1422 ảnh (pothole 1074, crack 1750, manhole 691); val: 276 (66, 470, 103); test: 311 (121, 299, 163). Colab đã kiểm tra không trùng session giữa các split.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 tăng F1 từ 0.0814 lên 0.2524 so với E0. Chưa đủ evidence E2–E3 để kết luận về projector hoặc ViT.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
E0 F1 0.0814, CI95 [0.0563, 0.1126]; E1 F1 0.2524, CI95 [0.2163, 0.2894]. Hai khoảng không chồng lấn; chưa thể so sánh E2/E3.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E0, crack kém nhất (F1 0.0253), tiếp theo pothole (0.1419), manhole (0.1490). Chưa có ảnh minh hoạ lỗi.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Chưa chạy E4; notebook có cấu hình ablation rank 4/64 nhưng chưa có số liệu.

## Nhìn lại

Lần chạy E1 ban đầu lỗi vì `image_grid_thw` bị `squeeze(0)` thành tensor một chiều. Đã sửa trong notebook; E1 hoàn thành 300 bước. E2 lỗi `Attempting to unscale FP16 gradients`; notebook đã sửa nhưng chưa có kết quả E2.
