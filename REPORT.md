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
Theo GUIDE, Qwen3-VL-4B có khoảng 4,44B tham số: LLM và embedding khoảng 4,0B, ViT khoảng 415M, projector gồm merger khoảng 27M và DeepStack khoảng 82M. Đây là quy mô tham khảo, chưa phải số đếm thực tế trên runtime. E1 đóng băng ViT và projector, chỉ cập nhật LoRA LLM (33,030,144 tham số trainable theo evidence). E2 dự kiến mở thêm projector; E3 dự kiến thêm LoRA trên attention/MLP của ViT. Chưa có evidence hoàn chỉnh E2/E3.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate = 0.8907`, tức khoảng 34/311 output không parse được. Trong các output parse được, crack có 170 FP và 293 FN, F1 chỉ 0.0253; manhole có 73 FP và 144 FN. Đây là lỗi dự đoán sai/thừa hoặc bỏ sót box, tách biệt với lỗi JSON. File dự đoán thô chưa được lưu trong repo, nên chưa thể dẫn ví dụ nguyên văn hay khẳng định riêng lỗi sai thang.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nhãn YOLO chuẩn hoá được nhân 1000 trong `yolo_to_box`; prompt cũng yêu cầu 0–1000, và evidence dùng `coord: norm1000`. Nếu đầu ra pixel 640×360 bị chấm như 0–1000, cùng vật thể sẽ lệch vị trí/kích thước, giảm IoU và có thể bị tính thành FP/FN ở ngưỡng 0.5. Chưa chấm lại cùng predictions với `--coord pixel`, nên không có F1 đo thực tế cho trường hợp lệch thang.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Notebook tạo chuỗi prompt và chuỗi đầy đủ, kiểm tra prompt là tiền tố rồi gán `labels[:, :prompt_len] = -100`. Loss vì thế chỉ học JSON trả lời cùng token kết thúc `<|im_end|>`; kiểm tra giải mã labels của mẫu đầu cho đúng chuỗi đó. Nếu tính cả prompt, gradient sẽ tối ưu dự đoán lại chỉ dẫn và các token ngữ cảnh thay vì tập trung vào nhãn box; token ảnh không phải câu trả lời cần sinh.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Lượng tử hoá NF4 dành cho LLM; cấu hình bỏ qua `visual` và `lm_head`, vì trọng số 4-bit không phù hợp để cập nhật đầy đủ bằng optimizer thông thường. Ở E2/E3, notebook mở tham số projector và chuyển trainable weights sang float32 trước optimizer. Lần thử E2 với projector float16 đã dừng ở `GradScaler.unscale_` (`Attempting to unscale FP16 gradients`), là lý do thực tế cho sửa đổi này. Chưa có kết quả chấm E2/E3 để xác minh chất lượng.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các ảnh là frame video; chia ngẫu nhiên có thể đặt frame gần như giống nhau ở train và test, làm F1 cao giả vì rò rỉ bối cảnh quay. Split theo phiên giữ cả phiên trong một tập; Colab kiểm tra giao session bằng 0. Train: 1422 ảnh (pothole 1074, crack 1750, manhole 691); val: 276 (66, 470, 103); test: 311 (121, 299, 163). Val chỉ có 66 box pothole, vì vậy chỉ số lớp này dễ dao động.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 tăng F1 từ 0.0814 lên 0.2524 so với E0, đổi lại 33,030,144 tham số trainable, VRAM đỉnh 8.70 GB và 39.94 phút. Chưa có F1/VRAM/thời gian của E2–E3 nên không thể kết luận mở projector hay ViT có đáng chi phí.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
E0 F1 0.0814, CI95 [0.0563, 0.1126]; E1 F1 0.2524, CI95 [0.2163, 0.2894]. Hai khoảng không chồng lấn, ủng hộ E1 tốt hơn E0 trên split test này. Với 311 ảnh từ chỉ ba phiên test, CI bootstrap theo ảnh không loại trừ biến động giữa phiên quay khác; chênh lệch nhỏ giữa E1/E2/E3 cần evidence mới, không suy từ một lần chạy.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
E0 kém nhất ở crack (F1 0.0253); E1 kém nhất ở pothole (0.1393), sau đó crack (0.1791), còn manhole đạt 0.5418. Pothole E1 có 14 TP, 66 FP, 107 FN; crack có 61 TP, 321 FP, 238 FN. Box nhỏ (diện tích trung vị dataset khoảng 0.7% ảnh) khiến sai lệch ít pixel cũng giảm IoU mạnh; phân tích ngữ nghĩa và ảnh minh hoạ cần file predictions/ảnh thực tế, hiện chưa có trong repo.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Chưa chạy E4; notebook có cấu hình ablation rank 4/64 nhưng chưa có số liệu.

## Nhìn lại

Lần chạy E1 ban đầu lỗi vì `image_grid_thw` bị `squeeze(0)` thành tensor một chiều. Đã sửa trong notebook; E1 hoàn thành 300 bước. E2 lỗi `Attempting to unscale FP16 gradients`; notebook đã sửa nhưng chưa có kết quả E2.
