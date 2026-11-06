# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU / T4 GPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145 (Google Colab Runtime)

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1:  
   `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
   **Trả lời:** Record này nói về **cả bức ảnh** (Image-level classification). Mô hình phân loại ảnh gán một xác suất phân phối lớp cho toàn bộ ma trận pixel của bức ảnh và xếp hạng các nhãn có khả năng nhất đại diện cho tổng thể khung hình, hoàn toàn không xuất ra tọa độ bounding box hay vị trí không gian của từng vật thể riêng biệt bên trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
   **Trả lời:** Danh sách tên lớp đi kèm **taxonomy/checkpoint** cố định từ trước (ở đây là bộ phân loại ImageNet-1K gồm 1.000 danh mục đã được xác định khi huấn luyện mô hình). Mô hình không thể tự sáng tạo thêm lớp mới nằm ngoài taxonomy đã được cấu hình trong checkpoint `yolo11n-cls.pt`.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
   **Trả lời:** Cần ghi cả 3 trường vì các taxonomy khác nhau (như COCO-80, Pascal VOC, ImageNet-1K, OpenImages) có thể đặt cùng một tên lớp nhưng gán mã `class_id` khác nhau, hoặc cùng một tên nhưng mang định nghĩa và phạm vi ngữ nghĩa bao hàm khác nhau. Việc lưu trữ đầy đủ `taxonomy_name`, `class_id` và `class_name` đảm bảo tính toàn vẹn nguồn gốc (data provenance), tránh xung đột khi ánh xạ hoặc tích hợp dữ liệu vào các hệ sinh thái khác.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
   **Trả lời:** Guideline cần đưa ra tiêu chí ưu tiên định lượng rõ ràng: (1) Quy tắc chủ thể chiếm ưu thế diện tích (vật thể chiếm tỷ lệ diện tích pixel lớn nhất ở tiền cảnh, ví dụ > 50%), (2) Quy tắc trọng tâm bức ảnh (vật thể nằm ở vùng trung tâm mà nhiếp ảnh gia lấy nét chính), hoặc (3) Bảng phân cấp ưu tiên danh mục (Class Priority Hierarchy, ví dụ: ưu tiên phương tiện giao thông hơn kiến trúc đường phố).
6. Vì sao không được lấy `score` của model làm ground truth?  
   **Trả lời:** Vì `score` (confidence score) chỉ là xác suất softmax do mô hình ước lượng dựa trên các đặc trưng học được từ dữ liệu huấn luyện trong quá khứ. Điểm số này có thể bị sai lệch (overconfident, hallucination, hoặc domain shift khi gặp bối cảnh lạ). Ground truth bắt buộc phải là sự thật khách quan do con người thẩm định và xác nhận dựa trên guideline quy chuẩn.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record:  
   `class_name`=person ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
   **Trả lời:** Bounding box có góc trên-trái tại tọa độ pixel (x=385.33, y=69.24) tính từ góc trên cùng bên trái của bức ảnh (0, 0); góc dưới-phải tại tọa độ pixel (x=498.92, y=348.92). Hộp bao có chiều rộng 113.58 pixel và chiều cao 279.68 pixel, ôm trọn dáng người đứng bên phải gian bếp từ đỉnh đầu xuống thân người.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
   - Ngưỡng **0.20**: có **17** prediction (gồm: `person`, `bowl`, `bowl`, `oven`, `oven`, `person`, `bowl`, `bowl`, `cup`, `cup`, `bowl`, `spoon`, `potted plant`, `spoon`, `dining table`, `spoon`, `bottle`).
   - Ngưỡng **0.60**: có **6** prediction (chỉ còn các đối tượng rõ ràng nhất: `person`, `bowl`, `bowl`, `oven`, `oven`, `person`).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
   **Trả lời:** Khi hạ ngưỡng (tăng recall nhưng giảm precision), mô hình sinh ra rất nhiều hộp dương tính giả (False Positives) đối với các đồ vật nhỏ hoặc vùng vân bề mặt mờ nhạt (ví dụ model đoán nhầm bóng đổ/đồ bếp thành `spoon`, `potted plant`). Người reviewer phải tốn công rà soát từng hộp, xác minh độ chính xác của nhãn và thủ công xóa bỏ các hộp rác/hộp nhận diện sai.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
   **Trả lời:** **Quy tắc Hộp Bao Khít Biên Ngoài (Tight Bounding Box Rule):** Bounding box phải bao trọn toàn bộ các điểm cực biên có thể nhìn thấy của vật thể (dung sai tối đa 2 đến 3 pixel tính từ mép ngoài), không được cắt xén phần nhô ra của vật thể (tay, chân, quai cầm) và tuyệt đối không để chừa khoảng trống nền thừa quá 5% diện tích hộp.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
   **Trả lời:**  
   - Vật thể: Người phụ nữ / cánh tay ở góc dưới cùng bên trái bức ảnh (tọa độ `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`), bị mép ảnh cắt ngang và chỉ thấy một phần cẳng tay/áo.
   - Xử lý: **Viết thành quy tắc trong Guideline**.
   - Vì sao: Vì hiện tượng vật thể bị cắt mép (truncation) hoặc che khuất (occlusion) là tình huống xuất hiện lặp đi lặp lại rất thường xuyên trong thực tế chụp ảnh đời thực. Guideline cần chuẩn hóa một ngưỡng định lượng cụ thể (ví dụ: "Chỉ gán nhãn nếu phần nhìn thấy của vật thể đạt từ 20% trở lên và có đủ đặc trưng nhận diện; nếu dưới 20% thì bỏ qua hoặc gắn cờ truncated"). Không cần leo thang hỏi Lab Coach cho những hiện tượng mang tính phổ quát này.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:  
   `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`: `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
   **Trả lời:** Polygon cung cấp đường biên bao chính xác đến từng pixel (pixel-level contour) theo đúng hình dạng học thực tế của vật thể. Trong khi bounding box chỉ là một hình chữ nhật thô chứa cả nhiều pixel của phông nền (background noise), polygon tách biệt hoàn toàn ranh giới giữa bề mặt đối tượng và môi trường xung quanh, đặc biệt có giá trị với các vật thể có hình dáng uốn lượn, nghiêng, hoặc có phần rỗng.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
   **Trả lời:** `instance_id` là mã định danh duy nhất dùng để phân biệt từng thực thể cá biệt cùng loại xuất hiện trong một mẫu ảnh (ví dụ phân biệt người thứ nhất `kitchen-001` với người thứ hai `kitchen-006` cùng chung `class_name = person`). Nó không phải `class_id` (vì `class_id` là mã phân loại danh mục chung cho toàn bộ giống loài/chủng loại, ví dụ `class_id=0` là person), và không phải `tracking_id` (vì `tracking_id` dùng để bám vết duy trì danh tính của một vật thể chuyển động xuyên suốt qua nhiều khung hình video liên tiếp theo thời gian).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
   **Trả lời:** **Quy tắc Bám Sát Mép Tiếp Xúc Ranh Giới (Boundary Contiguity Rule):** Các điểm mút của đa giác (polygon vertices) phải đặt trực tiếp trên đường tiếp giáp giữa màu sắc/độ sáng của vật thể và nền, mật độ điểm phải đủ dày tại các cung cong uốn lượn (khoảng cách giữa các đỉnh không quá 5 pixel), tuyệt đối không để đa giác tự cắt chéo (self-intersecting) và không bao trùm các khoảng trống nhìn xuyên qua thân vật.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
   **Trả lời:**  
   - Chỗ chạm nhau: Các chiếc bát/đĩa xếp chồng lên nhau trên giá bếp hoặc khu vực chân ly/bát tiếp xúc trực tiếp với mặt bàn.
   - Xử lý: **Dùng Guideline**.
   - Vì sao: Đây là quan hệ hình học vật lý điển hình (Spatial Contact / Overlap). Guideline quy định: "Khi hai vật thể cùng hoặc khác lớp chạm sát nhau, người gán nhãn phải vẽ đường ranh giới chia cắt tại mép tiếp xúc thực tế của vật ở phía trước (tiền cảnh); vật bị che khuất chỉ vẽ phần diện tích thực sự nhìn thấy được, không vẽ bù phần bị che khuất phía sau". Chỉ escalation khi hai vật thể có cùng màu sắc/chất liệu hòa trộn vào nhau đến mức mắt người bình thường không thể phân biệt nổi ranh giới.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| :--- | :--- | :--- | :--- | :--- |
| Phân loại ảnh | Một nhãn danh mục duy nhất cho toàn bộ ảnh (`class_id`, `class_name`). | Ảnh `traffic` có cả xe hơi, xe bus, người đi bộ, tòa nhà; model đoán top-1 là `cab` (0.51) và top-2 là `minibus` (0.16) gây tranh cãi về chủ thể chính. | Nhìn toàn cảnh bức ảnh, áp dụng quy tắc chủ thể ưu tiên diện tích để gán đúng 1 nhãn chuẩn. | Kiểm tra bức ảnh có vi phạm quy tắc phân loại không, đối chiếu nhãn với bảng hướng dẫn taxonomy. |
| Phát hiện vật thể | Danh sách các hộp chữ nhật pixel `[x_min, y_min, x_max, y_max]` kèm `class_id` cho từng vật. | Ảnh `kitchen` tại ngưỡng 0.20 bị nhận nhầm thìa (`spoon`) và cây cảnh (`potted plant`); người ở mép trái chỉ lộ cánh tay bị cắt biên (`[0.12, 263.18, 61.55, 310.78]`). | Vẽ hộp bao khít 4 cạnh của từng vật thể nhìn thấy, kiểm tra cờ vật thể bị che/cắt mép. | Kiểm tra độ khít (tightness) của bounding box, rà soát vật thể bị bỏ sót (False Negatives) và xóa box rác (False Positives). |
| Instance segmentation | Tập hợp các tọa độ đa giác pixel `[[x1, y1], [x2, y2], ...]` bao bọc viền từng instance riêng biệt. | Ảnh `kitchen` có các chén bát đặt sát cạnh nhau; phần ranh giới tiếp xúc giữa bát và kệ bếp bị mờ do ánh sáng chói, đường viền mask khó tách rời. | Chấm các điểm polygon bám sát đường bao thực tế của từng đối tượng riêng lẻ theo chiều kim đồng hồ. | Phóng to kiểm tra độ mịn đường viền, rà soát lỗi đa giác tự cắt chéo, không để mask lấn sang vật thể khác hoặc lọt nền trống. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
   **Trả lời:** Tuyệt đối chỉ sử dụng các ảnh mẫu công khai có giấy phép Creative Commons (CC BY 2.0) đã được kiểm định mã băm SHA-256; không tải lên Google Colab hoặc kho lưu trữ GitHub công khai bất kỳ hình ảnh cá nhân, thông tin nhận dạng khuôn mặt, biển số xe hay dữ liệu nội bộ doanh nghiệp.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
   **Trả lời:** Dừng ngay lập tức các thao tác xử lý dữ liệu và báo cáo trực tiếp cho Giảng viên / Lab Coach phụ trách lớp học để được hướng dẫn cô lập và xử lý theo quy trình.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
