# Báo cáo Day 5 — điền trực tiếp trong fork của bạn

**Cách dùng:** Thay mọi dấu `…` bằng bài làm thật của bạn trước khi nộp link fork trên VLearn. Giữ nguyên bốn mục và bảng để coach đọc nhanh. Viết ngắn, cụ thể theo ảnh/vùng; không cần thuật ngữ chuyên sâu. Ví dụ trong [hướng dẫn mẫu](reports/REPORT_TEMPLATE.md) chỉ giúp hiểu cách điền, không phải câu trả lời để chép lại.

- Mã học viên theo lớp: 2A202602215
- Ngày / CVAT local: 2026-09-17 / http://localhost:8080
- Công cụ đã dùng: Polygon / Brush / CVAT UI / PyTorch baseline

Mã học viên là mã lớp cấp; không cần ghi họ tên trong report nếu kênh VLearn đã nhận diện bạn. Chỉ ghi công cụ thật sự đã dùng; không có SAM vẫn làm bài bình thường.

## 1. Bài đã nộp

Ghi tên ZIP đúng như file trong `submissions/` và số ảnh đã vẽ, Save. Chưa làm hoặc export lỗi thì ghi `chưa có`, không tạo ZIP rỗng. Cột điểm là điểm tối đa của task, **không phải điểm tự chấm**.

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | :---: | ---: |
| easy_semantic | easy_semantic.zip | 3 / 3 | 20 |
| medium_instance | medium_instance.zip | 3 / 3 | 32 |
| hard_panoptic | hard_panoptic.zip | 2 / 2 | 30 |
| cp1_holes | cp1_holes.zip | 1 / 1 | 3 |
| cp2_slice | cp2_slice.zip | 1 / 1 | 3 |
| cp5_occlusion | cp5_occlusion.zip | 1 / 1 | 3 |
| cp3_thin | cp3_thin.zip | 1 / 1 | 3 |
| cp4_curb | cp4_curb.zip | 1 / 1 | 3 |
| cp6_coverage | cp6_coverage.zip | 1 / 1 | 3 |
| **Tổng tối đa** | | **12 / 12** | **100** |

Nếu export lỗi, ghi task, dữ liệu đã Save đến đâu và lỗi đã báo coach.

## 2. Một quyết định trước khi dùng gợi ý

Chọn object đầu tiên bạn tự vẽ ở `medium_instance`, trước khi xem bất kỳ đề xuất tự động nào cho object đó. Ghi ảnh/vị trí đủ để tìm lại; “quy tắc biên” là lý do bạn chọn hoặc dừng mask ở ranh đó.

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: Ảnh `000000181542.jpg`, chiếc xe ô tô trắng đỗ ở tiền cảnh góc dưới bên phải màn hình.
- Class và quy tắc tôi dùng để chọn biên: Class `car`. Quy tắc: Chỉ bao lấy phần thân xe thực sự nhìn thấy được (vỏ xe, lốp, cản xe, gương chiếu hậu). Phần kính chắn gió trong suốt nhìn xuyên qua vẫn giữ nguyên trong mask (không khoét rỗng). Dừng đường biên tại mép dưới của lốp xe tiếp xúc mặt đường, kiên quyết không vẽ lấn sang vùng bóng đổ (shadow) trên mặt đường.
- Nếu dùng gợi ý sau đó: vùng gợi ý sai/đúng, hành động sửa/giữ và lý do: Gợi ý của model ban đầu bị lem một phần vệt bóng sẫm màu dưới gầm xe vào mask xe. Tôi đã dùng brush xóa phần bóng lem vào nền đường và giữ lại phần viền thân xe thực tế.
- Nếu không dùng gợi ý: không dùng.

## 3. Một lỗi tôi tìm thấy và sửa

Chọn một lỗi **có thật** trong bài. Nếu công cụ lỗi khiến bạn chưa sửa được, ghi rõ đã thử gì và cần coach hỗ trợ gì; không ghi “đã sửa” khi chưa sửa.

- Task/ảnh/vùng: Task `cp2_slice`, ảnh `000000017627.jpg`, cụm hai xe ô tô cùng class đỗ sát cạnh nhau ở khu vực trung tâm.
- Lỗi thuộc loại: gộp-tách (hai đối tượng độc lập bị gộp dính vào nhau thành một mask duy nhất).
- Bằng chứng tôi nhìn thấy: Khoảng cách giữa hai sườn xe rất hẹp và có bóng tối, khiến thuật toán dự đoán nối liền hai thân xe thành 1 thực thể.
- Quy tắc và hành động sửa: Quy tắc instance segmentation yêu cầu mỗi cá thể vật lý phải là một instance riêng biệt. Tôi đã phóng to (zoom), dùng đường cắt dọc theo khe hẹp giữa hai cửa xe để tách thành 2 polygon riêng biệt với 2 ID độc lập trong danh sách Objects.
- Sau sửa đã Save và export lại chưa? Đã Save trong CVAT và export lại ra `submissions/cp2_slice.zip`.

Nếu bạn **đã xem Summary tự đánh giá trên GitHub Actions hoặc tự chạy script**, ghi ngắn một kết quả liên quan lỗi vừa sửa (ví dụ task, metric trước/sau nếu có): Chạy scorer cục bộ `python3 -m scoring.scorecard --group tiers` đạt 54.5 / 82 điểm trên 3 tier, kết quả clean không có cờ cảnh báo (review_flags rỗng). Scorecard ba tier tối đa **82**, không phải điểm cuối trên 100. Không tự ghi PASS/top 3/bonus; người phụ trách xác nhận theo tiêu chí lớp. Không đưa file ground truth vào fork.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Mỗi ca là một **vùng cụ thể** khiến bạn phải cân nhắc hai cách hiểu. Ghi dấu hiệu nhìn thấy hoặc quy tắc đã dùng, rồi nêu quyết định hoặc câu hỏi cho coach. Không cần ba lỗi; ca đã quyết định được cũng hợp lệ.

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| 1 | `cp4_curb` (`7d83710e-4697c3b2.jpg`) - Ranh giới giữa mép đường và vỉa hè | Phân định theo màu sắc mặt đường (chỗ nhựa đen sẫm) hay theo cấu trúc gờ bó vỉa | Quan sát thấy có dải gờ bó vỉa phân tách rõ rệt cao độ giữa lòng đường xe chạy và phần vỉa hè dành cho người đi bộ | Quyết định: Gán toàn bộ dải gờ bó vỉa nâng cao vào `sidewalk`, chỉ tính phần mặt đường phẳng dưới rãnh là `road`, không phụ thuộc vào màu sắc của vật liệu |
| 2 | `cp1_holes` (`000000144300.jpg`) - Ô kính cửa sổ xe ô tô trong suốt nhìn thấy hậu cảnh | Khoét rỗng (hole) phần kính trong suốt vì nhìn thấy nền phía sau, hay giữ nguyên trong thân xe | Quy tắc bài cp1_holes: "windows/gaps stay inside the mask — do NOT cut them out" | Quyết định: Giữ nguyên vùng kính cửa sổ xe nằm bên trong mask của `car`, không khoét rỗng |
| 3 | `cp5_occlusion` (`000000336232.jpg`) - Xe ô tô bị cột che khuất ở giữa thành 2 mảng | Tách thành 2 object xe riêng biệt hay gộp chung thành 1 instance đa vùng (multi-polygon) | Quy tắc occlusion: cùng một thực thể vật lý bị che khuất thì các mảng nhìn thấy vẫn thuộc về 1 instance duy nhất | Quyết định: Gán cả hai mảng của xe bị che vào chung 1 instance trong danh sách Objects, không sinh thêm object thứ hai |
