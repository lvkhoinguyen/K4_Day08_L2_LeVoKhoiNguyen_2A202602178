# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Võ Khôi Nguyên

Công cụ gán nhãn đã dùng: CVAT (Ultralytics YOLO Detection 1.0)

## 1. Dữ liệu và cách chia tập

- **Lý do chia tập theo trục thời gian kèm vùng đệm (buffer zone):**  
  Video đường cao tốc là chuỗi khung hình liên tục theo thời gian, các khung hình cạnh nhau (cách nhau phần mười giây) có tính tương quan cực kỳ cao (temporal correlation / data leakage). Nếu chia ngẫu nhiên (random split), các khung hình rất giống nhau của cùng một chiếc xe chạy qua camera sẽ vừa rơi vào tập huấn luyện vừa rơi vào tập kiểm thử. Do đó, việc chia tập theo trục thời gian (Time-series split) và chèn thêm vùng đệm (buffer zone khoảng vài giây) giúp cách ly hoàn toàn dòng phương tiện giữa tập pool và tập test, đảm bảo tính độc lập và khả năng tổng quát hóa (generalization) thực tế.
- **Nếu chia ngẫu nhiên, số đo kiểm thử bị lệch thế nào và vì sao:**  
  Số đo (AP50, Precision, Recall) sẽ bị **thổi phồng quá mức (over-optimistic bias)**. Mô hình chỉ việc "học vẹt" ghi nhớ hình ảnh của các phương tiện quen thuộc đã thấy ở các frame liền kề trong tập train thay vì học các đặc trưng thị giác tổng quát của xe trong đêm.

## 2. Mô hình khởi đầu lạnh (cold start)

- **Dòng vòng 0 từ `reports/rounds_table.md`:**
  ```markdown
  | 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
  ```
- **Sự không khớp nhãn tham chiếu dựa trên `outputs/compare_round0.jpg`:**  
  Mô hình cold start (pretrained COCO yolov8n) phát hiện khá tốt các xe kích thước trung bình và lớn ở cự ly gần (Precision đạt 0.925), nhưng bỏ sót nghiêm trọng các xe ở xa hoặc xe trong điều kiện ánh sáng yếu (chỉ thấy cụm đèn hoặc bị khuất).
- **Độ phủ (Recall) theo kích thước xe:**  
  - Xe nhỏ (`small`): Recall chỉ đạt **0.182** (18.2%), nghĩa là bỏ sót hơn 80% xe nhỏ ở xa.
  - Xe vừa (`medium`): Recall đạt **0.547** (54.7%).
  - Xe lớn (`large`): Recall đạt **0.561** (56.1%).  
  Điều này cho thấy mô hình tiền huấn luyện COCO tiêu chuẩn không nhạy với các phương tiện nhỏ bị nhòe và chói sáng trong bối cảnh camera giám sát giao thông ban đêm.
- **Trường hợp cần người rà lại nhãn tham chiếu:**  
  Trong các tình huống xe ở rất xa chỉ còn hai chấm đèn le lói hoặc vệt sáng phản chiếu kéo dài trên dải phân cách: nhãn tham chiếu test (vốn được tạo tự động bởi mô hình khác) có thể đã gán nhãn cho các đốm sáng không rõ thân xe, hoặc bỏ sót xe bị che khuất. Trước khi kết luận mô hình dự đoán sai (FP hay FN), cần có chuyên viên thẩm định lại nhãn tham chiếu theo chuẩn `GUIDELINE_LABEL.md`.

## 3. Chiến lược chọn mẫu

- **Giải thích công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`:**  
  - $U$ (Uncertainty): Đo lường độ bất định của mô hình trên khung hình (dựa trên phân phối xác suất và độ tin cậy của các box quanh ngưỡng 0.5). Trọng số $W_U = 0.5$ ưu tiên lấy các ảnh mà mô hình phân vân nhất.  
  - $A$ (Ambiguity): Tỷ lệ các box mơ hồ (có confidence nằm trong khoảng biên $[0.25, 0.75]$) so với tổng số box. Trọng số $W_A = 0.3$ giúp chọn các ảnh có nhiều đối tượng khó phân định.  
  - $D$ (Diversity): Đo khoảng cách thời gian từ khung hình đang xét tới các khung hình đã được chọn trước đó. Trọng số $W_D = 0.2$ khuyến khích lấy mẫu phân bố đều theo thời gian.  
  - `MIN_GAP_S = 2.0s`: Là ngưỡng thời gian tối thiểu bắt buộc giữa hai ảnh được chọn trong cùng một đợt. Vai trò của nó là loại bỏ triệt để các khung hình gần trùng (near-duplicates), tránh lãng phí chi phí nhân công gán nhãn vào những cảnh hầu như không có thông tin mới.
- **Minh chứng 4 frame từ `reports/SELECTION.md` và `outputs/selection_round1.csv`:**  
  - `frame_0182.jpg` (Rank 1, Score 0.9591, t = 72.8s, U = 0.9182, 18 box mơ hồ): Được chọn vì độ bất định và mật độ box mơ hồ cao nhất toàn tập.  
  - `frame_0369.jpg` (Rank 2, Score 0.9324, t = 147.6s, U = 0.9315, 43 box): Mật độ giao thông cực kỳ đông đúc, cung cấp nhiều ca xe nhỏ ở hậu cảnh.  
  - `frame_0099.jpg` (Rank 8, Score 0.9063, t = 39.6s): Được chọn vào lô vì có khoảng cách thời gian tốt ở giai đoạn đầu video, bổ sung tính đa dạng bối cảnh cho tập train.  
  - `frame_0372.jpg` (Rank 6, Score 0.9101, t = 148.8s): Dù có điểm tổng hợp rất cao nằm trong top 6 nhưng **bị loại bỏ** do chỉ cách `frame_0369.jpg` đúng 1.2s (< 2.0s), minh chứng cho việc hệ thống chủ động tránh chi phí rà nhãn cho ảnh thừa thãi.
- **Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không? Vì sao?**  
  **Không.** Điểm bất định chỉ phản ánh sự lúng túng của mô hình hiện tại tại không gian đặc trưng đó. Nếu ảnh chứa quá nhiều nhiễu (chói đèn cực mạnh, nhòe ống kính, vật thể không xác định), việc đưa vào gán nhãn có thể đưa thêm nhãn nhiễu hoặc gây overfit cục bộ, thậm chí làm giảm chất lượng tổng thể của mô hình.

## 4. Các vòng học chủ động (active learning)

- **Bảng tổng hợp từ `reports/rounds_table.md`:**
  | vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
  | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
  | 1 | yolov8n fine-tune vong 1..1 | 12 | 284 | 0.397 | -0.374 | 1.000 | 0.072 | 0.134 | 0.000 | 0.051 | 0.342 |

- **Mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md`):**  
  - Tổng số ảnh trong lô: 12 ảnh.  
  - Model đề xuất ban đầu: 169 box. Sau khi rà sửa: 284 box.  
  - Box giữ nguyên (`accepted`): 162 box (96% box AI gợi ý là hợp lệ).  
  - Box chỉnh sửa (`edited`): 0 box.  
  - Box xóa (`deleted` - False Positive của AI): 7 box (chủ yếu là vệt đèn pha rọi mặt đường, bóng xe, hoặc dải phân cách phát sáng).  
  - Box thêm mới (`added` - False Negative của AI): 122 box (các xe ở xa, xe tối màu ở mép làn hoặc xe bị che khuất).
- **Biến thiên AP50 và các nhóm xe:**  
  - AP50 giảm từ 0.771 xuống 0.397 ($\Delta = -0.374$) tại ngưỡng conf 0.25. Tuy nhiên Precision tăng tuyệt đối lên **1.000** (không có FP nào).  
  - Do tập huấn luyện mới chỉ có 12 ảnh (quá nhỏ so với tập dữ liệu COCO gốc hàng trăm nghìn ảnh), mô hình sau 50 epoch fine-tune đã trở nên cực kỳ thận trọng: nó chỉ dự đoán các box có độ chắc chắn rất cao (TP = 29, FP = 0), dẫn đến Recall sụt giảm mạnh (từ 0.489 xuống 0.072). Nhóm xe lớn giữ được Recall 0.342, trong khi nhóm xe nhỏ và vừa bị sụt giảm Recall ở ngưỡng conf 0.25.
- **Phân tích ca thay đổi trên `outputs/compare_round1.jpg`:**  
  Trên ảnh `frame_0250` và `frame_0350`, mô hình round 1 không còn vẽ các box nhầm vào vệt đèn phản chiếu trên mặt đường (loại bỏ hoàn toàn FP so với cold start). Tuy nhiên, các xe kích thước nhỏ ở làn đối diện xa bị mô hình bỏ qua do chưa đủ số lượng mẫu xe nhỏ đa dạng để khái quát hóa.
- **Đối chiếu ba lớp thông tin:**  
  - *Quan sát độc lập (`BLIND_SCAN.md` trên `frame_0099.jpg`)*: Đếm thấy 15 xe, chỉ ra vùng xe bị khuất trong tối và xe ở cuối đường chỉ thấy đèn.  
  - *Sửa pre-label (`REVIEW_LOG.csv` và `round1_diff.md`)*: Bổ sung 9 box bị bỏ sót ở `frame_0099.jpg`, xóa 1 box giả vệt đèn ở `frame_0182.jpg` và xóa 3 box trùng ở `frame_0331.jpg`.  
  - *Kết quả sau train*: Mô hình học được tính kỷ luật cao về Precision (P = 1.000), loại bỏ box rác nhưng cần thêm dữ liệu để khôi phục Recall cho xe nhỏ.  
- **Mô tả ca khó theo guideline:**  
  Ca xe ở rất xa chỉ nhìn thấy hai chấm đèn le lói ở làn trái (chiều cao box xấp xỉ 16 pixel). Theo guideline, trường hợp này cho phép bỏ qua nếu chiều cao dưới 16 pixel để tránh đưa nhiễu nhãn vào mô hình.

## 5. Kết luận và giới hạn

- **Đánh giá kết quả và quyết định:**  
  Vòng 1 đã hoàn thành đầy đủ chu trình: chọn mẫu có chủ đích $\rightarrow$ rà sửa nhãn $\rightarrow$ huấn luyện và đánh giá khách quan. Quyết định: **Dừng lại ở Vòng 1** để tập trung phân tích sâu nguyên nhân sụt giảm Recall (over-fitting do cỡ lô 12 ảnh nhỏ) thay vì chạy thêm vòng một cách mù quáng.
- **Đề xuất hai ca cho vòng tiếp theo:**  
  1. Các khung hình có mật độ xe nhỏ dày đặc ở dải xa (như `frame_0107.jpg` và `frame_0326.jpg`) để huấn luyện chuyên sâu cho dải `small` và `medium`.  
  2. Các khung hình có chói đèn pha trực diện mạnh nhưng không có xe phía trước để mô hình học âm tính (negative samples) tốt hơn.  
  *Cân nhắc chi phí & trùng cảnh*: Chi phí rà các frame đông xe rất cao (30-40 box/ảnh), đồng thời cần duy trì `MIN_GAP_S >= 2.0s` để không lặp lại các cảnh quay trùng.
- **Giới hạn của tập kiểm thử:**  
  Tập test chỉ gồm 20 ảnh với 403 box tham chiếu được tạo tự động bởi mô hình khác. Việc AP50 đo mức độ trùng khớp với bộ nhãn tham chiếu này không đồng nghĩa tuyệt đối với chất lượng thực tế ngoài đời.
- **Quy trình kiểm tra nếu AP50 giảm:**  
  Khi AP50 giảm, trước khi train thêm cần: (1) Kiểm tra xem có hiện tượng triệt tiêu Recall do ngưỡng confidence hay không; (2) So sánh trực quan trên `compare_round1.jpg` xem mô hình có thực sự dự đoán sai hay do nhãn tham chiếu test chưa chuẩn; (3) Tinh chỉnh lại learning rate hoặc số epoch để tránh hiện tượng catastrophic forgetting trọng số gốc COCO.
