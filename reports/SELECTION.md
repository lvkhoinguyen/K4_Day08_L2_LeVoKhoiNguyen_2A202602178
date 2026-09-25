# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:
1. **frame_0182.jpg** (Rank 1, Score: 0.9591, t = 72.8s, U = 0.9182, A = 1.0, D = 1.0, 28 boxes, 18 ambiguous): Điểm tổng hợp cao nhất trong toàn bộ pool, độ bất định cực cao với 18 box mập mờ (conf quanh 0.5), phân bố xe phức tạp ở dải tốc độ cao.
2. **frame_0369.jpg** (Rank 2, Score: 0.9324, t = 147.6s, U = 0.9315, A = 0.8889, D = 1.0, 43 boxes, 16 ambiguous): Mật độ xe rất đông (43 boxes), U rất cao phản ánh tình trạng xe dày đặc, nhiều xe ở xa bị che khuất và lóa đèn pha.
3. **frame_0380.jpg** (Rank 3, Score: 0.9170, t = 152.0s, U = 0.9340, A = 0.8333, D = 1.0, 40 boxes, 15 ambiguous): Cách frame_0369 4.4s (> MIN_GAP_S = 2.0s), mật độ phương tiện cao, có nhiều xe tải và xe khách hỗn hợp.
4. **frame_0326.jpg** (Rank 4, Score: 0.9155, t = 130.4s, U = 0.9310, A = 0.8333, D = 1.0, 39 boxes, 15 ambiguous): Cảnh đường cao tốc ban đêm có nhiều xe di chuyển ở làn đối diện, đèn pha chiếu thẳng vào camera gây khó cho mô hình.
5. **frame_0099.jpg** (Rank 8, Score: 0.9063, t = 39.6s, U = 0.9460, A = 0.7778, D = 1.0, 29 boxes, 14 ambiguous): Ưu tiên frame này thay vì các frame rank 5-7 (`frame_0331.jpg`, `frame_0372.jpg`, `frame_0312.jpg`) vì phân bổ thời gian ở đầu video (t = 39.6s), giúp tăng tính đa dạng thời gian (Diversity) cho mô hình học thay vì tập trung dồn dập vào cụm t = 120s - 150s. Đồng thời tránh ảnh gần trùng: loại bỏ `frame_0372.jpg` (Rank 6, t = 148.8s) vì chỉ cách `frame_0369.jpg` 1.2s (< 2.0s), việc gán cả hai sẽ lãng phí ngân sách và gây overfit vào một cảnh quay cục bộ.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
- **frame_0182.jpg**: Rank 1, U = 0.9182, 18 box mơ hồ. Trên contact sheet cho thấy vùng đèn pha phản chiếu trên mặt đường ướt khiến mô hình phân vân giữa vệt sáng và xe thật.
- **frame_0331.jpg**: Rank 5, Score: 0.9154, t = 132.4s, n_boxes = 47, n_ambiguous = 18. Mật độ xe cao nhất trong lô (47 xe), trên contact sheet xuất hiện nhiều xe kích thước nhỏ ở hậu cảnh sát dải phân cách bị AI nhận diện trùng lặp box.
- **frame_0392.jpg**: Rank 15, Score: 0.8874, t = 156.8s, U = 0.9747 (điểm bất định U cao nhất trong top 15). Mặc dù số box mơ hồ thấp hơn (12 box), U cao cho thấy các dự đoán xe ở rìa ảnh có confidence dao động mạnh quanh ngưỡng phân lớp.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- **frame_0372.jpg** (Rank 6, Score: 0.9101, t = 148.8s): Có điểm tổng hợp nằm trong top 6 nhưng **không được chọn** vào lô vì vi phạm ràng buộc khoảng cách thời gian tối thiểu `MIN_GAP_S = 2.0s` (nó chỉ cách `frame_0369.jpg` được chọn ở rank 2 đúng 1.2 giây). Đây là trường hợp ảnh gần trùng (near-duplicate), hình ảnh camera hầu như không thay đổi, các xe cùng vị trí nên việc bỏ qua giúp tiết kiệm chi phí gán nhãn và tránh trùng lặp dữ liệu huấn luyện.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Phép chọn theo độ bất định (Uncertainty Sampling: `score = W_U*U + W_A*A + W_D*D`) chỉ tìm ra các mẫu mà mô hình hiện tại **phân vân nhất hoặc khó dự đoán nhất**, chứ không đảm bảo rằng khi gán nhãn xong thì mô hình sau fine-tune chắc chắn sẽ tăng hiệu năng (AP50).
- Các mẫu có U cao đôi khi là do nhiễu sensor, chói đèn cực mạnh hoặc các vật thể không thể gán nhãn dứt khoát (out-of-distribution), khiến việc gán nhãn có thể đưa thêm nhiễu vào tập huấn luyện nếu không tuân thủ nghiêm ngặt guideline. Ngoài ra, việc tập trung vào mẫu khó có thể gây hiện tượng "catastrophic forgetting" trên các trường hợp xe thông thường nếu tập dữ liệu huấn luyện quá nhỏ.
