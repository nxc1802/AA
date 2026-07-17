# Kịch Bản Thuyết Trình: Thuật Toán Tấn Công Sparse PGD & Các Kỹ Thuật Phòng Thủ

Bản kịch bản chi tiết này được thiết kế để báo cáo dự án/bài báo, làm nổi bật những đóng góp khoa học cốt lõi của thuật toán đề xuất **Proposed-TopkPGD**.

---

## MỞ ĐẦU: Giới thiệu chung

### **Slide 1: Trang tiêu đề**
* **Lời thoại:**
  > *"Kính chào thầy cô và các bạn. Hôm nay, tôi xin phép trình bày báo cáo phân tích về đề tài: **'Thuật toán tấn công thưa Sparse PGD và các kỹ thuật phòng thủ tương ứng'**. Nghiên cứu này tập trung vào việc tối ưu hóa hiệu quả tấn công dạng thưa ($L_0$-bounded) nhằm duy trì tính tàng hình cao cho ảnh đối kháng, đồng thời đánh giá khả năng sinh tồn của chúng trước các bộ lọc phòng thủ phổ biến hiện nay."*

---

## PHẦN 1: CƠ SỞ LÝ THUYẾT (Làm nền tảng)

### **Slide 1.1: Chi tiết Thuật toán: Proposed-TopkPGD**
* **Lời thoại:**
  > *"Đầu tiên, về mặt lý thuyết, phương pháp đề xuất của chúng tôi mang tên **Proposed-TopkPGD**. Khác với các phương pháp PGD truyền thống rải nhiễu lên toàn bộ ảnh, thuật toán của chúng tôi giải quyết bài toán tấn công thưa bằng cách lọc ra các vị trí có độ nhạy cảm cao nhất. 
  > Cụ thể, sau khi tính Gradient của hàm mất mát, chúng tôi áp dụng lớp mặt nạ Top-k để đóng băng các pixel không quan trọng và chỉ cập nhật nhiễu trên các pixel có tác động mạnh nhất. Đặc biệt, cơ chế **Lập lịch $k$ động** giúp tỷ lệ pixel bị thay đổi giảm dần theo số bước lặp, giúp tối ưu hóa tối đa độ thưa (Sparsity) theo thời gian."*
* **Hành động:** Chỉ tay vào công thức lập lịch $k_t$ động để thể hiện tính tối ưu toán học.

### **Slide 1.2: Các Phương pháp Phòng thủ Tiền xử lý**
* **Lời thoại:**
  > *"Để chống lại các đòn tấn công thưa, các nghiên cứu trước đây thường sử dụng nhóm kỹ thuật tiền xử lý ảnh gốc. Ở đây chúng tôi khảo sát 4 phương pháp cốt lõi: 
  > 1. **Median Smoothing** lọc bỏ nhiễu thưa nhờ tính chất lọc trung vị.
  > 2. **Bit Depth Reduction** lượng hóa kênh màu để làm mất biên độ nhiễu tinh vi.
  > 3. **JPEG Compression** dùng DCT để lọc các nhiễu tần số cao.
  > 4. Và **Random Noise** nhằm phá cấu trúc nhiễu cố định."*

### **Slide 1.3: Tổng hợp các Cơ chế Phòng thủ Bền vững**
* **Lời thoại:**
  > *"Bên cạnh tiền xử lý, chúng tôi cũng đưa vào thực nghiệm các mô hình có khả năng phòng thủ nội tại (nội suy đặc trưng). Đáng chú ý nhất là **Adversarial Training (AT)** truyền thống và biến thể **TRADES** tối ưu hóa cán cân trade-off giữa độ chính xác ảnh sạch và độ bền bỉ. Ngoài ra còn có **Randomized Smoothing** bỏ phiếu đa số và **Feature Denoising** khử nhiễu trực tiếp trong không gian ẩn của mạng CNN."*

---

## PHẦN 2: THỰC NGHIỆM TẤN CÔNG (Làm nổi bật thuật toán của bạn)

### **Slide 2.1 & 2.2: Tấn công cơ bản trên Mô hình Chuẩn và Đối kháng**
* **Lời thoại:**
  > *"Bước sang phần thực nghiệm, trước hết là đánh giá các đòn tấn công cơ bản. Trên mô hình Chuẩn, các đòn như BIM hay PGD dễ dàng đạt ASR 100% nhưng độ thưa cực kỳ tệ ($<3\%$). 
  > Ngược lại, khi chuyển sang mô hình Đối kháng (AT), khả năng chống chịu của mô hình tăng mạnh, kéo ASR của các đòn tấn công thông thường giảm sâu xuống chỉ còn quanh mức 26% đến 31%."*

### **Slide 2.3: Phân tích quá trình hội tụ của PGD truyền thống**
* **Lời thoại:**
  > *"Một khảo sát nhỏ về số bước lặp cho thấy PGD hội tụ rất nhanh trên mô hình Chuẩn chỉ sau 5 bước lặp. Tuy nhiên, nó bắt buộc phải hy sinh hoàn toàn độ thưa để đạt được hiệu quả này."*

### **Slide 2.4: Quá trình hội tụ của Proposed-TopkPGD**
* **Lời thoại:**
  > *"Đây là kết quả khảo sát quá trình hội tụ của thuật toán đề xuất **Proposed-TopkPGD**. Nhìn vào bảng số liệu, quý hội đồng có thể thấy đòn tấn công của chúng tôi đạt độ hội tụ ASR rất nhanh chóng chỉ sau **5 bước lặp**. Quan trọng nhất là nhờ cơ chế lập lịch $k$ động, độ thưa (Sparsity) và chất lượng ảnh (PSNR) luôn được kiểm soát ổn định ở mức cao trong suốt quá trình tối ưu lặp."*
* **Hành động:** Nhấn mạnh việc chỉ cần 5 bước lặp (T=5) đã đạt hiệu quả tối ưu, chứng minh tốc độ xử lý nhanh.

### **Slide 2.5: Đánh giá nhược điểm thuật toán SOTA SparseFool**
* **Lời thoại:**
  > *"Trước khi so sánh trực tiếp, chúng tôi muốn làm rõ nhược điểm của đối thủ SOTA lớn nhất là **SparseFool**. Mặc dù thuật toán này đạt độ thưa cực kỳ ấn tượng ($>98\%$), nhưng nó lại vướng phải bài toán chi phí tính toán khi mất tới **160 đến 240 giây** để xử lý 1000 mẫu. Điều này khiến SparseFool hoàn toàn bất khả thi khi triển khai trong các ứng dụng thực tế đòi hỏi phản hồi nhanh."*

### **Slide 2.6 & 2.7: So sánh các Thuật toán Tấn công Thưa (Mô hình Chuẩn & Đối kháng)**
* **Lời thoại:**
  > *"Bây giờ, xin hãy nhìn vào bảng so sánh trực tiếp giữa thuật toán đề xuất của chúng tôi với hai đại diện SOTA là **Sparse-PGD** và **GreedyFool**.*
  > * Ở mức giới hạn gắt gao $k=0.1$ trên mô hình Chuẩn, thuật toán đề xuất đạt độ thưa vượt trội **88.23%** và chất lượng ảnh cực sạch **42.66 dB**, trong khi ASR vẫn giữ ở mức rất cao **83.65%**.*
  > * Trên mô hình Đối kháng ở mức $k=0.2$, trong khi đối thủ SOTA là Sparse-PGD bị sụp đổ hoàn toàn độ thưa xuống còn **9.98%** (tức là không còn tính thưa nữa), thuật toán của chúng tôi vẫn giữ vững độ thưa lý tưởng là **79.48%**."*
* **Hành động:** Chỉ tay vào các dòng được highlight màu vàng trên slide để người nghe thấy rõ khoảng cách chênh lệch lớn giữa thuật toán của bạn và SOTA.

---

## PHẦN 3: THỰC NGHIỆM PHÒNG THỦ (Chốt hạ độ bền vững)

### **Slide 3.1: Đánh giá Phòng thủ Tiền xử lý (Mô hình Chuẩn)**
* **Lời thoại:**
  > *"Một đòn tấn công thưa dù mạnh đến đâu cũng vô nghĩa nếu dễ dàng bị triệt tiêu bởi các bộ lọc cơ bản. Khi thử nghiệm phòng thủ tiền xử lý trên mô hình Chuẩn, các bộ lọc như Median Filter và JPEG Compression quả thực là khắc tinh của tấn công thưa."*

### **Slide 3.2: Phân tích chuyên sâu: Khả năng 'Xuyên Giáp' trước Phòng thủ**
* **Lời thoại:**
  > *"Tuy nhiên, đây chính là **điểm đặc sắc nhất** trong nghiên cứu của chúng tôi: **Khả năng sinh tồn (Robustness) của nhiễu trước phòng thủ**.*
  > * **Thứ nhất**, trước bộ lọc Randomized Smoothing, đòn GreedyFool bị vô hiệu hóa mạnh khi ASR giảm từ 94.9% xuống 77.9%. Trong khi đó đòn của chúng tôi hầu như không đổi (79.6%) và **chính thức vượt mặt GreedyFool**.*
  > * **Thứ hai**, trước nén JPEG, thuật toán đề xuất chỉ mất 50.8% sát thương, trong khi GreedyFool mất tới 64.4%, đưa cả hai về thế cân bằng mặc dù chúng tôi xuất phát điểm thấp hơn.*
  > * **Thứ ba**, đáng kinh ngạc nhất, khi cộng thêm Random Noise, đòn của chúng tôi được 'cộng hưởng' đẩy ASR lên mức **tuyệt đối 100%**.*
  > * **Giải thích:** Các SOTA cố nhồi nhét nhiễu biên độ lớn vào số ít pixel nên rất dễ bị làm mịn hoặc lọc sạch. Thuật toán của chúng tôi phân bổ thông minh, giữ ảnh tự nhiên và tạo cấu trúc nhiễu cực kỳ kiên cường."*
* **Nhấn mạnh:** Dùng giọng điệu tự tin, nhấn mạnh ba ý đầu dòng. Đây là phần "bán" bài báo thuyết phục nhất.

### **Slide 3.3 & 3.4: Đánh giá trên Mô hình AT và TRADES**
* **Lời thoại:**
  > *"Trên mô hình AT, độ bền bỉ nội tại của mô hình quá mạnh khiến các tiền xử lý không đem lại nhiều hiệu quả. Tuy nhiên trên mô hình TRADES, thuật toán của chúng tôi tiếp tục chứng minh đặc tính khó bị hóa giải khi giữ nguyên hiệu quả công phá 71.88% trước hầu hết các bộ lọc."*

### **Slide 3.5: Khảo sát khả năng phòng thủ của kiến trúc GG-SAT**
* **Lời thoại:**
  > *"Cuối cùng, chúng tôi khảo sát mô hình kiến trúc đồ thị GG-SAT. Mô hình này kháng cự tốt ở mức $k$ nhỏ, nhưng khi lượng nhiễu tăng dần lên $k \ge 0.5$, lớp phòng thủ này cũng dần bị khuất phục."*

---

## PHẦN 4: HƯỚNG PHÁT TRIỂN & KẾT LUẬN

### **Slide 4.1: Hướng phát triển: Phòng thủ với cấu trúc GGSAT**
* **Lời thoại:**
  > *"Từ những phát hiện trên, hướng đi tiếp theo của chúng tôi là thực hiện **GGSAT-Defense**. Bằng cách dùng chính đòn tấn công thưa bền bỉ Proposed-TopkPGD để huấn luyện đối kháng (Adversarial Training) cho mô hình GG-SAT. Mục tiêu là tạo ra một kiến trúc mạng AI có khả năng miễn nhiễm toàn diện với các đòn tấn công thưa mà không phải hy sinh độ chính xác trên ảnh sạch.*
  > *Trên đây là toàn bộ phần báo cáo của tôi. Xin chân thành cảm ơn thầy cô và các bạn đã chú ý lắng nghe. Tôi xin nhận các câu hỏi đóng góp ý kiến."*
