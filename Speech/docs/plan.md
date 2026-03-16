Dưới đây là **Kế hoạch triển khai chi tiết (Detailed Project Plan)** đã được điều chỉnh, **loại bỏ hoàn toàn phần tấn công đa mục tiêu (MORE/REDO/EOS)** để tập trung tối đa vào độ sâu toán học của thuật toán **Mảnh đối kháng không gian con (Subspace Hessian-Patch)** nhằm làm suy giảm độ chính xác (tăng Word Error Rate - WER) và cơ chế phòng thủ. 

---

### Giai đoạn 1: Khởi tạo, Thiết lập môi trường và Tái tạo Baseline (Tháng 1 - 2)
**Mục tiêu:** Xây dựng cơ sở hạ tầng thử nghiệm và kiểm chứng các thuật toán nền tảng.
*   **Nhiệm vụ 1.1 - Thiết lập dữ liệu và mô hình:** 
    *   Triển khai bộ mô hình nhận dạng giọng nói Whisper của OpenAI (từ cấu hình Tiny đến Large) để đánh giá khả năng chuyển giao của Patch.
    *   Chuẩn bị và tiền xử lý bộ dữ liệu âm thanh LibriSpeech và LJ-Speech (đưa về định dạng chuỗi thời gian 1D, tần số lấy mẫu 16kHz).
*   **Nhiệm vụ 1.2 - Tái tạo các thuật toán Baseline:** 
    *   Lập trình các thuật toán tấn công đối kháng bậc nhất truyền thống: FGSM và PGD (với 10 bước lặp, PGD-10).
    *   Tái tạo thuật toán SCORPIO cơ bản (Frank-Wolfe kết hợp Forward Euler) trên dữ liệu ảnh để đảm bảo tính chính xác của mã nguồn trước khi chuyển sang xử lý âm thanh 1D.
*   **Cột mốc (Milestone):** Hoàn thiện môi trường code và có bảng số liệu benchmark ban đầu của PGD trên hệ thống Whisper.

### Giai đoạn 2: Phát triển Thuật toán Mảnh đối kháng Bậc hai Cục bộ (Tháng 3 - 5)
**Mục tiêu:** Cài đặt luồng thuật toán "Tối ưu hóa không gian con" (Subspace Optimization) đột phá đã đề xuất.
*   **Nhiệm vụ 2.1 - Định vị Patch tĩnh thời gian tuyến tính $O(N)$:** 
    *   Lập trình thuật toán lan truyền ngược để tính toán Gradient bậc 1 toàn cục ($g_0$).
    *   Tích hợp thuật toán **Prefix-Sum + Sliding Window** trên mảng độ lớn $|g_0|$ để tìm ra vị trí Mảnh đối kháng (kích thước cố định $<10\%$ âm thanh) có độ dốc cực đại.
*   **Nhiệm vụ 2.2 - Khóa không gian con và Xấp xỉ Hessian:** 
    *   Xây dựng cơ chế Mặt nạ (Mask $M$) để loại bỏ các phép tính ngoài vùng Patch.
    *   Viết hàm tính toán Tích Hessian-Vector (HVP) sử dụng Sai phân hữu hạn (Forward Euler) giới hạn nghiêm ngặt trong không gian con của Patch.
*   **Nhiệm vụ 2.3 - Tích hợp Frank-Wolfe (FW):** 
    *   Sử dụng $g_0$ làm điểm khởi tạo hoàn hảo cho nhiễu.
    *   Áp dụng FW để giải bài toán tuyến tính hóa phụ ở mỗi bước lặp, đảm bảo cập nhật nhiễu không cần phép chiếu (projection-free) và tự động cập nhật bước nhảy $\gamma_k$.
*   **Cột mốc:** Hoàn thành module **Hessian-Patch-Optimizer**, chứng minh được module này có thể hội tụ thành công trên 1D waveform với chi phí bộ nhớ tối thiểu.

### Giai đoạn 3: Đánh giá Sức mạnh Tấn công Mảnh đối kháng (Tháng 6 - 8)
**Mục tiêu:** Đánh giá hiệu quả làm sai lệch độ chính xác và chi phí tính toán của Hessian-Patch so với các phương pháp truyền thống.
*   **Nhiệm vụ 3.1 - Tối ưu hóa Hàm mất mát (Loss Function):** 
    *   Sử dụng hàm Cross-Entropy để thực hiện tấn công Không có đích (Untargeted Attack - đẩy quỹ đạo giải mã ra khỏi nhãn đúng) hoặc Có đích (Targeted Attack - ép mô hình nhận diện ra một từ/câu cụ thể).
*   **Nhiệm vụ 3.2 - Tiến hành thử nghiệm Tấn công:** 
    *   Thực thi Hessian-Patch-Attack trên toàn bộ tập dữ liệu kiểm thử.
*   **Nhiệm vụ 3.3 - Đánh giá hiệu năng:** 
    *   *Về sức mạnh tấn công:* Đo lường mức độ suy giảm độ chính xác thông qua Tỷ lệ lỗi từ (Word Error Rate - WER).
    *   *Về độ tàng hình:* Đo lường chỉ số SNR (Signal-to-Noise Ratio) của đoạn âm thanh sau khi chèn Patch.
    *   *Về chi phí tính toán:* So sánh tổng số lần truyền xuôi-ngược (forward-backward passes). Mục tiêu là chứng minh Hessian-Patch-Attack (dùng FW+FE) chỉ tốn $N+1$ lần truyền nhưng đạt hiệu quả phá vỡ bằng hoặc hơn PGD tốn 10 lần truyền.
*   **Cột mốc:** Báo cáo chi tiết chứng minh Hessian-Patch-Attack vượt trội hơn PGD-10 về khả năng đánh lừa Whisper và tốc độ chạy.

### Giai đoạn 4: Nghiên cứu Phòng thủ Bậc hai Cục bộ (Tháng 9 - 10)
**Mục tiêu:** Cung cấp giải pháp an ninh (Robustness) cho Whisper dựa trên độ cong cục bộ.
*   **Nhiệm vụ 4.1 - Phạt độ cong cục bộ (Local Curvature Penalty):** 
    *   Trích xuất giới hạn trên của các giá trị riêng (eigenvalues) của ma trận Hessian bên trong không gian con của Patch.
    *   Tích hợp thành phần phạt này ($\gamma K_{local}$) vào hàm Loss để tinh chỉnh (fine-tune) lại mô hình Whisper, làm "phẳng" ranh giới quyết định tại các điểm yếu.
*   **Nhiệm vụ 4.2 - Kiểm tra Gradient Masking & Chứng chỉ bền vững:** 
    *   Sử dụng công cụ tấn công hộp đen (như SimBA) để xác minh rằng bề mặt loss của Whisper đã thực sự phẳng lại chứ không phải bị "che giấu gradient" (gradient obfuscation).
    *   Tính toán Chứng chỉ bền vững (CRC) để đảm bảo mô hình an toàn trước mọi Patch trong bán kính $L_2$ cho trước.
*   **Cột mốc:** Hoàn thành mô hình Whisper được gia cố (Defended Whisper) có khả năng vô hiệu hóa Mảnh đối kháng cục bộ.

### Giai đoạn 5: Phân tích Diễn giải, Tổng hợp và Công bố (Tháng 11 - 12)
**Mục tiêu:** Nâng tầm dự án bằng các phân tích diễn giải AI và chuẩn bị bài báo khoa học.
*   **Nhiệm vụ 5.1 - Phân tích Khả năng Diễn giải (Interpretability):** 
    *   Trích xuất Bản đồ Nổi bật (Saliency Maps) từ mô hình phòng thủ. 
    *   Sử dụng phương pháp Giải thích phản thực tế (Contrastive Explanations) để tìm ra các "Pertinent Positives" (đặc trưng ngữ âm quan trọng giúp nhận diện đúng) và "Pertinent Negatives" (đặc trưng cần thêm vào để đổi nhãn). Phân tích xem Mảnh đối kháng thực chất đang phá hoại các đặc trưng ngữ âm nào của Whisper.
*   **Nhiệm vụ 5.2 - Viết báo cáo và Bản thảo (Manuscript):** 
    *   Tổng hợp số liệu thành các biểu đồ so sánh mức độ bền vững (Adversarial Accuracy vs Epsilon).
    *   Hoàn thiện mã nguồn (clean code) để chuẩn bị công khai.
    *   Soạn thảo bài báo khoa học chuẩn bị nộp cho các hội nghị học thuật hàng đầu chuyên ngành (như ICLR, NeurIPS, hoặc ICASSP).
*   **Cột mốc:** Hoàn thiện Proposal dự án và Bản thảo bài báo khoa học (Draft Paper).
