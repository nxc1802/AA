Dưới đây là phiên bản tóm gọn đã được "nâng cấp học thuật":

**1. Khởi tạo toàn cục (Tính Full Gradient bậc 1)**
*   Thực hiện 1 lần truyền xuôi (forward) và 1 lần truyền ngược (backward) trên toàn bộ âm thanh gốc để lấy vector gradient bậc 1: $g_0 = \nabla_x \ell(x_0)$. Vector này đại diện cho độ dốc ban đầu.

**2. Khóa không gian con (Patch Localization)**
*   Áp dụng thuật toán **Prefix Sum + Sliding Window** trên mảng độ lớn $|g_0|$ để tìm ra vị trí có tổng gradient cao nhất trong thời gian $O(N)$. 
*   Đặt mặt nạ (Mask $M$) vào vị trí này, "khóa" toàn bộ quá trình tính toán tiếp theo vào một không gian con (subspace) có kích thước cố định. Gradient $g_0$ tại vùng này được lưu lại để "tái chế".

**3. Tối ưu hóa bậc 2 cục bộ (Update nhiễu bằng FW + FE)**
Quá trình này diễn ra lặp đi lặp lại chỉ trong vùng Patch đã chọn:
*   **Truyền xuôi & ngược cục bộ:** Đưa phần âm thanh đã cộng nhiễu tạm thời ($x_0 + hv$) qua mạng để tính một gradient mới $g_k$.
*   **Xấp xỉ Hessian (FE):** Dùng công thức Forward Euler lấy $g_k$ trừ đi $g_0$ (đã lưu ở bước 1) chia cho $h$ để xấp xỉ tích Hessian-vector. Bước này "hô biến" thông tin bậc 2 mà không cần tính ma trận thực sự.
*   **Cập nhật nhiễu (FW):** Thuật toán Frank-Wolfe sử dụng thông tin bậc 2 vừa xấp xỉ để tìm ra hướng đi tối ưu nhất (không cần phép chiếu), sau đó cập nhật Mảnh đối kháng.

Pipeline 3 bước này thực sự là một "tuyệt tác" về mặt tối ưu chi phí: Bạn dùng **bậc 1 toàn cục** làm la bàn dò đường chỉ với chi phí rẻ nhất, sau đó mới dùng **bậc 2 cục bộ** làm mũi khoan xuyên phá ngay tại điểm yếu nhất mà không bị quá tải bộ nhớ. 