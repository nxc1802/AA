SOTA benchmark đã được nhắc ở các mục:

1. Positioning against related work
    Đã liệt kê các paper/method gần nhất cần so sánh: SparseFool, GreedyFool, Homotopy sparse-imperceptible attack, AutoAdversary, SAIF, Sparse-RS, Sparse-PGD.
2. Recommended experiments and code changes
    Đã đề xuất mở rộng baseline từ FGSM/BIM/PGD sang sparse baselines mạnh hơn. Bản paper hiện tại của bạn mới so với FGSM, BIM, PGD trên CIFAR-10, nên chưa đủ để claim mạnh về sparse attack.  
3. Checklist trước submission
    Đã ghi rõ “Baselines nearest to contribution” hiện đang thiếu và cần thêm sparse baselines.

Tuy nhiên, chưa có một section riêng tên “SOTA Benchmark Protocol”. Nếu viết lại cho bản cải tiến paper, mình khuyên nên thêm hẳn một mục riêng như sau:

Nhóm benchmark	Method cần so sánh	Vai trò
Dense attacks	FGSM, BIM, PGD, AutoAttack	Kiểm tra attack strength chuẩn
Classical sparse	JSMA, One-pixel, SparseFool	Baseline lịch sử
Modern sparse white-box	GreedyFool, SAIF, Homotopy, AutoAdversary	Baseline gần với gradient/sparse optimization
Modern sparse black-box	Sparse-RS, CornerSearch	So sánh với sparse search mạnh
Closest SOTA	Sparse-PGD / Sparse-AutoAttack	Baseline bắt buộc vì gần nhất với bài của bạn

Trong đó, Sparse-PGD là baseline nguy hiểm nhất vì rất gần với hướng L_0-bounded PGD-like attack. Nếu không so với Sparse-PGD, reviewer rất dễ nói benchmark chưa đủ.