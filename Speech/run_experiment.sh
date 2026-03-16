#!/bin/bash

# Kích hoạt môi trường ảo
source venv/bin/activate

# Xoá log cũ nếu có
rm -f evaluation_results_pgd200.log

echo "=========================================================="
echo "Bắt đầu chạy thực nghiệm PGD-200 trên 1D Waveform (75 samples)"
echo "Sử dụng thiết bị (Device): MPS/CUDA (Tùy hệ thống)"
echo "Log sẽ được lưu đồng thời vào terminal và file evaluation_results_pgd200.log"
echo "=========================================================="

# Chạy script và pipe log ra file
mkdir -p output_experiment
python scripts/evaluate_baselines.py 2>&1 | tee evaluation_results_pgd200.log

echo "=========================================================="
echo "Hoàn thành! Kết quả tổng quát:"
cat output_experiment/summary.json
echo "=========================================================="
echo "Dữ liệu chi tiết nằm trong thư mục output_experiment/"
echo "- Audio đối kháng: output_experiment/audio/"
echo "- Visualization: output_experiment/visualization/"
echo "- Bảng kết quả: output_experiment/results_detailed.csv"
