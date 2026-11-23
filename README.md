# Predicting Credit Card Fraud

Dự án tham gia cuộc thi [Predicting Credit Card Fraud — Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11).

## Mục tiêu

Xây dựng mô hình dự đoán xác suất một giao dịch thẻ tín dụng là gian lận. Cuộc thi đánh giá bài nộp bằng **ROC AUC** giữa xác suất dự đoán và nhãn thực tế.

## Dữ liệu

Dữ liệu được cung cấp trên [trang cuộc thi](https://www.kaggle.com/competitions/playground-series-s6e11/data). Để tải dữ liệu bằng Kaggle CLI, trước tiên hãy tham gia cuộc thi, chấp nhận các điều khoản và cấu hình thông tin xác thực Kaggle theo [hướng dẫn của Kaggle](https://github.com/Kaggle/kaggle-api#authentication).

```bash
python -m pip install kaggle
mkdir -p data
kaggle competitions download -c playground-series-s6e11 -p data
unzip data/playground-series-s6e11.zip -d data
```

Thư mục `data/` được Git bỏ qua để tránh đưa dữ liệu cuộc thi vào repository. Kiểm tra quy định cuộc thi trước khi chia sẻ dữ liệu.

## Bài nộp

Tạo tệp CSV có một dòng cho mỗi `id` trong tập test. Cột `is_fraud` chứa **xác suất**, không phải nhãn phân loại cứng:

```csv
id,is_fraud
699960,0.3
699961,0.9
```

Có thể nộp tệp qua trang cuộc thi hoặc Kaggle CLI sau khi tạo dự đoán:

```bash
kaggle competitions submit -c playground-series-s6e11 -f submissions/submission.csv -m "Baseline"
```

## Mô hình và validation

- **LightGBM** dùng cây nông, đặc trưng gốc và target encoding cho các nhóm giá trị. Target encoding của tập fit được tính qua ba fold nội bộ. Validation và test dùng từ điển học từ tập fit của fold ngoài.
- **Digit GLM** dùng logistic regression với one-hot encoding của các nhóm chữ số, phần thập phân và độ lớn của các trường số.
- **Ensemble** phối hợp logit theo tỷ lệ 80/20 giữa LightGBM và GLM. Cả ba ứng viên được chấm ROC AUC trên cùng 5 fold stratified. Ứng viên có AUC out-of-fold cao nhất được chọn và dự đoán test được lấy trung bình qua 5 fold.

Mô hình sử dụng dữ liệu chính thức của cuộc thi. Ý tưởng cây nông và nhóm chữ số tham khảo [Scale-Aware Digit GLM](https://www.kaggle.com/code/denpugovkin/s6e11-scale-aware-digit-glm). Cách phối hợp nhiều nguồn tín hiệu tham khảo [Board Model + RealMLP](https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95560). Các dự đoán đều được tạo bởi mã huấn luyện trong repo.

## Chạy trên Kaggle

Từ thư mục gốc, dùng Kaggle CLI để chạy mã trên CPU với dữ liệu cuộc thi tự động gắn vào. Trong `kaggle/kernel-metadata.json`, đổi phần tài khoản của `id` nếu chạy bằng tài khoản khác.

```bash
kaggle kernels push -p kaggle
kaggle kernels status huthnhlng/s6e11-credit-card-fraud-cv
kaggle kernels output huthnhlng/s6e11-credit-card-fraud-cv -p submissions
```

Kết quả gồm `metrics.json` chứa AUC theo fold và AUC tổng, cùng `submission.csv` theo mẫu của cuộc thi. Notebook chạy ở chế độ riêng tư.

Để chạy cùng mã trên máy có dữ liệu đã tải:

```bash
python -m pip install -r requirements.txt
python kaggle/train.py --data data --output submissions
```
