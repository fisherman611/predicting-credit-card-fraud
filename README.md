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

## Kết quả Kaggle

Phiên bản 1 chạy trên CPU trong khoảng 9 phút 29 giây với 699.960 dòng train và 299.984 dòng test.

| Ứng viên | ROC AUC out-of-fold, 5 fold | Public ROC AUC |
| --- | ---: | ---: |
| LightGBM | 0.95052 | |
| Digit GLM | 0.94602 | |
| **Ensemble 80/20** | **0.95057** | **0.95122** |

Kaggle chấm xong ngày 23/11/2026. Bài nộp `58752634` dùng ensemble. [Phiên chạy Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-credit-card-fraud-cv) · [Danh sách bài nộp](https://www.kaggle.com/competitions/playground-series-s6e11/submissions).

## Thử nghiệm v2 cho nhóm CVV bằng 0

Bản tương tác thêm ba cặp `CVV × merchant risk`, `CVV × velocity` và `merchant risk × velocity` vào LightGBM. Merchant risk và velocity được chia theo phân vị của tập fit của mỗi fold. Mã tính target encoding qua ba fold nội bộ. Bản có mô hình riêng kết hợp thêm một LightGBM chỉ học trên nhóm CVV bằng 0, với trọng số logit cố định 25% cho nhóm đó.

| Ứng viên, seed 42 | OOF ROC AUC | Ca CVV bằng 0 bị bỏ sót ở top 5% | Cảnh báo nhầm ở top 5% |
| --- | ---: | ---: | ---: |
| v1 | 0.95057 | 2.607 | 26.471 |
| v2, tương tác | 0.95043 | 2.616 | 26.497 |
| v2, tương tác và mô hình CVV riêng | 0.95035 | 2.623 | 26.503 |

Top 5% là 34.998 giao dịch có điểm cao nhất trong OOF. Hai ứng viên v2 đều thua v1 trên lần chia này. [Phân tích lỗi v1](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-error-audit) · [Phiên chạy v2](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v2-cvv).

## Chạy trên Kaggle

Từ thư mục gốc, dùng Kaggle CLI để chạy mã trên CPU với dữ liệu cuộc thi tự động gắn vào. Trong `kaggle/kernel-metadata.json`, đổi phần tài khoản của `id` nếu chạy bằng tài khoản khác.

```bash
kaggle kernels push -p kaggle
kaggle kernels status huthnhlng/s6e11-fraud-v2-cvv
kaggle kernels output huthnhlng/s6e11-fraud-v2-cvv -p submissions
```

Kết quả v2 gồm `metrics.json` chứa AUC theo fold, AUC tổng và số ca bỏ sót ở top 5%. Mỗi ứng viên có một CSV theo mẫu cuộc thi, và `submission.csv` chứa ứng viên có OOF AUC cao nhất trong phiên chạy. Notebook chạy ở chế độ riêng tư.

Để chạy cùng mã trên máy có dữ liệu đã tải:

```bash
python -m pip install -r requirements.txt
python kaggle/train.py --data data --output submissions
python kaggle/train.py --experiment v2 --seed 2026 --compare-baseline --data data --output submissions/seed-2026
python kaggle/train.py --experiment v1 --data data --output submissions/v1
```
