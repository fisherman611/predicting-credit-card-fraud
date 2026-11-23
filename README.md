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

Lần chia thứ hai (seed 2026) huấn luyện lại v1 và cả hai ứng viên trên cùng các fold.

| Ứng viên, seed 2026 | OOF ROC AUC | Ca CVV bằng 0 bị bỏ sót ở top 5% | Cảnh báo nhầm ở top 5% |
| --- | ---: | ---: | ---: |
| v1 | 0.95049 | 2.606 | 26.473 |
| v2, tương tác | 0.95038 | 2.609 | 26.493 |
| v2, tương tác và mô hình CVV riêng | 0.95030 | 2.620 | 26.501 |

Cả hai lần chia đều cho AUC thấp hơn và bỏ sót thêm ca CVV bằng 0 ở v2. Bài nộp hiện tại giữ v1 với public AUC **0.95122**. [Phiên đối chứng seed 2026](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v2-seed-check).

## Thử nghiệm v3 với cụm chữ số theo vị trí

LightGBM được bổ sung target encoding của cụm 2–3 chữ số và phần thập phân theo độ dài phần nguyên. Ba cột thử nghiệm là `merchant_risk_score`, `velocity_score` và `account_balance_usd`. V3 giữ nguyên Digit GLM và tỷ trọng phối hợp 80/20 của v1.

| Ứng viên, seed 42 | OOF ROC AUC | AUC trong nhóm CVV bằng 0 | AUC giữa các nhóm CVV | Ca CVV bằng 0 bị bỏ sót ở top 5% |
| --- | ---: | ---: | ---: | ---: |
| v1 | 0.95057 | 0.93129 | 0.97053 | 2.607 |
| v3, cụm chữ số | 0.95047 | 0.93110 | 0.97051 | 2.635 |

AUC giữa các nhóm tính trên các cặp gian lận và hợp lệ thuộc hai nhóm CVV khác nhau. Hai phiên chạy khớp ID, nhãn và fold trên cả 699.960 dòng, nên có thể so sánh trực tiếp. V3 giảm AUC tổng và bỏ sót thêm 28 ca ở nhóm mục tiêu. [Phiên v1 đối chứng](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v3-control) · [Phiên thử cụm chữ số](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v3-positional-digits).

## Hiệu chỉnh phần sai số của v1

Mỗi fold OOF của v1 được chia đôi có phân tầng. Hai LightGBM nông học phần hiệu chỉnh logit từ một nửa và dự đoán nửa còn lại. Mô hình v1 của fold đã học ở ngoài cả hai nửa. Hệ số hiệu chỉnh cố định là 0,5. Với tập test, mã cộng phần hiệu chỉnh trung bình của các cây vào logit của dự đoán v1.

Ở seed 42, OOF AUC tăng từ **0.95057** lên **0.95064**. Số ca CVV bằng 0 bị bỏ sót ở top 5% giảm từ 2.607 xuống 2.601. AUC trong nhóm này tăng từ 0.93129 lên 0.93143 và AUC giữa các nhóm ở mức 0.97053. [Phiên hiệu chỉnh seed 42](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v3-residual).

Trên seed 2026, hiệu chỉnh giảm OOF AUC từ **0.95049** xuống **0.95038** và bỏ sót thêm 4 ca CVV bằng 0 ở top 5%. AUC trong nhóm này giảm từ 0.93116 xuống 0.93097 và AUC giữa hai nhóm giảm từ 0.97051 xuống 0.97047. Mức tăng ở seed 42 không lặp lại, nên bài nộp vẫn dùng v1 với public AUC **0.95122**. [Phiên hiệu chỉnh seed 2026](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v3-residual-2026).

## Chạy trên Kaggle

Từ thư mục gốc, dùng Kaggle CLI để chạy mã trên CPU với dữ liệu cuộc thi tự động gắn vào. Trong `kaggle/kernel-metadata.json`, đổi phần tài khoản của `id` nếu chạy bằng tài khoản khác.

```bash
kaggle kernels push -p kaggle
kaggle kernels status huthnhlng/s6e11-fraud-v3-positional-digits
kaggle kernels output huthnhlng/s6e11-fraud-v3-positional-digits -p submissions
```

Kết quả gồm `metrics.json` chứa AUC theo fold, AUC tổng, AUC theo nhóm CVV và số ca bỏ sót ở top 5%. `submission.csv` theo mẫu của cuộc thi. Phiên v3 lưu thêm `oof.npz`, gồm ID, fold, nhãn, nhóm CVV và điểm OOF của từng giao dịch. Dữ liệu và dự đoán nằm trong đầu ra riêng tư của Kaggle hoặc thư mục `submissions/` được Git bỏ qua.

Để chạy cùng mã trên máy có dữ liệu đã tải:

```bash
python -m pip install -r requirements.txt
python kaggle/train.py --data data --output submissions
python kaggle/train.py --experiment v3 --compare-baseline --data data --output submissions/v3-comparison
python kaggle/train.py --experiment v1 --save-oof --data data --output submissions/v1
python kaggle/train.py --experiment residual --control submissions/v1 --data data --output submissions/residual
```
