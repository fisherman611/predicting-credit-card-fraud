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

Cả hai lần chia đều cho AUC thấp hơn và bỏ sót thêm ca CVV bằng 0 ở v2. V2 được nộp thêm để đối chiếu điểm public. [Phiên đối chứng seed 2026](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v2-seed-check).

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

Trên seed 2026, hiệu chỉnh giảm OOF AUC từ **0.95049** xuống **0.95038** và bỏ sót thêm 4 ca CVV bằng 0 ở top 5%. AUC trong nhóm này giảm từ 0.93116 xuống 0.93097 và AUC giữa hai nhóm giảm từ 0.97051 xuống 0.97047. Mức tăng ở seed 42 không lặp lại. Bản hiệu chỉnh seed 42 được nộp thêm để đối chiếu điểm public. [Phiên hiệu chỉnh seed 2026](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-v3-residual-2026).

## Đối chiếu bài nộp trên public leaderboard

| Bản nộp, seed 42 | ID bài nộp | OOF ROC AUC | Public ROC AUC |
| --- | ---: | ---: | ---: |
| v1, ensemble 80/20 | 58752634 | 0.95057 | **0.95122** |
| v2, thêm tương tác | 58756159 | 0.95043 | 0.95076 |
| v3, hiệu chỉnh điểm v1 | 58756163 | 0.95064 | 0.95119 |

V1 vẫn đứng đầu trong ba bài nộp. V3 hiệu chỉnh thấp hơn v1 0.00003 trên public leaderboard. [Danh sách bài nộp trên Kaggle](https://www.kaggle.com/competitions/playground-series-s6e11/submissions).

## Thử nghiệm mạng neural

RealMLP và TabM dùng cùng năm fold seed 42 với v1. Mỗi fold dùng 10% phần train làm tập chọn epoch. Đầu vào gồm cột gốc, phần thập phân và tần suất xuất hiện của giá trị đã làm tròn, tính trong phần huấn luyện. Khả năng phối hợp được kiểm tra bằng ba công thức rank blend cố định, 75/25 cho từng mạng với v1 và 70/15/15 cho cả hai mạng. [RealMLP trong pytabkit](https://github.com/dholzmueller/pytabkit) · [TabM](https://github.com/yandex-research/tabm).

| Mô hình, seed 42 | OOF ROC AUC | AUC trong nhóm CVV bằng 0 | Ca CVV bằng 0 bị bỏ sót ở top 5% |
| --- | ---: | ---: | ---: |
| **v1** | **0.95057** | **0.93129** | **2.607** |
| RealMLP | 0.94197 | 0.91856 | 2.855 |
| TabM-mini | 0.94594 | 0.92439 | 2.789 |
| v1 + RealMLP | 0.95017 | 0.93079 | 2.648 |
| v1 + TabM-mini | 0.95052 | 0.93117 | 2.622 |
| v1 + cả hai mạng | 0.95033 | 0.93098 | 2.632 |

V1 + TabM-mini là blend gần v1 nhất. AUC tổng giảm 0.00005 và bỏ sót thêm 15 ca CVV bằng 0 ở top 5%. V1 vẫn là bài nộp có điểm public cao nhất với 0.95122.

Mã khớp ID, nhãn và fold với OOF của v1 trước khi chấm. Báo cáo gồm ROC AUC tổng, AUC trong nhóm CVV bằng 0 và số ca của nhóm đó bị bỏ sót trong top 5%.

```bash
python -m pip install pytabkit==1.8.0 rtdl-num-embeddings==0.0.12
python kaggle/neural/train.py --model realmlp --data data --output submissions/realmlp
python kaggle/neural/train.py --model tabm --data data --output submissions/tabm
python kaggle/neural/compare.py --v1 submissions/v1 --realmlp submissions/realmlp --tabm submissions/tabm
```

Đầu ra của từng mạng gồm `metrics.json`, `oof.npz` và `submission.csv`. Mã so sánh tạo `comparison.json` và một CSV cho mỗi blend. [Phiên RealMLP](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-realmlp) · [Phiên TabM](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-tabm).

## Holdout cố định và huấn luyện lại mạng neural

Holdout được khóa bằng stratified split seed 8128 trước vòng thử này. V1 học trên 559.968 dòng và được chấm trên 139.992 dòng còn lại. Mỗi mạng dùng 55.997 dòng trong phần train để chọn epoch, rồi huấn luyện lại trên đủ 559.968 dòng. Hai dự đoán trước và sau refit được đối chiếu trên cùng ID, nhãn và nhóm CVV.

RealMLP dùng refit của pytabkit. Wrapper TabM trong pytabkit chưa hỗ trợ refit, nên thử nghiệm dùng TabM-mini từ gói chính thức và hai lượt train với cùng kiến trúc. Bản trước và sau refit của TabM trong bảng dùng cùng quy trình native.

| Mô hình | AUC trước refit | AUC sau refit | Ca CVV bằng 0 bị bỏ sót ở top 5% sau refit |
| --- | ---: | ---: | ---: |
| v1 | 0.95222 | | 536 |
| RealMLP | 0.94536 | 0.94497 | 561 |
| TabM-mini | 0.94593 | 0.94620 | 586 |

Blend tốt nhất theo AUC là v1 + 25% TabM đã refit. Trên cùng holdout, nó đạt 0.95195, thấp hơn v1 khoảng 0.00027, và bỏ sót 531 ca CVV bằng 0 ở top 5%. Đây là điểm của holdout cố định, đo riêng với OOF và public leaderboard. [Phiên đối chứng v1](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-locked-v1) · [Phiên refit neural](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-locked-neural).


```bash
python -m pip install pytabkit==1.8.0 rtdl-num-embeddings==0.0.12 tabm==0.0.3
python kaggle/train.py --locked-holdout --data data --output submissions/locked-v1
python kaggle/neural/train.py --locked-holdout --model both --data data --output submissions/locked-neural
python kaggle/neural/compare.py --locked-baseline submissions/locked-v1 --locked-neural submissions/locked-neural
```

## V4 thử token GPT-2 và tín hiệu từ dữ liệu gốc

Vòng thử dùng ba fold **chỉ trên 559.968 dòng development** trong split seed 8128. Mỗi fold chạy v1 đối chứng, v1 cộng tần suất token và v1 cộng cả tần suất lẫn target encoding của token. Token được tách từ cách viết số thập phân nhất quán. Tần suất chỉ học trên phần fit và target encoding của phần fit dùng cross-fitting nội bộ.

Hai ứng viên bổ trợ là Transformer nhỏ học từ token mỗi trường, và GLM chỉ học trên bộ gốc 20.000 dòng. Kaggle dẫn tới bộ gốc ở trang Data và cho phép dữ liệu ngoài công khai theo thể lệ. Mã đối chiếu ID, nhãn và fold của cả ba đầu ra trước khi chấm các rank blend ở tỷ trọng cố định. [Bộ dữ liệu gốc CC0](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) · [Data cuộc thi](https://www.kaggle.com/competitions/playground-series-s6e11/data) · [Thể lệ](https://www.kaggle.com/competitions/playground-series-s6e11/rules).

```bash
python kaggle/tokens/bundle.py /tmp/s6e11-token-kaggle --model tree
python kaggle/tokens/bundle.py /tmp/s6e11-transformer-kaggle --model transformer
python kaggle/tokens/bundle.py /tmp/s6e11-transfer-kaggle --model transfer
kaggle kernels push -p /tmp/s6e11-token-kaggle
kaggle kernels push -p /tmp/s6e11-transformer-kaggle --accelerator NvidiaTeslaT4
kaggle kernels push -p /tmp/s6e11-transfer-kaggle
```

Dùng `kaggle/tokens/compare.py --tree DIR --transformer DIR --transfer DIR` với các đầu ra Kaggle đã tải. Mã chọn ứng viên có AUC tổng cao hơn v1 và thắng ít nhất hai fold. Chỉ ứng viên đã chốt mới dùng holdout 20%.

Stacking logistic học từ logit của v1, Transformer và GLM dữ liệu gốc. Để đo stacking trên development, mỗi fold OOF của mô hình gốc được chia đôi. Stacker học trên một nửa và dự đoán nửa còn lại, cùng dùng mô hình gốc đã huấn luyện ngoài cả hai nửa.

| Ứng viên | Development OOF AUC | Chênh so với v1 | Số fold thắng v1 |
| --- | ---: | ---: | ---: |
| v1 | 0.948916 | | |
| Tần suất token | 0.948859 | −0.000056 | 1/3 |
| Tần suất và target encoding token | 0.948930 | +0.000014 | 2/3 |
| Transformer | 0.938601 | −0.010315 | 0/3 |
| GLM dữ liệu gốc | 0.924009 | −0.024907 | 0/3 |
| Stacking logistic | 0.947991 | −0.000925 | 0/3 |
| **v1 + 5% Transformer** | **0.948952** | **+0.000036** | **3/3** |

Công thức v1 + 5% Transformer được chốt trước khi chấm holdout. Để tái lập, tải dự đoán v1 trên split đã khóa và chạy `--stage holdout` cho Transformer. Sau đó truyền `--selected v1_plus_transformer_05` và báo cáo development đã lưu vào `kaggle/tokens/compare.py`. Bản CPU là lượt hoàn tất trước do GPU đang chờ.

Trên holdout 139.992 dòng, AUC tăng từ 0.952216 lên 0.952231. Số ca CVV bằng 0 bị bỏ sót ở top 5% giữ nguyên 536; tổng số ca bắt được giảm từ 1.697 xuống 1.693. Khoảng tin cậy xấp xỉ 95% của chênh lệch AUC theo dự đoán cố định là [−0.000070, +0.000101]. Mức tăng này được kiểm tra thêm bằng một bài nộp. [Phiên holdout trên CPU](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-token-transformer-holdout-cpu).

Tạo lại bài nộp bằng cách chạy `--stage full --variant v1` cho cây và `--stage full --model transformer` cho Transformer, rồi xuất blend từ hai CSV có ID trùng khớp:

```bash
python kaggle/tokens/export.py --development-report development.json --holdout-report holdout.json \
  --v1 v1/submission.csv --transformer transformer/submission.csv --output submissions/token-v4.csv
```

[Token LightGBM](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-tokens-development) · [Transformer](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-token-transformer-development) · [GLM dữ liệu gốc](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-original-glm-development).

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
