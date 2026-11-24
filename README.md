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
| v4, 95% v1 refit + 5% Transformer | 58767546 | | 0.95117 |

V1 vẫn đứng đầu trong bốn bài nộp. V3 hiệu chỉnh thấp hơn v1 0.00003 trên public leaderboard. [Danh sách bài nộp trên Kaggle](https://www.kaggle.com/competitions/playground-series-s6e11/submissions).

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

Trên holdout 139.992 dòng, AUC tăng từ 0.952216 lên 0.952231. Số ca CVV bằng 0 bị bỏ sót ở top 5% giữ nguyên 536; tổng số ca bắt được giảm từ 1.697 xuống 1.693. Khoảng tin cậy xấp xỉ 95% của chênh lệch AUC theo dự đoán cố định là [−0.000070, +0.000101]. Bài v4 `58767546` đạt public AUC 0.95117, thấp hơn bài v1 0.00005. Hai thành phần v4 đều huấn luyện lại trên toàn bộ 699.960 dòng, còn bài v1 cũ trung bình dự đoán từ năm fold. OOF ở bảng trên chỉ dùng phần development và không so trực tiếp với OOF toàn bộ train của v1. [Phiên holdout trên CPU](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-token-transformer-holdout-cpu). [Phiên full refit v1](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-tokens-full) · [Phiên full refit Transformer CPU](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-token-transformer-full-cpu).

Tạo lại bài nộp bằng cách chạy `--stage full --variant v1` cho cây và `--stage full --model transformer` cho Transformer, rồi xuất blend từ hai CSV có ID trùng khớp:

```bash
python kaggle/tokens/export.py --development-report development.json --holdout-report holdout.json \
  --v1 v1/submission.csv --transformer transformer/submission.csv --output submissions/token-v4.csv
```

[Token LightGBM](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-tokens-development) · [Transformer](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-token-transformer-development) · [GLM dữ liệu gốc](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-original-glm-development).

## Tái lập Board Model + RealMLP v39

Step Z của goodpjw2008 và các tác giả đóng góp đạt **0.95562 Public ROC AUC**. Bản tái lập dùng sáu bộ đầu vào công khai và trọng số đã công bố để dựng lại ensemble. Điểm 0.95562 đến từ bài nộp của tác giả, sau khi tinh chỉnh trên public leaderboard. Bản chạy này kiểm tra khả năng tái lập thứ hạng trên test. [Notebook gốc v39](https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95562?scriptVersionId=372884035).

```bash
curl --fail --location https://www.kaggle.com/kernels/scriptcontent/372884035/download -o /tmp/s6e11-board-v39.ipynb
python kaggle/board/bundle.py /tmp/s6e11-board-replay --notebook /tmp/s6e11-board-v39.ipynb
kaggle kernels push -p /tmp/s6e11-board-replay
kaggle kernels output huthnhlng/s6e11-board-model-v39-replay -p submissions/board-v39
```

Script kiểm tra checksum của notebook gốc khi đóng gói và dùng các kiểm tra checksum đầu vào của tác giả khi chạy. `replay.json` ghi checksum của CSV và thứ hạng trên 299.984 giao dịch. Phiên chạy báo lỗi nếu danh sách ID, miền xác suất hoặc thứ hạng sai. Lượt chạy đối chiếu tạo CSV trùng từng byte với đầu ra v39 của tác giả. [Phiên tái lập trên Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-board-model-v39-replay).

## Hai ứng viên public và validation

Bản [Fraud Signal Lab v43 của Malyshev Danil](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion?scriptVersionId=372954959) được dựng từ hai bộ dữ liệu đầu vào công khai, khớp checksum `5689d07cc601becb3b2530b87fa5c03241a79d54fb2aba10fe2773d6eaf0464c` của tác giả. Bài nộp 58770152 đạt **0.95567 public AUC**. Công thức được chọn qua các lần chấm public leaderboard.

Để tái lập, chạy cell mã đầu tiên của notebook v43 với dữ liệu cuộc thi và hai bộ [November 21 frontier](https://www.kaggle.com/datasets/malyshevdanil/s6e11-nov21-expanded-public-frontier), [November 23 round 2 frontier](https://www.kaggle.com/datasets/malyshevdanil/s6e11-nov23-round2-public-frontier). Notebook kiểm tra checksum các đầu vào và đầu ra.

Ứng viên [Wide Stack + 2nd LM của goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298) có OOF stacker **0.952565**, so với v1 **0.950571** trên cùng năm fold. Bài nộp 58770176 dùng `wide_lm_stack_test.npy` của tác giả và đạt **0.95285 public AUC**. Trên OOF, Wide Stack đạt 0.934225 AUC cho nhóm CVV bằng 0 so với 0.931288 của v1, và bắt được 8.636 ca fraud ở top 5% so với 8.527 ca của v1.

Các mô hình gốc của OOF công khai đã học trên phần train đầy đủ theo từng fold. Stacker được chấm ngoài fold của nó, nhưng các mô hình gốc tạo đặc trưng cho các fold khác có thể đã học từ fold đánh giá stacker. Do vậy kết quả này chưa phải holdout độc lập hoàn toàn.

Để tạo lại bài nộp Wide Stack, tải [artifacts công khai](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts) và mẫu cuộc thi, sau đó chạy:

```bash
python kaggle/dual/export_wide.py --artifacts data/s6e11-fraud-series-artifacts \
  --sample data/sample_submission.csv --output submissions/wide_lm.csv
python kaggle/dual/stack.py --input data --output submissions/core_stack
```

Thử nghiệm stack nhỏ dùng 20 mô hình đại diện của [OOF bank](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank) đạt 0.951560 OOF. Thêm dự đoán GPT-2 compact trong [bundle công khai](https://www.kaggle.com/datasets/amanatar/s6e11-rank1-q17-meta-bundle) nâng lên 0.951954 trên cả năm fold. Đây là một thí nghiệm mới, không phải phép tái huấn luyện toàn bộ Wide Stack của tác giả.

## EDA train, validation và test

[Báo cáo EDA](eda/report/report.html) gồm các biểu đồ phân phối, tỷ lệ fraud theo nhóm với khoảng tin cậy Wilson, tương quan Spearman, khác biệt phân phối và lỗi OOF của v1. Validation là fold 0 trong StratifiedKFold 5 fold seed 42 của v1, gồm 139.992 dòng. Train là bốn fold còn lại, gồm 559.968 dòng. Test gồm 299.984 dòng không có nhãn. Fold này đã được dùng ở các thí nghiệm trước.

Thống kê tổng hợp và biểu đồ được lưu ở `eda/report/`. Để tạo lại:

```bash
python -m pip install -r requirements.txt
python eda/analyze.py --train data/train.csv --test data/test.csv \
  --oof submissions/v1/oof.npz --output outputs/eda
```

Bỏ `--oof` khi không có dự đoán OOF của v1. Khi có OOF, mã kiểm tra ID, nhãn và fold trước khi vẽ biểu đồ lỗi. Các biểu đồ phân phối số dùng cùng cạnh bin và gộp đuôi ngoài phân vị 0,1%–99,9%; thống kê min, max và các phân vị được lưu trong CSV.

## Coarse-to-fine cho nhóm CVV bằng 0

Thử nghiệm [CVV Specialist Holdout](https://www.kaggle.com/code/huthnhlng/s6e11-cvv-specialist-holdout) dùng v1 làm baseline, thêm LightGBM chuyên gia cho nhóm CVV bằng 0. Xác suất của chuyên gia được hiệu chỉnh từ dự đoán OOF rồi ghép theo logit. Ba fold development chọn trọng số trong 0%, 10%, 25% và 50%; holdout seed 73191 chỉ được dùng sau khi chọn trọng số.

| Mô hình | Development OOF AUC | Holdout AUC | Fraud bắt được ở top 5% holdout | Fraud CVV bằng 0 bị bỏ sót |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.949221 | **0.952311** | **1.712** | **503** |
| Baseline + 10% chuyên gia | **0.949248** | 0.952298 | 1.710 | 506 |

Chênh lệch holdout là −0.0000135, khoảng tin cậy bootstrap phân tầng ghép cặp 95% xấp xỉ [−0.0000668, +0.0000442] với 300 lượt lấy mẫu. Development tăng trên 3/3 fold, nhưng mức tăng nhỏ. Các dự đoán OOF dùng huấn luyện bộ hiệu chỉnh có thể chia sẻ dòng train giữa các fold, nên kết quả development có thể lạc quan. Mô hình và holdout cùng tập train đã được khám phá trong các EDA trước; holdout không tham gia fit hoặc chọn trọng số trong thử nghiệm này.

Trong 628 fraud bị bỏ sót trên holdout có 506 ca CVV bằng 0. Nhóm bỏ sót có risk score trung bình 41,78 so với 47,36 ở toàn bộ fraud, velocity trung bình 23,81 so với 27,21, và chỉ 3,8% là giao dịch nước ngoài so với 13,4% ở toàn bộ fraud. Thử nghiệm tiếp theo có thể tập trung vào fraud thiếu các cờ cảnh báo, kiểm tra loss hoặc đặc trưng tương tác theo từng fold và duy trì hiệu chỉnh thứ hạng giữa các nhóm.

```bash
python kaggle/coarse/bundle.py /tmp/s6e11-cvv-specialist
kaggle kernels push -p /tmp/s6e11-cvv-specialist
kaggle kernels output huthnhlng/s6e11-cvv-specialist-holdout -p outputs/cvv-specialist
```

Mã và thống kê tổng hợp nằm trong `kaggle/coarse/`. Kết quả holdout chưa cho thấy cải thiện so với baseline.

Sau đó chạy [full refit trên Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-cvv-specialist-full) theo yêu cầu để đối chiếu public. Baseline và chuyên gia học trên toàn bộ 699.960 dòng train, bộ hiệu chỉnh dùng OOF development của lượt trước và trọng số chuyên gia giữ ở 10%.

| Bài nộp cùng lượt refit | ID | Public ROC AUC |
| --- | ---: | ---: |
| Baseline | 58772074 | 0.95103 |
| Baseline + 10% chuyên gia | 58772064 | **0.95104** |

Chênh lệch public là +0.00001; ứng viên vẫn thấp hơn ensemble public hiện tại 0.95567. Đây là full refit của pipeline LightGBM + Digit GLM và chuyên gia, không phải chỉnh sửa ensemble 0.95567. Kết quả holdout dùng mô hình chỉ học trên 80% train và public dùng mô hình refit trên toàn bộ train.

```bash
python kaggle/coarse/bundle_full.py /tmp/s6e11-cvv-specialist-full
kaggle kernels push -p /tmp/s6e11-cvv-specialist-full
```

## Rank voting với CatBoost và GPT-2

[Thử nghiệm Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-rank-voting-holdout) giữ riêng 20% train theo seed 24681357. Ba fold trên development huấn luyện lại LightGBM, Digit GLM và CatBoost. Baseline ghép logit 80/20 đạt **0.949209 OOF** và thắng các cấu hình rank voting 0.948359–0.949016 cùng logistic stacking 0.949180. Baseline được chọn trước khi chấm và đạt **0.953237 AUC holdout**, bắt 1.733 trong 2.338 fraud ở top 5%. Nhóm bị bỏ sót gồm 605 fraud, trong đó 510 có CVV bằng 0. Lượt full refit tạo dự đoán test của từng mô hình.

Thử nghiệm riêng dùng OOF công khai cho CatBoost và GPT-2 target-first và OOF LightGBM/Digit GLM cũ, cùng fold seed 42. Trên phần development của split mới, rank vote trọng số 40/10/25/25 đạt **0.950551 OOF**, so với 0.949797 của rank vote LightGBM/Digit GLM. Các mô hình OOF công khai có thể đã học trên các dòng holdout mới, do đó phép này chỉ thăm dò. GPT-2 không được huấn luyện lại trong job holdout. Kết quả baseline ba mô hình có holdout tách trước khi fit và chọn công thức, nhưng dữ liệu train đã được EDA và xem xét ở các vòng trước; điểm meta OOF còn có thể chịu ảnh hưởng chồng lấn train giữa các fold của mô hình gốc.

Bài 58784987 ghép **95% rank của frontier 0.95567 và 5% rank vote bốn mô hình**, đạt **0.95565 public AUC**. Hai thành phần LightGBM/Digit GLM dùng full refit của job mới; CatBoost/GPT-2 lấy từ bundle công khai của amanatar và tác giả upstream. Vote 40/10/25/25 được đo bằng các dự đoán OOF công khai; phần blend với frontier chưa được đánh giá trên holdout độc lập. Các điểm và trọng số chọn sau khi quan sát development cần được xem là thăm dò.

```bash
python kaggle/voting/bundle.py /tmp/s6e11-rank-voting
kaggle kernels push -p /tmp/s6e11-rank-voting
kaggle kernels output huthnhlng/s6e11-rank-voting-holdout -p outputs/rank-voting
python kaggle/voting/explore.py --baseline-oof submissions/v1/oof.npz --public-bundle data/rank1_q17_streams.npz --output outputs/gpt-exploratory.json
python kaggle/voting/export.py --frontier data/frontier_095567.csv --members outputs/rank-voting/test_members.npz --public-bundle data/rank1_q17_streams.npz --output submissions/voting_frontier.csv
```

[Bundle OOF công khai và ghi công](https://www.kaggle.com/datasets/amanatar/s6e11-rank1-q17-meta-bundle) · [Frontier v43](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion?scriptVersionId=372954959).

## Thử nghiệm residual và tương tác

[Phiên Kaggle residual và tương tác](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-residual-interactions) giữ riêng 20% train với seed 718231. Trên development 3 fold, mô hình dùng các đặc trưng tương tác CVV–merchant risk–velocity và mô hình residual từ OOF baseline. Residual có hai trọng số 25% và 50%. Phiên bản residual 25% được chọn vì có AUC development cao nhất trong các ứng viên mới.

| Phiên bản | Development OOF AUC |
| --- | ---: |
| Baseline | **0.950054** |
| Tương tác | 0.949779 |
| Residual 25% | 0.949986 |
| Residual 50% | 0.949914 |

Trên holdout, baseline đạt 0.949606, residual 25% đạt 0.949653; số fraud bắt được ở top 5% là 1.697 và 1.700. Bootstrap phân tầng ghép cặp 300 lượt trên các dự đoán cố định cho khoảng chênh AUC 95% [0.0000234, 0.0000755]. Development và holdout cho kết quả trái chiều; meta-OOF có thể chia sẻ dòng train giữa các fold của mô hình gốc. Các vòng EDA trước cũng đã khám phá toàn bộ train.

Phiên Kaggle lưu file test của từng phiên bản. Các mô hình test học trên development 80%, đồng nhất với tập dùng tạo OOF huấn luyện residual. Bài nộp 58801958 ghép 95% thứ hạng của frontier 0.95567 với 5% residual được chọn, đạt **0.95568 public AUC**, tăng 0.00001 so với frontier. Phép ghép cuối chưa có holdout độc lập cho toàn bộ công thức; mức tăng public rất nhỏ và chưa xác nhận khả năng tổng quát hóa. Checksum trong `kaggle/repair/export_metadata.json`.

```bash
python kaggle/repair/bundle.py /tmp/s6e11-repair-kernel
kaggle kernels push -p /tmp/s6e11-repair-kernel
kaggle kernels output huthnhlng/s6e11-fraud-residual-interactions -p outputs/repair-kaggle
python kaggle/repair/export.py --frontier data/frontier_095567.csv --candidate outputs/repair-kaggle/submission.csv \
  --metrics outputs/repair-kaggle/metrics.json --output outputs/repair/frontier_residual.csv
```

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

## Sàng lọc residual cho cặp xếp hạng khó

Thử nghiệm `kaggle/pairwise_repair/evaluate.py` dùng dự đoán baseline OOF từ phiên residual trước. Mô hình residual được tăng gấp đôi trọng số cho fraud CVV retry = 0 và 10% giao dịch hợp lệ có điểm baseline cao nhất. Đây là heuristic nhấn mạnh hai phía của cặp xếp hạng khó, chưa phải hàm loss pairwise. Thử các trọng số hiệu chỉnh 0, 0.05, 0.10 và 0.20 bằng 3 fold meta với hai seed 113 và 937.

| Trọng số | AUC development trung bình hai seed |
| --- | ---: |
| 0 (baseline) | 0.95005411 |
| 0.05 | 0.95004909 |
| 0.10 | 0.95003804 |
| 0.20 | 0.94999412 |

Ở seed 113, trọng số 0.05 giảm fraud CVV = 0 bị bỏ sót tại top 5% từ 2.096 xuống 2.086, nhưng AUC toàn bộ giảm. Cả hai seed chọn trọng số 0; vì vậy holdout của bản được chọn trùng baseline: AUC 0.94960557, bắt 1.697 fraud ở top 5%. Không tạo bài nộp mới.

Hai seed chỉ thay đổi các meta fold trên cùng một bộ dự đoán baseline OOF; chúng không phải hai lần huấn luyện baseline độc lập. Meta fold không lồng hoàn toàn trong vòng tạo dự đoán baseline; holdout cũng từng được xem trước đây. Chưa có dự đoán holdout tương ứng cho toàn bộ blend frontier public 0.95568. Kết quả này chỉ sàng lọc heuristic trên baseline residual, không kết luận trực tiếp về blend frontier. Chi tiết: `kaggle/pairwise_repair/results.json`. Thử nghiệm chạy trên máy vì token Kaggle đã được xóa sau lần nộp trước.

Để chạy lại, truyền `--train`, `--development`, `--holdout` và `--output` cho script; hai file NPZ là `development_versions.npz` và `holdout.npz` trong output của [phiên Kaggle residual và tương tác](https://www.kaggle.com/code/huthnhlng/s6e11-fraud-residual-interactions/output).

## Tái lập clipped correction và Step AB

[Kết quả clipped correction của Areeb Ali](https://www.kaggle.com/code/areebbb/s6e11-which-signals-rank-fraud-lb-0-95568?scriptVersionId=373269898) đạt 0.95568 public. `kaggle/clipped_replay/replay.py` tái lập công thức từ hai kho rank công khai của Malyshev Danil, Step AA của goodpjw2008 và bốn vùng rủi ro thấp được ghi công trong notebook nguồn. Script kiểm tra checksum v43, quarter-step, Step AA và file clipped. Mã đồng thời xác minh và lưu bản [Step AB của goodpjw2008](https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95569), tác giả công bố 0.95569 public, dựa trên Sriram S orthogonal continuation; nguồn không phải mô hình do repo này tự train.

Script dựng lại bài của repo 0.95568 từ v43 và residual 25%, kiểm tra checksum rồi so sánh thứ hạng trên test không nhãn. Các điểm công khai và công thức được chọn qua public leaderboard; so sánh thứ hạng không đo được AUC hay khả năng tổng quát hóa. Chưa có holdout khớp độc lập cho toàn bộ các ensemble.

```bash
python kaggle/clipped_replay/bundle.py /tmp/s6e11-clipped-kernel
kaggle kernels push -p /tmp/s6e11-clipped-kernel
kaggle kernels output huthnhlng/s6e11-clipped-and-step-ab-replay -p outputs/clipped-replay
```

Phiên [Kaggle replay](https://www.kaggle.com/code/huthnhlng/s6e11-clipped-and-step-ab-replay) hoàn tất với các checksum khớp. Bài `clipped.csv` (ID 58814996) đạt **0.95568**; bài `step_ab.csv` (ID 58815003) đạt **0.95569**, hơn bài trước của repo 0.00001. Tại lần tải leaderboard ngày 24/11/2026 06:14 UTC, tài khoản hạng 49/2.908; hạng 10 đạt 0.95782. Đây là ảnh chụp tạm thời của public leaderboard.

Trên 299.984 dòng test không nhãn, clipped đổi 142 dòng và Step AB đổi 155 dòng trong top 5% so với bài cũ. Chênh lệch thứ hạng không thể hiện số dự đoán đúng hoặc sai. Các file này tái lập công khai: Step AB sử dụng trực tiếp đầu ra của tác giả, còn clipped khớp checksum của tác giả.

## Blend Dennis 12% và sàng lọc spline/digit

`kaggle/dennis_blend/` tái lập công thức 12% trên [notebook four refinements của Dennis](https://www.kaggle.com/code/dennis6886/s6e11-fraud-signal-rank-fusion-four-refinements), dùng Step AB và clipped correction đã lưu ở phiên trước. Tác giả sau đó cập nhật notebook thành công thức 9% khác; phiên này dùng công thức 12% cũ và checksum đầu ra tự cố định, chưa đối chiếu được đầu ra tác giả cho đúng phiên bản cũ. [Phiên chạy Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-dennis-12pct-replay) tái lập đúng checksum nội bộ. Submission 58826620 đạt **0.95569**, bằng Step AB; tại top 5% test có 6 dòng thay đổi. Không có nhãn test để biết các thay đổi đúng hay sai.

`kaggle/synthetic_likelihood/evaluate.py` sàng lọc logistic regression với spline, thêm chữ số thập phân, và thêm tương tác rủi ro–velocity/CVV. Dùng cùng 3 fold development của baseline cũ; blend rank với trọng số 0, 5%, 10%, 20%. Lựa chọn trên mean AUC từng fold là spline 10%: **0.950107** so với baseline **0.950061**. Thêm digit hoặc digit + tương tác trong thử nghiệm này không vượt spline thuần. Trên holdout seed 718231 đã xem trước đây, spline 10% đạt **0.949643**, baseline **0.949606**. Các fold không đồng thuận và khoảng bootstrap ghép cặp 95% trên dự đoán cố định của chênh lệch holdout là **[-0.000134, +0.000202]**. Trong 2.338.000 cặp fraud–legit lấy mẫu, spline sửa 2.975 cặp và làm sai 2.968 cặp. Các cặp có dùng lại dòng, đây không phải số cặp độc lập.

`digit_audit.json` mô tả phân phối chữ số theo fold và nhãn; chênh lệch biên không chứng minh nguyên nhân hay sự khác biệt có ý nghĩa sau khi xét nhiều nhóm. Mọi EDA trước đã xem dữ liệu train, holdout không còn nguyên trạng, lựa chọn bị ảnh hưởng bởi nhiều thử nghiệm và baseline OOF không phải Step AB 0.95569. Không nộp model spline vì chưa thấy bằng chứng ổn định cho blend vào Step AB. Chi tiết và source nằm trong `kaggle/synthetic_likelihood/`.

[Phiên kiểm tra spline/digit trên Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-synthetic-likelihood-audit) hoàn tất sau 215 giây; chọn cùng cấu hình spline 10%, mean development AUC 0.95010683 và holdout AUC 0.94964297, khớp kết quả local trong sai số số học.

## GPT-2 medium và ablation target encoding

[GPT-2 medium target-first của megayak](https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack) hiện công bố OOF chỉ cho fold 0. `kaggle/medium_survey/medium_eval.py` kiểm tra chính xác ID/fold thông qua bundle, đối chiếu với wide stack OOF, so từng trọng số rank cố định và lấy mẫu cặp fraud–legit. Trên 139.992 dòng fold 0, GPT-2 medium đạt 0.947990, GPT-2 small 0.948506 và wide stack 0.951774. Blend 10% medium vào wide đạt 0.951739, chênh −0.0000343; bootstrap ghép cặp 95% trên dự đoán cố định [−0.000140, +0.000067]. [Phiên audit Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-gpt2-medium-oof-audit) tái lập kết quả local. Bootstrap không bao gồm bất định do huấn luyện hay lựa chọn. Chưa có OOF của Step AB, và stack OOF có thể chia sẻ dữ liệu huấn luyện ở tầng meta.

`kaggle/medium_survey/encoding_ablation.py` so sánh 350 cây LightGBM trên fold 0–2 cùng `StratifiedKFold(5, seed=42)`. Từng fold tạo target encoding bằng 3 inner fold, smoothing 20. So sánh lần lượt raw, exact value, digit, GPT-2 token đầu/cuối/số lượng của bốn cột, rồi thêm log count tính trên phần train outer. Mean AUC ba fold lần lượt: **0.945703, 0.947291, 0.948622, 0.948625, 0.948694**. Count tốt hơn bản token ở 3/3 fold; token so với digit biến thiên giữa các fold. Đây là thí nghiệm sàng lọc tree 350 vòng, không tái huấn luyện toàn bộ notebook gốc và chưa đánh giá holdout độc lập hoặc các fold 3–4. Các cột exact/digit đã có trong những pipeline trước của repo.

Tokenizer lấy từ GPT-2 `tokenizer.json`, checksum `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`; bundle kiểm tra checksum và nhúng asset vào phiên Kaggle. Kết quả là thăm dò, không được xem như xác nhận blend với Step AB 0.95569.

So sánh thêm model `count` được chọn sau ablation với wide stack trên cùng ba fold bằng `encoding_ablation.py --variants count --wide ...`. Với blend rank 5%, chênh AUC từng fold lần lượt **−0.000058, +0.000065, −0.000030**; trọng số 10% cho **−0.000145, +0.000099, −0.000087**. Biến thể `count` đã được chọn từ chính các fold này nên kết quả chỉ để thăm dò; wide stack còn có thể chia sẻ dòng train ở tầng meta. Chưa có tín hiệu ổn định để nộp blend với Step AB.

[Phiên ablation trên Kaggle](https://www.kaggle.com/code/huthnhlng/s6e11-encoding-ablation-audit) hoàn tất sau 438,6 giây, với AUC từng fold khớp chạy local trong sai số số học (chênh tối đa 1.2e-16). Không nộp bài vì cả thử GPT-2 medium fold 0 và blend count với wide không cho thấy cải thiện ổn định; điểm 0.95569 giữ làm đối chứng.

Theo yêu cầu thử public, [phiên full-refit count](https://www.kaggle.com/code/huthnhlng/s6e11-count-step-ab-blend) huấn luyện count model trên toàn train với 350 cây và inner target encoding 3 fold, sau đó ghép 95% rank Step AB với 5% rank count. Local và Kaggle khớp checksum `16133e7ff219ab6442b745fa5646c5b8148f05e23294ba3ca0c8dc41d3d8e79c`. Submission **58830735 đạt 0.95567**, thấp hơn Step AB 0.95569 là 0.00002. Kết quả cho một trọng số và một lần full refit; không có OOF tương ứng của cả blend với Step AB để đánh giá private.
