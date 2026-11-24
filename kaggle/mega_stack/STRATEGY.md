# Chiến lược v5: Mega stack kiểm chứng bằng CV

## Bối cảnh (24/11/2026)

- Bài tốt nhất của repo: Step AB 0.95569 public, là blend rank các nguồn công khai với trọng số chọn theo public leaderboard, không có OOF.
- Top leaderboard lên tới 0.97051, nhưng mọi notebook công khai dừng ở khoảng 0.9557.
- Đã kiểm tra rò rỉ trước khi chọn hướng:
  - thứ tự `id` (AUC 0.4985), tự tương quan nhãn theo `id`, dòng trùng train/test: không có;
  - bộ dữ liệu gốc `uditjain13/credit-card-fraud-detection-2026` (20.000 dòng): không có dòng trùng, số cột trùng với dòng gốc chỉ ở mức ngẫu nhiên, nhãn gốc không tách hoàn hảo (logistic AUC 0.9435).
- Không có rò rỉ, nên khoảng cách 0.01 ở top nhiều khả năng đến từ dò public leaderboard. Repo không theo hướng này vì nó không tổng quát sang private.

## Ý tưởng

Thay blend chỉnh theo public bằng stacker có thước đo CV:

1. **Thu thập mọi cặp OOF/test công khai** (`members.py`). Mỗi thành phần phải có OOF trên đủ 699.960 dòng train và dự đoán test trên đủ 299.984 dòng. Bỏ thành phần trùng (Spearman > 0.99995) và thành phần AUC < 0.90.
2. **Meta model** trên logit-rank của các thành phần, 5 fold riêng (seed 2027):
   - logistic regression (L2);
   - LightGBM nông trên thành phần cộng đặc trưng gốc, để học tương tác mà blend tuyến tính bỏ sót (ví dụ vùng rủi ro thấp).
3. **Vùng không gian lận**: các luật đã có 0 ca fraud trên 39.149 dòng train được đẩy xuống cuối thứ hạng. Hiệu quả được đo trên OOF.
4. **Bài nộp**: stacker thuần, và blend với Step AB nếu stacker cho thấy tín hiệu bổ sung.

## Giới hạn

- Các thành phần công khai dùng fold khác nhau, nên OOF của meta model có thể lạc quan nhẹ.
- Wide stack của goodpjw2008 và anchor 0.95370 là stack của chính các thành phần khác, nên chúng làm stacker phụ thuộc vào lựa chọn của tác giả.
- Khoảng cách tới top 10 là khoảng 0.002–0.003 AUC trên public. Một stacker trung thực khó đạt mức đó nếu top 10 đến từ dò leaderboard.
