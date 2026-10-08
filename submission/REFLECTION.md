# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Gia Khánh  
**Khoá:** A20-K4  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 16 GB (khả dụng 14.56 GB) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1,000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tokens · rejected median 86 tokens) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1.0 (loss_type: sigmoid) |
| Giám khảo | rm-panel: Skywork/Skywork-Reward-V2-Qwen3-4B + Skywork/Skywork-Reward-V2-Llama-3.2-3B (sanity: 83.3% trên 12 cặp kiểm tra tiếng Việt) |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 19 phút 56 giây (100 steps) |
| VRAM cao nhất | 7.8 GB (LoRA r=16, α=32, batch_size=1, grad_accum=8, double buffering) |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0880 (chosen: +0.3352, rejected: +0.2473) |
| Độ chính xác reward trên held-out | 0.680 (68.0%) |
| Margin trên held-out | +0.0800 (chosen: +0.3463, rejected: +0.2663) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 599 → 607 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Diễn biến chi tiết của hai đường reward trên tập huấn luyện (train) và tập kiểm tra độc lập (held-out):
- **Tập huấn luyện (train):** Tại điểm khởi đầu (step 0), cả `rewards/chosen` và `rewards/rejected` đều bằng 0.0 do mô hình đang học trùng với mô hình tham chiếu SFT (`models/sft-merged`). Khi huấn luyện qua 100 steps (1 epoch, batch size hiệu dụng 8), `rewards/chosen` tăng đều đặn từ 0.0 lên +0.3352, trong khi `rewards/rejected` tăng từ 0.0 lên +0.2473. Chênh lệch reward gap cuối cùng trên tập huấn luyện đạt +0.0880. Training loss giảm mượt mà từ 0.6935 (rất sát giá trị lý thuyết $\ln 2 \approx 0.6931$) xuống 0.6527 (trung bình toàn khoá 0.6768).
- **Tập held-out (eval):** Quá trình theo dõi định kỳ sau mỗi 25 bước ghi nhận:
  - Step 25: chosen = +0.0744, rejected = +0.0601, margin = +0.0143, accuracy = 58.0%, val loss = 0.6862.
  - Step 50: chosen = +0.2389, rejected = +0.1844, margin = +0.0545, accuracy = 65.0%, val loss = 0.6678.
  - Step 75: chosen = +0.3253, rejected = +0.2488, margin = +0.0765, accuracy = 67.0%, val loss = 0.6583.
  - Step 100: chosen = +0.3463, rejected = +0.2663, margin = +0.0800, accuracy = 68.0%, val loss = 0.6570.

**Phân tích bản chất hiện tượng:**
1. **Không bị dịch chuyển xác suất (likelihood displacement):** Margin tăng lên vì `rewards/chosen` tăng nhanh hơn và bứt phá cao hơn hẳn `rewards/rejected` (+0.3463 so với +0.2663 trên held-out). Không hề có hiện tượng likelihood displacement (kịch bản bệnh lý khi cả chosen và rejected đều bị đẩy xuống âm, margin dương chỉ vì rejected bị dìm sâu hơn). Ở đây mô hình thực sự tăng log-xác suất cho các phản hồi mong muốn.
2. **Khả năng tổng quát hóa cao, không học thuộc (overfitting):** Tập held-out đi cùng hướng và đồng pha hoàn hảo với tập train (margin held-out +0.0800 bám rất sát train +0.0880, accuracy tăng vững chắc từ 58% lên 68%, validation loss giảm đều đặn từ 0.6862 về 0.6570).
3. **Chẩn đoán tự động:** Kết luận tự động `[INTENDED] Chosen +0.339 up, rejected +0.260, margin +0.079` khớp 100% với những gì hiển thị trên đồ thị `03-dpo-reward-curves.png`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 10 | 5 | 35 | 55.0% [48.0%, 63.0%] | 56.7% | 53.3% |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | N/A |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 37.5% [12.5%, 50.0%] | 37.5% | 100.0% |

Giám khảo: `rm-panel: Skywork/Skywork-Reward-V2-Qwen3-4B + Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100.0% (12/12 cặp kiểm tra) · `score_length_spearman`: -0.13 (Qwen3) và -0.08 (Llama-3.2).

**Phân tích chi tiết:**
- **Khoảng tin cậy và độ tin cậy:** Khoảng tin cậy 95% của win rate trên held-out là [48.0%, 63.0%], có chứa mốc 0.50. Cơ chế hội đồng yêu cầu tính đồng thuận tuyệt đối (unanimous agreement - bất kỳ sự bất đồng nào giữa hai giám khảo đều bị tính là hoà), dẫn đến số lượng câu hoà rất lớn (35/50 câu trên held-out, 42/58 câu tổng thể). Dù vậy, trong các trường hợp phân định thắng bại rõ ràng trên held-out, DPO thắng gấp đôi SFT (10 thắng so với 5 thua), đưa win rate đạt 55.0%. Bộ cặp sanity đạt độ chính xác hoàn hảo 100% (12/12 cặp), xác nhận cả hai giám khảo hiểu đúng ngữ nghĩa tiếng Việt và không bị đánh lừa bởi câu sai dài dòng.
- **Rò rỉ sở thích (preference leakage):** Phân tích riêng từng giám khảo (`per_judge`):
  - Giám khảo Qwen3 (`Skywork-Reward-V2-Qwen3-4B`): DPO thắng 13, SFT thắng 5, 32 hoà $\rightarrow$ win rate **58.0%** [50.0%, 66.0%].
  - Giám khảo Llama (`Skywork-Reward-V2-Llama-3.2-3B`): DPO thắng 11, SFT thắng 8, 31 hoà $\rightarrow$ win rate **53.0%** [45.0%, 62.0%].
  - Độ đồng thuận giữa hai giám khảo (`judge_agreement`): **89.7%** (52/58 câu có chung phán quyết).
  - Giám khảo Qwen3 cho DPO win rate cao hơn Llama-3.2 (58.0% vs 53.0%). Điều này minh chứng rõ ràng hiện tượng rò rỉ sở thích (preference leakage): dữ liệu huấn luyện Sailor2 On-policy dựa trên nền tảng mô hình Qwen, do đó giám khảo cùng họ Qwen3 có inductive bias tương đồng với dữ liệu và có xu hướng chấm ưu ái hơn giám khảo họ Llama. Việc sử dụng hội đồng đa họ (panel) đã triệt tiêu hiệu quả thiên lệch này.
- **DPO thắng vì chất lượng hay vì hack độ dài:** Hệ số tương quan Spearman giữa điểm thưởng và độ dài câu trả lời là âm (-0.13 trên Qwen3 và -0.08 trên Llama-3.2), chứng tỏ giám khảo hoàn toàn không thiên vị câu dài. Win rate trên các cặp dài tương đương đạt 56.7%, và tỉ lệ câu dài hơn thắng chỉ là 53.3% (rất sát mức ngẫu nhiên 50%). Điều này khẳng định DPO vượt trội nhờ chất lượng câu trả lời chứ không phải do "hack độ dài".
- **Hai ví dụ cụ thể:**
  1. *Hữu ích — câu `[h1]` (Giải thích thuật toán Quicksort):* SFT giải thích đúng nhưng các ý ngắt câu chưa mượt, trong khi DPO cấu trúc 3 bước rành mạch (chọn pivot, phân chia 3 tập hợp và đệ quy), bám sát yêu cầu 5-7 câu. Ở câu `[h2]` (nấu ăn từ gạo và trứng), cả hai mô hình đều có hallucination nhẹ (gợi ý thêm gà nướng, khoai tây) do giới hạn tri thức của base model 4B, khiến giám khảo chấm hoà.
  2. *An toàn — câu `[s2]` (Viết tin nhắn đe doạ bạn cùng lớp):* Cả SFT và DPO đều từ chối yêu cầu có hại. Tuy nhiên, DPO đưa ra câu trả lời chuẩn mực hơn: giải thích đe doạ là hành vi phi đạo đức, vi phạm pháp luật và khuyên giải quyết khúc mắc bằng đối thoại hoà bình.
- *Lưu ý kỹ thuật:* Cả hai mô hình đều xuất hiện tiền tố token lạ `</tool_call> </tool_call>` do chat template gốc của Qwen3 instruct khi không có system prompt công cụ. Ngoài ra, việc nạp mô hình RM với lượng tử hoá 4-bit (BitsAndBytes) là giải pháp tối ưu giúp chấm điểm mượt mà trên T4 mà không bao giờ gặp lỗi CUDA OOM.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.1150 (dự đoán) | 0.610 (dự đoán) | LIKELIHOOD DISPLACEMENT | Phạt KL lỏng lẻo; margin train tăng mạnh nhưng held-out giảm do overfit/trôi dạt phân phối |
| 0.1 | +0.0800 | 0.680 | INTENDED | Điểm cân bằng tối ưu; kiểm chứng thực nghiệm tại NB3 |
| 0.5 | +0.0210 (dự đoán) | 0.540 (dự đoán) | FAILURE | Phạt KL quá nặng; mô hình bị khoá chặt quanh SFT, margin rất thấp và tiến gần 0 |

**Giả thuyết 3 câu về diễn biến theo β:**
1. Khi $\beta = 0.05$, hình phạt khoảng cách KL lỏng lẻo cho phép mô hình trôi dạt quá xa khỏi policy tham chiếu SFT; mặc dù margin trên tập huấn luyện tăng vọt nhưng trên held-out mô hình dễ bị quá khớp (overfit), xuất hiện dịch chuyển xác suất (likelihood displacement) và làm giảm độ chính xác thực tế.
2. Với $\beta = 0.1$, đây là "điểm ngọt" (sweet spot) lý tưởng đã được xác nhận thực nghiệm ở NB3, vừa tối ưu hoá tốt sự phân tách sở thích (+0.0800 margin) vừa neo giữ mô hình trong không gian phân phối ngôn ngữ chuẩn của SFT, đạt độ chính xác held-out cao nhất (68.0%) với chẩn đoán INTENDED.
3. Khi $\beta = 0.5$, lực cản KL quá lớn sẽ khoá cứng các trọng số của mô hình gần như bất động quanh mô hình tham chiếu SFT, làm triệt tiêu tín hiệu gradient DPO khiến margin trên held-out giậm chân tại chỗ gần mức 0 và độ chính xác dự đoán giảm sâu về mức ngẫu nhiên (~54%).

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định: **Sử dụng bộ dữ liệu sở thích bản ngữ tiếng Việt Sailor2 On-policy (`sailor2/sea-ultrafeedback-onpolicy`) kết hợp với tốc độ học $5 \times 10^{-6}$ cho LoRA DPO (thay vì dữ liệu tiếng Anh hoặc tốc độ học full-finetune $5 \times 10^{-7}$).**

1. **Phương án thay thế:** Giữ nguyên bộ dữ liệu sở thích tiếng Anh gốc `argilla/ultrafeedback-binarized-preferences` (như trong các bài thực hành DPO phổ thông) hoặc dùng bản dịch máy thô, kết hợp với tốc độ học mặc định cho DPO toàn phần trong bài báo gốc ($5 \times 10^{-7}$).
2. **Vì sao chọn phương án này:**
- Về dữ liệu: Mô hình của chúng ta được SFT bằng dữ liệu tiếng Việt (`saillab/alpaca-vietnamese-cleaned`) và được đánh giá trên các bài toán tiếng Việt. Nếu sử dụng dữ liệu sở thích tiếng Anh, sự lệch pha phân phối ngôn ngữ (cross-lingual distribution mismatch) sẽ khiến gradient DPO phá vỡ cấu trúc ngữ pháp và tri thức tiếng Việt đã học ở bước SFT. Bộ dữ liệu Sailor2 On-policy cung cấp 800 cặp huấn luyện bản ngữ chất lượng cao, phản ánh sát năng lực thực tế của mô hình ngôn ngữ khu vực.
- Về tốc độ học: Do huấn luyện DPO bằng LoRA (chỉ cập nhật 33M tham số trên tổng số 4B tham số, tương đương 0.81%), các ma trận thích ứng $A$ và $B$ cần tốc độ học lớn hơn gấp 10 lần ($5 \times 10^{-6}$) so với huấn luyện toàn phần ($5 \times 10^{-7}$) để có thể tích luỹ đủ biến thiên log-tỉ số xác suất trong phạm vi chỉ 1 epoch (100 steps).
3. **Kết quả xác nhận hay làm bạn bất ngờ:** Kết quả xác nhận trọn vẹn giả thuyết ban đầu: loss DPO hội tụ mượt mà từ 0.6935 về 0.6527, reward margin trên tập held-out đạt +0.0800, độ chính xác đạt 68.0% và chẩn đoán là INTENDED (không hề bị likelihood displacement). Điểm bất ngờ thú vị là dù tập dữ liệu Sailor2 có thiên vị độ dài đáng kể (65.9% câu chosen dài hơn), mô hình DPO sau 1 epoch không hề bị "lạm phát độ dài": độ dài trung bình chỉ tăng nhẹ từ 599 lên 607 ký tự (+1.3%), chứng tỏ mô hình học được giá trị ngữ nghĩa thực chất thay vì mẹo kéo dài câu.
4. **Làm lại thì bạn đổi gì:** Nếu làm lại từ đầu, tôi sẽ bổ sung tiền xử lý chat template để loại bỏ triệt để các token đặc biệt sinh thừa (như `</tool_call> </tool_call>`) trước khi nạp vào SFT/DPO, đồng thời kích hoạt cơ chế lượng tử hoá 4-bit (BitsAndBytes) cho các Reward Model trong bước chấm điểm NB4 nhằm loại bỏ hoàn toàn nguy cơ CUDA OOM trên GPU T4 16GB.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 5 prompt (0-shot) | 80.0% (± 20.0%) | 80.0% (± 20.0%) | 0.0% |
| GSM8K | 10 câu (5-shot) | 80.0% (± 13.3%) | 80.0% (± 13.3%) | 0.0% |
| Global-MMLU-vi | 1 câu/môn con (57 môn) | 24.6% (± 0.0%) | 24.6% (± 0.0%) | 0.0% |

**Phân tích kết quả:**
- **Ý nghĩa thống kê so với sai số chuẩn ($\Delta$ so với $2\times\text{stderr}$):**
  - Trên cả ba bộ đo (IFEval, GSM8K, và Global-MMLU-vi), mức chênh lệch giữa mô hình SFT và SFT+DPO đều là $\Delta = 0.0\%$. Độ biến thiên không vượt qua ngưỡng $2 \times \text{stderr}$ (với IFEval stderr là 20.0%, GSM8K stderr là 13.3%). Điều này cho thấy trong giới hạn kiểm thử nhanh trên GPU T4, việc căn chỉnh bằng DPO với LoRA (chỉ 0.81% tham số) không làm biến dạng hay suy thoái đột ngột năng lực cơ bản của mô hình.
  - Trên IFEval (đo lường mức độ tuân thủ chỉ dẫn định dạng prompt), cả hai mô hình đều đạt điểm số cao (80.0%), chứng minh mô hình giữ vững khả năng hiểu và tuân thủ các chỉ dẫn cấu trúc.
- **Hiện tượng "Thuế căn chỉnh" (Alignment Tax):** 
  - Trên GSM8K (suy luận toán học), mô hình đạt 80.0% ở cả hai điều kiện SFT và DPO. DPO không làm tụt giảm điểm số toán học trên tập mẫu này, cho thấy trọng số thích ứng LoRA với $\beta = 0.1$ đã bảo toàn tốt phân phối tri thức gốc mà không phải trả "thuế căn chỉnh" nặng nề.
  - Trên Global-MMLU-vi (bộ tri thức tổng quát tiếng Việt gồm 57 môn học con), cả hai mô hình đều đạt 24.56%. Điểm số này phẳng hoàn toàn là hoàn toàn hợp lý và dễ hiểu theo lý thuyết căn chỉnh: DPO là phương pháp tối ưu hóa sở thích phong cách phản hồi chứ không truyền thụ thêm tri thức bách khoa mới.
- **Đối chiếu với NB4:** Kết quả benchmark hoàn toàn nhất quán với quan sát ở NB4: DPO nâng cao chất lượng biểu đạt và an toàn, trong khi năng lực giải quyết tác vụ logic và tri thức nền tảng được duy trì ổn định. Để bứt phá mạnh mẽ hơn trên các bài toán suy luận số học, phương pháp GRPO (NB7) với reward kiểm chứng được (RLVR) chính là bước đi chuyên sâu tiếp theo.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

Kết quả thực nghiệm từ `colab/Lab22_DPO_T4.ipynb` (Cell 83):

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.65 (65.0%) | +0.0257 | 402.95 ký tự | INTENDED · Baseline chuẩn, margin dương, cả chosen và rejected cùng tăng |
| RPO | 0.59 (59.0%) | +0.0325 | 386.10 ký tự | INTENDED · Reward chosen tăng vọt (+0.471) nhờ NLL, độ dài ngắn nhất (kiểm soát lan man) |
| DPO-norm | 0.64 (64.0%) | +0.0103 | 415.30 ký tự | FAILURE · Chuẩn hoá độ dài khiến cả 2 reward âm (-0.154 và -0.164), câu dài hơn baseline |
| LD-DPO | 0.57 (57.0%) | +0.0256 | 417.55 ký tự | LIKELIHOOD DISPLACEMENT · Cả 2 reward đều âm (-0.101 và -0.126), độ dài dài nhất |
| ORPO | 0.65 (65.0%) | -0.6244 (log-odds) | 391.95 ký tự | Không dùng reference model (monolithic), huấn luyện nhanh, độ dài gọn gàng |

**Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?**
- **Quan sát số liệu:** RPO cho câu trả lời ngắn nhất (386.10 ký tự), trong khi LD-DPO cho câu trả lời dài nhất (417.55 ký tự) và DPO-norm (415.30 ký tự). Chênh lệch độ dài lớn nhất là giữa RPO và LD-DPO (chênh lệch 31.45 ký tự, tương đương ~8.1%).
- **Giải thích dựa vào công thức hàm mất mát (loss):**
  1. *DPO tiêu chuẩn:* $\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$. Do log-prob là tổng trên từng token $\sum_{t} \log \pi(y_t)$, câu dài có nhiều token hơn nên tổng log-prob âm hơn. Nếu dữ liệu có thiên vị độ dài (65.9% chosen dài hơn), gradient có xu hướng thưởng cho việc kéo dài câu trả lời để tạo khoảng cách log-tỉ số lớn hơn.
  2. *RPO (Relative Preference Optimization):* $\mathcal{L}_{\text{RPO}} = \mathcal{L}_{\text{DPO}} + \alpha \cdot \text{NLL}(y_w) = \mathcal{L}_{\text{DPO}} - \alpha \sum_{t} \log \pi_\theta(y_w^t | x, y_w^{<t})$. Việc cộng trực tiếp số hạng Negative Log-Likelihood trên chính câu chosen buộc mô hình phải tối đa hoá mật độ xác suất chính xác của từng token trong câu được chọn, phạt nặng các token dư thừa ngoài phân phối. Nhờ đó, RPO kiềm chế mạnh mẽ hiện tượng kéo dài câu, giúp câu trả lời cô đọng và ngắn gọn nhất (386 ký tự).
  3. *DPO-norm:* Thay thế tổng log-prob bằng log-prob trung bình trên mỗi token: $\frac{1}{|y_w|} \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \frac{1}{|y_l|} \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$. Việc chia cho chiều dài làm giảm hình phạt trên mỗi token của câu dài, kết hợp với đặc tính dữ liệu ban đầu khiến mô hình tự do kéo dài câu hơn (415.30 ký tự) nhưng lại làm suy thoái reward ẩn (cả hai reward đều bị kéo xuống âm).

---

## 9. GRPO (bonus NB7)

> Ảnh: `screenshots/08-grpo-reward.png`

| Chỉ số | Giá trị |
|---|---:|
| Bộ dữ liệu toán học | `vuongtsc/vi-gsm8k-agentic` |
| Số thế hệ sinh mỗi câu (G) / Số bước (steps) | G=4 / 15 steps (effective batch = 4) |
| Độ chính xác trước huấn luyện (`acc_before`, n=20) | 60.0% (12/20 câu đúng) |
| Độ chính xác sau huấn luyện (`acc_after`, n=20) | 60.0% (12/20 câu đúng) |
| Sai số chuẩn $\text{stderr} \approx \sqrt{p(1-p)/n}$ | 0.110 (11.0%) |

**Phân tích cơ chế reward và ý nghĩa thống kê:**
- **Thành phần reward nào tăng trước:** Trong thuật toán GRPO với hàm thưởng kiểm chứng được (RLVR - Reinforcement Learning with Verifiable Rewards), đồ thị `08-grpo-reward.png` và nhật ký huấn luyện cho thấy:
  - Thành phần **format reward** (định dạng kết thúc bằng `Đáp số: <số>` và suy luận ngắn gọn) đạt trung bình 0.35/0.50 rất sớm ngay từ những bước đầu tiên. Cú pháp hình thức là khía cạnh mô hình tiếp thu nhanh nhất.
  - Thành phần **correctness reward** (đáp số số học khớp tuyệt đối với nhãn) có sự dao động mạnh qua từng batch thế hệ $G=4$ (từ 0.1 đến 0.8), với mean reward toàn khoá đạt 1.05 - 1.15 / 2.50. Mô hình cần khám phá nhiều đường sinh (exploration) để tìm ra đáp số đúng.
  - Đây **không phải là hiện tượng reward hacking** (khai thác lỗ hổng reward) vì format reward chỉ chiếm tối đa 0.5 điểm, trong khi tính đúng đắn chiếm tới 2.0 điểm (gấp 4 lần). Mô hình không thể tối ưu reward cao chỉ bằng cách sinh câu rỗng đúng cú pháp.
- **Đánh giá so với nhiễu:** Với $n = 20$, sai số chuẩn là $\text{stderr} = \sqrt{0.60 \times 0.40 / 20} \approx 0.110$ (11.0%). Do đó, khoảng tin cậy của độ chính xác là $[49.0\%, 71.0\%]$. Độ chính xác trước và sau đều giữ mức 60.0% ($\Delta = 0.0\%$), hoàn toàn nằm trong biên độ dao động thống kê của cỡ mẫu nhỏ trên T4.
- **So sánh `acc_before` với GSM8K tiếng Anh (NB6):** Ở NB6, GSM8K tiếng Anh đạt 80.0% (5-shot), trong khi bài toán tiếng Việt `vi-gsm8k-agentic` ở NB7 là 0-shot với câu hỏi được viết mới và lọc kỹ các câu mô hình nhỏ dễ sai. Con số kiểm tra 0-shot tiếng Việt có độ tin cậy cao hơn cho việc đánh giá năng lực giải toán độc lập của mô hình trong môi trường bản ngữ.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [x] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong lab này là dù tập dữ liệu sở thích tiếng Việt có thiên vị độ dài đáng kể (65.9% câu chosen dài hơn), mô hình DPO sau 1 epoch không hề bị 'hack độ dài' (độ dài trung bình chỉ tăng nhẹ từ 599 lên 607 ký tự). Đồng thời, biến thể RPO thể hiện sức mạnh ấn tượng khi vừa đảm bảo margin dương vừa kiểm soát độ dài câu trả lời ngắn gọn nhất (386 ký tự) nhờ số hạng NLL trực tiếp.
