# NB0 — Tự viết hàm loss của DPO (~10 phút, chạy được trên CPU)
**Mục đích:** hiểu công thức DPO trước khi dùng thư viện.

**Việc cần làm:** điền hàm `my_dpo_loss` (chỗ có `# TODO`). Hàm nhận log-xác suất của câu chosen và rejected dưới mô hình đang học (`pc`, `pr`) và dưới mô hình tham chiếu (`rc`, `rr`), rồi trả về loss trung bình. Gợi ý: dùng `torch.nn.functional.logsigmoid`; công thức có ngay trong notebook.

**Xong khi:** các `assert` chạy qua. Một kiểm tra hay: khi mô hình mới giống hệt mô hình tham chiếu, loss phải bằng `log 2 ≈ 0.693`.

---
## Trạng thái hoàn thành: [x] ĐÃ HOÀN THÀNH
- [x] Đã cài đặt `my_dpo_loss` trong [`notebooks/00_dpo_loss_from_scratch.py`](file:///f:/VinUniversity/Schedule_Study_VinUniversity/Period_Two/Day07/K4-L3-Track3-2A202602851-NguyenGiaKhanh-Day22-DPO-ORPO-Alignment/notebooks/00_dpo_loss_from_scratch.py).
- [x] Đã đồng bộ sang các Colab notebooks qua `python scripts/build_colab.py`:
  - [`colab/Lab22_DPO_T4.ipynb`](file:///f:/VinUniversity/Schedule_Study_VinUniversity/Period_Two/Day07/K4-L3-Track3-2A202602851-NguyenGiaKhanh-Day22-DPO-ORPO-Alignment/colab/Lab22_DPO_T4.ipynb)
  - [`colab/Lab22_DPO_BigGPU.ipynb`](file:///f:/VinUniversity/Schedule_Study_VinUniversity/Period_Two/Day07/K4-L3-Track3-2A202602851-NguyenGiaKhanh-Day22-DPO-ORPO-Alignment/colab/Lab22_DPO_BigGPU.ipynb)
- [x] Kết quả kiểm tra:
  - Khớp với hàm tham chiếu `ref_loss`: 0.6981.
  - Khi policy == reference: loss = $\ln 2 \approx 0.6931$.
  - 50/50 CPU unit tests trong `scripts/test_lab22.py` đã vượt qua (100%).