# Image Harmonization on iHarmony4

Dự án nghiên cứu và thực nghiệm bài toán **Image Harmonization (Hòa hợp ảnh)** trên bộ dữ liệu chuẩn **iHarmony4** (256×256): điều chỉnh màu sắc, độ sáng và tông ánh sáng của vùng **foreground (vật thể ghép)** sao cho hài hòa tự nhiên với **background (ảnh nền)** dựa trên ảnh ghép (`composite`) và mặt nạ (`mask`).

> **Dữ liệu & Checkpoint trọng số (~68 GB):** Toàn bộ tập dữ liệu iHarmony4 (bản gốc & resize 256×256) và checkpoint `.pt`/`.pth`/`.safetensors` được lưu trên Google Drive để giữ repository nhẹ và sạch:
> **[Google Drive — ImageHarmonization](https://drive.google.com/drive/folders/1oLmarz5jaqyxyRg762QeynYq4dQy-6mm?usp=sharing)**

### Tài liệu dự án
- **[Kiến trúc Mô hình & Cấu hình Train/Pre-train](docs/architectures_and_configs.md)** — Sơ đồ hình vẽ kiến trúc, Mermaid flow, chi tiết hyper-parameters và phân tích toàn bộ 8+1 mô hình.
- **[Cấu trúc Thư mục Google Drive & Quy tắc Nhóm](docs/drive_structure_and_rules.md)** — Bản đồ thư mục Drive (~68 GB), quy chuẩn đặt tên `runs/`, quy tắc chống data leakage & đánh giá Blind Test, checklist thêm mô hình mới.
- **[Tổng hợp Paper & Checkpoint Pre-trained](docs/image_harmonization_models_links.md)** — Danh sách paper, repo và đường dẫn checkpoint gốc (gồm cả nhánh mở rộng `DiffHarmony`, `IC-Light`).

---

## 1. Tổng quan kiến trúc & Các mô hình thực nghiệm

Dự án xây dựng, cải tiến và đối chiếu **8 mô hình** (+ baseline `Input Composite`) trên cùng một quy trình chia dữ liệu chuẩn (`seed=42`, chia Train/Val theo ảnh nền gốc `sourceID` để đảm bảo 0% data leakage, SHA-256 split fingerprint: `aad71bde27817a76a2ab959e795b0798dff7fe526c5fe16ec6663e43e79de6be`):

| # | Mô hình | Kiến trúc / Phương pháp | Tham số (Train / Total) | Trạng thái |
|---|---|---|---:|---|
| 1 | **Baseline U-Net (Full L1)** | U-Net + GroupNorm, loss L1 toàn ảnh (50 ep, best ep 39) | 7.76M | Đã train & đánh giá |
| 2 | **Baseline U-Net (Masked L1)** | U-Net + GroupNorm, Masked L1 trên vùng foreground (50 ep, best ep 39) | 7.76M | Đã train & đánh giá |
| 3 | **U-Net + 1 PatchGAN (Branch B)** | Fine-tune từ `best_30ep.pt` + 1 SN-PatchGAN Discriminator + D-Warmup (ep 31–50, best ep 38) | 7.76M | Đã train & đánh giá |
| 4 | **U-Net + 3 PatchGAN (Branch C)** | Fine-tune từ `best_30ep.pt` + 3 Multi-scale SN-PatchGAN Discriminators (ep 31–49, best ep 38) | 7.76M | Đã train & đánh giá |
| 5 | **Improved U-Net v2 (Hybrid)** | GatedConv Encoder + Bottleneck FG-BG Cross-Attention + Attention Gates + Color Affine Head (best ep 22) | 15.54M | Đã train & đánh giá |
| 6 | **Improved U-Net v3 (DINOv2 + Lab)** | Frozen DINOv2 ViT-S/14 Semantic Guide + Semantic FG-BG Cross-Attention + Hybrid Affine Residual Head + Lab L1 Loss (30 ep) | 11.23M / 33.29M | Đã train & đánh giá |
| 7 | **RainNet (Pretrained)** | Ling et al., CVPR 2021 — Region-aware Adaptive Instance Normalization (`RAIN`), bản 256×256 | 54.76M | Đã đánh giá + Audit |
| 8 | **HT+ (`HT_2H9L_allihd`, Pretrained)** | Guo et al., ICCV 2021 / TPAMI 2022 — Harmony Transformer `FC-TRE-DeCNN` (2 heads, 9 layers, Sine PosEmb) | 4.77M | Đã đánh giá |

*Hướng mở rộng Diffusion (đã lưu sẵn trọng số trong `Drive/checkpoints/`):* `DiffHarmony` (ICMR 2024) và `IC-Light` (ICLR 2025).

---

## 2. Kết quả đánh giá chuẩn hóa trên iHarmony4 (256×256)

Bộ dữ liệu gồm **73.146 ảnh** (`HCOCO`, `HAdobe5k`, `HFlickr`, `Hday2night`), chia thành **62.463 Train / 3.279 Val / 7.404 Test**, kèm phân tầng **6 nhóm độ khó** (`small`/`large` theo `mask_ratio` median ≈ `4.72%` × `low`/`medium`/`high` theo CIE Lab `DeltaE76` $q_1 \approx 11.87$, $q_2 \approx 18.15$).

Toàn bộ kết quả trong Bảng A–C được chấm trên **cùng 1 pipeline Blind Test float32 thống nhất** (`out = clip(raw * M + comp * (1 - M), 0, 1)`, tuyệt đối không dùng Ground-Truth `real` ở đầu vào hay nền đầu ra). File tổng hợp đầy đủ lưu trong [`results/evaluation_all_models/`](results/evaluation_all_models/).

### A. Tổng hợp toàn bộ 7.404 ảnh Test (Blind Test thống nhất)

| # | Mô hình | Ep | Params | ms/im | MSE ↓ | fMSE ↓ | ΔfMSE | PSNR ↑ | fPSNR ↑ | SSIM ↑ | fSSIM ↑ | fΔE76 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **Input Composite (No Harm)** | — | 0.00M | 0.01 | 172.90 | 1359.67 | +0.0% | 31.43 | 19.05 | 0.9731 | 0.8244 | 16.13 |
| 1 | **Baseline U-Net (Full L1)** | 39 | 7.76M | 4.68 | 60.59 | 586.10 | +56.9% | 34.61 | 22.55 | 0.9815 | 0.8786 | 11.69 |
| 2 | **Baseline U-Net (Masked L1)** | 39 | 7.76M | 4.79 | 41.67 | 401.16 | +70.5% | 35.95 | 24.16 | 0.9845 | 0.9055 | 10.05 |
| 3 | **U-Net + 1 PatchGAN (Branch B)** | 38 | 7.76M | 4.81 | 41.98 | 401.50 | +70.5% | 35.92 | 24.12 | 0.9843 | 0.9041 | 10.20 |
| 4 | **U-Net + 3 PatchGAN (Branch C)** | 38 | 7.76M | 4.78 | 41.87 | 400.34 | +70.6% | 35.93 | 24.14 | 0.9843 | 0.9045 | 10.15 |
| 5 | **Improved U-Net v2 (Hybrid)** | 22 | 15.54M | 8.41 | 40.37 | 386.48 | +71.6% | 36.15 | 24.44 | 0.9850 | 0.9093 | 9.91 |
| 6 | **Improved U-Net v3 (DINOv2 + Lab)** | 30 | 11.2/33.3M | 10.11 | **37.01** | **369.64** | **+72.8%** | **36.29** | **24.58** | **0.9854** | **0.9113** | 9.67 |
| 7 | **RainNet Pretrained (CVPR 2021)** | Pre | 54.76M | 6.93 | 58.09 | 619.18 | +54.5% | 34.27 | 22.09 | 0.9806 | 0.8654 | 11.95 |
| 8 | **HT+ Pretrained (`HT_2H9L`)** | Pre | 4.77M | 69.87 | 37.56 | 394.66 | +71.0% | 36.09 | 24.31 | 0.9847 | 0.9044 | **9.57** |

### B. Chi tiết `fMSE ↓ / PSNR ↑` theo 4 Sub-dataset

| # | Mô hình | HCOCO (4.283) | HAdobe5k (2.160) | HFlickr (828) | Hday2night (133) | ALL (7.404) |
|---|---|---:|---:|---:|---:|---:|
| 0 | Input Composite (No Harm) | 979.0 / 33.55 | 2036.1 / 28.28 | 1556.8 / 28.25 | 1406.0 / 33.82 | 1359.7 / 31.43 |
| 1 | Baseline U-Net (Full L1) | 616.5 / 35.25 | 367.6 / 35.02 | 953.7 / 30.03 | 868.0 / 35.66 | 586.1 / 34.61 |
| 2 | Baseline U-Net (Masked L1) | 389.3 / 36.83 | 280.9 / 35.97 | 703.4 / 31.28 | 854.3 / 36.23 | 401.2 / 35.95 |
| 3 | U-Net + 1 PatchGAN (Branch B) | 389.3 / 36.80 | 282.6 / 35.95 | 705.7 / 31.22 | 830.9 / 36.26 | 401.5 / 35.92 |
| 4 | U-Net + 3 PatchGAN (Branch C) | 387.8 / 36.82 | 282.2 / 35.95 | 703.9 / 31.24 | 834.8 / 36.24 | 400.3 / 35.93 |
| 5 | Improved U-Net v2 (Hybrid) | 380.3 / 36.94 | **249.9** / 36.45 | 707.9 / 31.27 | 803.1 / 36.37 | 386.5 / 36.15 |
| 6 | **Improved U-Net v3 (DINOv2 + Lab)** | **360.0** / 37.11 | 252.8 / **36.50** | 667.4 / 31.46 | **725.3 / 36.55** | **369.6 / 36.29** |
| 7 | RainNet Pretrained (Blind) | 634.2 / 35.11 | 431.4 / 34.19 | 967.4 / 30.09 | 1016.6 / 34.65 | 619.2 / 34.27 |
| 8 | HT+ Pretrained (`HT_2H9L`) | 366.4 / **37.18** | 340.4 / 35.52 | **618.3 / 31.92** | 793.0 / 36.13 | 394.7 / 36.09 |

### C. Chi tiết `fMSE ↓` theo 6 Nhóm Độ Khó (`mask_size × color_shift_DeltaE76`)

| # | Mô hình | `small_low` | `small_medium` | `small_high` | `large_low` | `large_medium` | `large_high` |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Input Composite (No Harm) | 351.97 | 905.28 | 2459.99 | 371.43 | 973.10 | 3039.45 |
| 1 | Baseline U-Net (Full L1) | 308.17 | 561.36 | 1062.07 | 292.97 | 491.34 | 823.40 |
| 2 | Baseline U-Net (Masked L1) | 253.46 | 402.48 | 683.45 | 206.22 | 356.13 | 517.23 |
| 3 | U-Net + 1 PatchGAN (Branch B) | 262.10 | 406.85 | 663.30 | 213.05 | 361.44 | 512.51 |
| 4 | U-Net + 3 PatchGAN (Branch C) | 260.93 | 404.90 | 662.48 | 212.20 | 360.20 | 511.64 |
| 5 | Improved U-Net v2 (Hybrid) | 254.92 | 395.95 | 654.97 | 188.20 | 332.64 | 500.98 |
| 6 | **Improved U-Net v3 (DINOv2 + Lab)** | **245.76** | **380.73** | **630.95** | **184.09** | 314.20 | **471.91** |
| 7 | RainNet Pretrained (Blind) | 449.86 | 639.71 | 1094.48 | 296.12 | 471.45 | 779.66 |
| 8 | HT+ Pretrained (`HT_2H9L`) | 274.25 | 433.33 | 683.21 | 209.89 | **307.51** | 474.47 |

### D. Kiểm định riêng cho RainNet — Blind (Không gian lận) vs Protocol gốc `junleen/RainNet` (`GT-BG*`)

Repo gốc `junleen/RainNet` (`data/iharmony4_dataset.py`, dòng 72) dùng Ground-Truth `real` để thay nền vào đầu vào (`comp = comp*mask + real*(1-mask)`). Kết quả kiểm định tách bạch 4 chế độ trên cùng checkpoint `RainNet_net_G_last.pth` (7.404 ảnh Test):

| Chế độ đo RainNet (256×256) | Chế độ | MSE ↓ | fMSE ↓ | PSNR ↑ | fPSNR ↑ | SSIM ↑ | fSSIM ↑ | fΔE76 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **R1. Blind (KHÔNG gian lận — `comp` gốc)** | Blind | 58.09 | 619.18 | 34.27 | 22.09 | 0.9806 | 0.8654 | 11.95 |
| **R2. GT-BG In + Fair Comp BG** | GT-BG* | 48.74 | 531.44 | 34.83 | 22.74 | 0.9816 | 0.8730 | 11.07 |
| **R3. Official Repo (Float32, `MSE_bg=0`)** | GT-BG* | 46.76 | 531.44 | 35.62 | 22.74 | 0.9868* | 0.8742 | 11.07 |
| **R4. Official `evaluate.py` (`tensor2im` uint8)** | GT-BG* | 47.27 | 536.63 | 35.57 | 22.69 | 0.9865* | 0.8720 | 11.13 |

> **Lưu ý khoa học:** Ngay cả khi cho RainNet dùng protocol gốc có lợi thế `GT-BG*` (`R3: fMSE 531.44`), cả **Baseline U-Net Masked L1** (`401.16`) lẫn **Improved U-Net v3** (`369.64`) chạy 100% Blind đều vượt xa RainNet trên mọi chỉ số vùng tiền cảnh (`fMSE`, `fPSNR`, `fSSIM`, `fΔE76`).

### E. Kết quả Validation tốt nhất của tất cả nhánh tự huấn luyện (3.279 ảnh Val)

| Mô hình | Best Epoch | Val MSE ↓ | Val fMSE ↓ | Val PSNR ↑ | Val fPSNR ↑ |
|---|---:|---:|---:|---:|---:|
| **Improved U-Net v3 (DINOv2 + Lab)** | 30 | **36.89** | **361.33** | **36.38** | **24.82** |
| **Improved U-Net v2 (Hybrid)** | 22 | 39.71 | 382.23 | 36.24 | 24.67 |
| **Baseline U-Net (Masked L1)** | 39 | 39.64 | 384.83 | 36.11 | 24.48 |
| **U-Net + 3 PatchGAN (Branch C)** | 38 | 40.05 | 386.84 | 36.10 | 24.46 |
| **U-Net + 1 PatchGAN (Branch B)** | 38 | 40.20 | 388.14 | 36.08 | 24.44 |
| **Baseline U-Net (Full L1)** | 39 | 59.28 | 556.07 | 34.77 | 22.88 |

---

## 3. Cấu trúc mã nguồn dự án

```text
ImageHarmonization/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── docs/
│   ├── architectures_and_configs.md          # Sơ đồ kiến trúc, cấu hình train/pre-train & kết quả chi tiết
│   ├── drive_structure_and_rules.md          # Cấu trúc thư mục Google Drive & quy tắc làm việc nhóm
│   ├── image_harmonization_models_links.md   # Tổng hợp paper, repo & checkpoint
│   └── figures/                              # 7 hình vẽ kiến trúc độ phân giải cao (PNG)
├── src/
│   └── iharmony/                             # Thư viện Python tách chuẩn từ notebooks
│       ├── utils.py                          # Seed, atomic checkpoint, path helpers
│       ├── losses.py                         # Full L1, Masked L1, Lab L1, LSGAN losses
│       ├── metrics.py                        # MSE, fMSE, PSNR, fPSNR, SSIM, fSSIM, fDeltaE76
│       ├── visualization.py                  # Lưới so sánh ảnh & biểu đồ phân tích
│       ├── data/
│       │   ├── preprocess.py                 # Giải nén & resize iHarmony4 về 256×256
│       │   └── dataset.py                    # Indexer, SHA-256 fingerprint, Dataset, 6 nhóm độ khó
│       └── models/
│           ├── blocks.py                     # GatedConv, AttentionGate, Cross-Attn, ColorAffine heads
│           ├── unet.py                       # Baseline U-Net (GroupNorm)
│           ├── discriminators.py             # SN-PatchGAN (1-scale & 3-scale)
│           ├── improved_unet_v2.py           # Improved U-Net v2 (GatedConv + Cross-Attn + Affine)
│           ├── improved_unet_v3.py           # Improved U-Net v3 (DINOv2 + Semantic Cross-Attn + Hybrid Head)
│           ├── rainnet.py                    # RainNet (CVPR 2021) tự chứa
│           ├── ht_plus.py                    # Harmony Transformer (HT_2H9L) tự chứa
│           └── registry.py                   # Factory nạp model & suy luận thống nhất
├── scripts/
│   ├── preprocess_iharmony4.py               # CLI tiền xử lý dữ liệu 256×256
│   ├── train_unet.py                         # CLI huấn luyện U-Net (Full L1 / Masked L1)
│   ├── train_patchgan.py                     # CLI huấn luyện Branch B (1-PatchGAN) & Branch C (3-PatchGAN)
│   ├── train_improved_unet.py                # CLI huấn luyện Improved U-Net v2 / v3
│   ├── evaluate_all.py                       # CLI đánh giá đầy đủ tất cả 8+1 mô hình
│   └── evaluate_rainnet_audit.py             # CLI kiểm định riêng RainNet (Blind vs Official GT-BG*)
├── notebooks/                                # Bản notebook Colab gọn sạch để chạy trên T4/L4
│   ├── 01_unet_full_l1.ipynb
│   ├── 02_unet_masked_l1.ipynb
│   ├── 03_unet_masked_l1_1patchgan.ipynb
│   ├── 04_inference_rainnet.ipynb
│   ├── 05_inference_ht_plus.ipynb
│   └── 06_evaluate_all_models.ipynb          # Gồm cả Phần 7 kiểm định riêng RainNet
└── results/                                  # Log metrics, config & bảng tổng hợp (không chứa weights)
    ├── dataset_groups/
    ├── evaluation_all_models/                # JSON & CSV tổng hợp 8+1 mô hình + rainnet_audit/
    └── runs/
```

---

## 4. Cài đặt & Hướng dẫn chạy nhanh

```bash
# 1. Cài đặt thư viện & package ở chế độ editable
pip install -r requirements.txt
pip install -e .

# 2. (Tùy chọn) Tiền xử lý dữ liệu iHarmony4 gốc về 256×256
python scripts/preprocess_iharmony4.py \
  --raw-dir /path/to/iharmony4_raw \
  --out-dir /path/to/iharmony4_256

# 3. Huấn luyện U-Net Masked L1 (50 epochs, tự lưu mốc best_30ep.pt)
python scripts/train_unet.py \
  --data-dir /path/to/iharmony4_256 \
  --run-dir runs/unet_masked_l1_50ep_all4_bs32_seed42 \
  --loss masked_l1 --epochs 50 --batch-size 32

# 4. Fine-tune PatchGAN (Branch B: --num-d 1 | Branch C: --num-d 3)
python scripts/train_patchgan.py \
  --data-dir /path/to/iharmony4_256 \
  --base-ckpt runs/unet_masked_l1_50ep_all4_bs32_seed42/best_30ep.pt \
  --run-dir runs/unet_branchB_1patchgan_50ep_all4_bs32_seed42 \
  --num-d 1

# 5. Đánh giá đồng bộ toàn bộ 8+1 mô hình trong runs/ (100% Blind Test)
python scripts/evaluate_all.py \
  --data-dir /path/to/iharmony4_256 \
  --runs-dir /path/to/ImageHarmonization/runs \
  --groups-csv /path/to/ImageHarmonization/iHarmony4_groups_train_test/test.csv

# 6. Kiểm định riêng RainNet (R1–R4: Blind vs Official GT-BG*)
python scripts/evaluate_rainnet_audit.py \
  --data-root /path/to/iharmony4_256 \
  --checkpoint /path/to/ImageHarmonization/runs/RainNet_Pretrained/RainNet_net_G_last.pth \
  --out-dir /path/to/ImageHarmonization/runs/evaluation_all_models/rainnet_audit \
  --groups-csv /path/to/ImageHarmonization/iHarmony4_groups_train_test/test.csv
```

---

## 5. Tài liệu tham khảo chính
- **iHarmony4 & DoveNet:** Cong et al., *DoveNet: Deep Image Harmonization via Domain Verification*, CVPR 2020.
- **RainNet:** Ling et al., *Region-aware Adaptive Instance Normalization for Image Harmonization*, CVPR 2021.
- **Harmony Transformer (HT+):** Guo et al., *Image Harmonization with Transformer* (ICCV 2021) / *Transformer for Image Harmonization and Beyond* (TPAMI 2022).
- **DINOv2:** Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
- **Diffusion Extensions:** Zhou et al. (*DiffHarmony*, ICMR 2024); Zhang et al. (*IC-Light*, ICLR 2025).
