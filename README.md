# Image Harmonization on iHarmony4

Dự án nghiên cứu và thực nghiệm bài toán **Image Harmonization (Hòa hợp ảnh)** trên bộ dữ liệu chuẩn **iHarmony4** (256×256): điều chỉnh màu sắc, độ sáng và tông ánh sáng của vùng **foreground (vật thể ghép)** sao cho hài hòa tự nhiên với **background (ảnh nền)** dựa trên ảnh ghép (`composite`) và mặt nạ (`mask`).

> **Dữ liệu & Checkpoint trọng số (~68 GB):** Toàn bộ tập dữ liệu iHarmony4 (bản gốc & resize 256×256) và checkpoint `.pt`/`.pth`/`.safetensors` được lưu trên Google Drive để giữ repository nhẹ và sạch:
> **[Google Drive — ImageHarmonization](https://drive.google.com/drive/folders/1oLmarz5jaqyxyRg762QeynYq4dQy-6mm?usp=sharing)**

---

## 1. Tổng quan kiến trúc & Các mô hình thực nghiệm

Dự án xây dựng, cải tiến và đối chiếu **8 mô hình** trên cùng một quy trình chia dữ liệu chuẩn (`seed=42`, chia Train/Val theo ảnh nền gốc để tránh data leakage, SHA-256 split fingerprint: `aad71bde2781...`):

| # | Mô hình | Kiến trúc / Phương pháp | Tham số (Train / Total) | Trạng thái |
|---|---|---|---:|---|
| 1 | **Baseline U-Net (Full L1)** | U-Net + GroupNorm, loss L1 toàn ảnh (50 ep) | 7.76M | Đã train & test |
| 2 | **Baseline U-Net (Masked L1)** | U-Net + GroupNorm, Masked L1 trên vùng foreground (50 ep, lưu mốc 30 ep) | 7.76M | Đã train & test |
| 3 | **U-Net + 1 PatchGAN (Branch B)** | Fine-tune từ `best_30ep.pt` + 1 SN-PatchGAN Discriminator + D-Warmup (ep 31–50) | 7.76M | Đã train |
| 4 | **U-Net + 3 PatchGAN (Branch C)** | Fine-tune từ `best_30ep.pt` + 3 Multi-scale SN-PatchGAN Discriminators (ep 31–49) | 7.76M | Đã train |
| 5 | **Improved U-Net v2 (Hybrid)** | GatedConv Encoder + Bottleneck FG-BG Cross-Attention + Attention Gates + Color Affine Head | 15.54M | Đã train |
| 6 | **Improved U-Net v3 (DINOv2 + Lab)** | Frozen DINOv2 ViT-S/14 Semantic Guide + Semantic FG-BG Cross-Attention + Hybrid Affine Residual Head + Lab L1 Loss | 11.23M / 33.29M | Đã train & test |
| 7 | **RainNet (Pretrained)** | Ling et al., CVPR 2021 — Region-aware Adaptive Instance Normalization (`RAIN`) | 54.76M | Đã đánh giá |
| 8 | **HT+ (`HT_2H9L_allihd`, Pretrained)** | Zhenglab — Harmony Transformer (2 heads, 9 layers, Sine PosEmb) | 4.77M | Đã đánh giá |

*Hướng mở rộng Diffusion (tham khảo trong `docs/image_harmonization_models_links.md`):* `DiffHarmony` và `IC-Light` (đã lưu checkpoint pretrained trên Drive).

---

## 2. Kết quả hiện tại trên iHarmony4 (256×256)

Bộ dữ liệu gồm **73.146 ảnh** (`HCOCO`, `HAdobe5k`, `HFlickr`, `Hday2night`), chia thành **62.463 Train / 3.279 Val / 7.404 Test**, kèm phân tầng **6 nhóm độ khó** (`small`/`large` theo `mask_ratio` median ≈ 4.72% × `low`/`medium`/`high` theo CIE Lab `DeltaE76` q1 ≈ 11.87, q2 ≈ 18.15).

### A. Kết quả trên toàn tập Test (7.404 ảnh, ghép nền gốc)

| Mô hình | MSE ↓ | fMSE ↓ | PSNR ↑ (dB) | fPSNR ↑ (dB) | SSIM ↑ |
|---|---:|---:|---:|---:|---:|
| **Improved U-Net v3 (DINOv2 + Lab, Ep 30)** | **37.01** | **369.64** | **36.29** | **24.58** | **0.9854** |
| **HT+ Pretrained (`HT_2H9L_allihd`)** | 36.99 | 387.83 | 36.17 | 24.42 | 0.9593 |
| **Baseline U-Net (Masked L1, Ep 39)** | 41.67 | 401.16 | 35.95 | 24.16 | 0.9845 |
| **RainNet Pretrained (CVPR 2021)** | 46.76 | 531.44 | 35.61 | 22.74 | — |
| **Baseline U-Net (Full L1, Ep 39)** | 60.59 | 586.10 | 34.61 | 22.55 | 0.9815 |

### B. Kết quả Validation tốt nhất của tất cả nhánh tự huấn luyện (3.279 ảnh Val)

| Mô hình | Best Epoch | Val MSE ↓ | Val fMSE ↓ | Val PSNR ↑ | Val fPSNR ↑ |
|---|---:|---:|---:|---:|---:|
| **Improved U-Net v3 (DINOv2 + Lab)** | 30 | **36.89** | **361.33** | **36.38** | **24.82** |
| **Improved U-Net v2 (Hybrid)** | 22 | 39.71 | 382.23 | 36.24 | 24.67 |
| **Baseline U-Net (Masked L1)** | 39 | 39.64 | 384.83 | 36.11 | 24.48 |
| **U-Net + 3 PatchGAN (Branch C)** | 38 | 40.05 | 386.84 | 36.10 | 24.46 |
| **U-Net + 1 PatchGAN (Branch B)** | 38 | 40.20 | 388.14 | 36.08 | 24.44 |
| **Baseline U-Net (Full L1)** | 39 | 59.28 | 556.07 | 34.77 | 22.88 |

> Chạy `scripts/evaluate_all.py` hoặc `notebooks/06_evaluate_all_models.ipynb` để chấm đồng bộ **toàn bộ 8+1 mô hình** theo đủ 7 chỉ số (`MSE`, `fMSE`, `PSNR`, `fPSNR`, `SSIM`, `fSSIM`, `fDeltaE76`) cùng bảng theo 4 sub-dataset và 6 nhóm độ khó.

---

## 3. Cấu trúc mã nguồn dự án

```text
ImageHarmonization/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── docs/
│   └── image_harmonization_models_links.md   # Tổng hợp paper, repo & checkpoint
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
│   └── evaluate_all.py                       # CLI đánh giá đầy đủ tất cả 8+1 mô hình
├── notebooks/                                # Bản notebook Colab gọn sạch để chạy trên T4/L4
│   ├── 01_unet_full_l1.ipynb
│   ├── 02_unet_masked_l1.ipynb
│   ├── 03_unet_masked_l1_1patchgan.ipynb
│   ├── 04_inference_rainnet.ipynb
│   ├── 05_inference_ht_plus.ipynb
│   └── 06_evaluate_all_models.ipynb
└── results/                                  # Log metrics, config & thống kê nhóm độ khó (không chứa weights)
    ├── dataset_groups/
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

# 5. Đánh giá đồng bộ toàn bộ 8+1 mô hình trong runs/
python scripts/evaluate_all.py \
  --data-dir /path/to/iharmony4_256 \
  --runs-dir /path/to/ImageHarmonization/runs \
  --groups-csv /path/to/ImageHarmonization/iHarmony4_groups_train_test/test.csv
```

---

## 5. Tài liệu tham khảo chính
- **iHarmony4 & DoveNet:** Cong et al., *DoveNet: Deep Image Harmonization via Domain Verification*, CVPR 2020.
- **RainNet:** Ling et al., *Region-aware Adaptive Instance Normalization for Image Harmonization*, CVPR 2021.
- **Harmony Transformer (HT+):** Guo et al., *Image Harmonization with Transformer* (ICCV 2021) / *Transformer for Image Harmonization and Beyond* (TPAMI).
- **DINOv2:** Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
- **Diffusion Extensions:** Zhou et al. (*DiffHarmony*, ICMR 2024); Zhang et al. (*IC-Light*, ICLR 2025).
