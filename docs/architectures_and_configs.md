# Kiến trúc Mô hình & Cấu hình Huấn luyện / Pre-trained — Image Harmonization

Tài liệu này tổng hợp đầy đủ **hình vẽ kiến trúc**, **luồng dữ liệu / kích thước tensor**, và **cấu hình huấn luyện (train) hoặc pre-trained** của toàn bộ các mô hình đã sử dụng trong dự án **ImageHarmonization** trên bộ dữ liệu **iHarmony4 (256×256)**.

![Lộ trình Kiến trúc và Tiến hóa Thử nghiệm](figures/arch_00_evolution_overview.png)

---

## 0. Thiết lập Dữ liệu & Quy ước Thử nghiệm Chung

Toàn bộ các mô hình tự huấn luyện và pre-trained đều được đánh giá trên cùng một tập dữ liệu chuẩn hóa để đảm bảo so sánh công bằng:

| Thành phần | Giá trị chuẩn hóa |
|---|---|
| **Bộ dữ liệu** | **iHarmony4** (73.146 ảnh từ 4 tập con: `HCOCO`, `HAdobe5k`, `HFlickr`, `Hday2night`) |
| **Kích thước ảnh** | `256 × 256` (RGB dùng `BICUBIC`, mask nhị phân dùng `NEAREST` giữ nguyên `0/255`) |
| **Chia tập dữ liệu** | **62.463 Train / 3.279 Val / 7.404 Test** (`val_fraction = 0.05`, chia nhóm theo `sourceID` nền gốc để đảm bảo 0% rò rỉ dữ liệu) |
| **Mã xác minh split (SHA-256)** | `aad71bde27817a76a2ab959e795b0798dff7fe526c5fe16ec6663e43e79de6be` (khớp 100% ở tất cả 6 run tự train) |
| **Seed tái lập** | `seed = 42`, `cudnn.deterministic = True`, `cudnn.benchmark = False` |
| **Phân tầng độ khó Test** | 6 nhóm (`small`/`large` theo `mask_ratio` median = `4.72%` × `low`/`medium`/`high` theo CIE Lab D65 $\Delta E_{76}$ với $q_1 = 11.87$, $q_2 = 18.15$) |
| **Hậu xử lý ghép nền gốc** | $I_{\text{out}} = \text{clip}(\hat{I}_{\text{raw}} \odot M + I_{\text{comp}} \odot (1 - M),\ 0,\ 1)$ |
| **Môi trường train thực tế** | Google Colab — GPU **NVIDIA Tesla T4 (15 GB VRAM)**, PyTorch `2.14.1+cu130`, Python `3.13.16` |

---

## Phần I — Các Kiến trúc Tự Huấn Luyện (Trained on iHarmony4)

### 1. Baseline U-Net (`unet_groupnorm_v1`) — Full L1 & Masked L1 (Branch A)

- **Mã nguồn:** `src/iharmony/models/unet.py` (`Block`, `UNet`) | `scripts/train_unet.py`
- **Thư mục run:** `runs/unet_full_l1_all4_bs32_seed42/` và `runs/unet_masked_l1_50ep_all4_bs32_seed42/`
- **Quy mô:** **7.76M tham số** (64 tensors)

![Kiến trúc Baseline U-Net GroupNorm v1](figures/arch_01_unet_baseline.png)

#### Luồng kiến trúc (Mermaid)
```mermaid
flowchart LR
    IN["Input [Comp, Mask] (4x256x256)"] --> E0["enc.0 Block (32x256x256)"]
    E0 -->|"MaxPool2d(2)"| E1["enc.1 Block (64x128x128)"]
    E1 -->|"MaxPool2d(2)"| E2["enc.2 Block (128x64x64)"]
    E2 -->|"MaxPool2d(2)"| E3["enc.3 Block (256x32x32)"]
    E3 -->|"MaxPool2d(2)"| E4["Bottleneck enc.4 (512x16x16)"]
    E4 -->|"ConvTranspose2d(2)"| D0["dec.0 Block (256x32x32)"]
    E3 -.->|"Skip Concat"| D0
    D0 -->|"ConvTranspose2d(2)"| D1["dec.1 Block (128x64x64)"]
    E2 -.->|"Skip Concat"| D1
    D1 -->|"ConvTranspose2d(2)"| D2["dec.2 Block (64x128x128)"]
    E1 -.->|"Skip Concat"| D2
    D2 -->|"ConvTranspose2d(2)"| D3["dec.3 Block (32x256x256)"]
    E0 -.->|"Skip Concat"| D3
    D3 --> HEAD["Conv1x1(32->3) + Sigmoid"]
    HEAD --> BLEND["out = raw*M + C*(1-M)"]
```

#### Đặc điểm kỹ thuật
- Mỗi `Block` gồm 2 lớp `Conv2d(3×3, padding=1, bias=False) → GroupNorm(8) → ReLU`. Dùng `GroupNorm(8)` thay cho `BatchNorm` giúp ổn định khi batch size thay đổi và không phụ thuộc thống kê giữa các ảnh khác nhau.
- **Khác biệt duy nhất giữa Run 1 và Run 2 là hàm Loss:**
  - **Run 1 (Full L1):**
    $$\mathcal{L}_{\text{full}} = \frac{1}{3HW} \sum_{c,h,w} |\hat{I}_{\text{raw}} - I_{\text{real}}|$$
  - **Run 2 (Masked L1 — Branch A):** chuẩn hóa theo diện tích vùng foreground $M$ của từng ảnh để vùng vật thể nhỏ không bị lu mờ bởi nền lớn:
    $$\mathcal{L}_{\text{masked}} = \frac{1}{B} \sum_{b=1}^{B} \frac{\sum_{c,h,w} |\hat{I}_{\text{raw}}^{(b)} - I_{\text{real}}^{(b)}| \cdot M^{(b)}}{3 \cdot \max(\sum_{h,w} M^{(b)},\ \epsilon)}$$

#### Cấu hình huấn luyện
| Tham số | Run 1: U-Net Full L1 | Run 2: U-Net Masked L1 (Branch A) |
|---|---|---|
| **Kiến trúc** | `unet_groupnorm_v1` (`base_channels=32`) | `unet_groupnorm_v1` (`base_channels=32`) |
| **Hàm Loss** | `raw_full_image_l1` | `masked_l1` ($\epsilon = 10^{-6}$) |
| **Epochs** | 50 (Best tại Epoch **39**) | 50 (Best tại Epoch **39**; lưu mốc `best_30ep.pt` cho GAN) |
| **Batch Size / Workers** | `32` / `2` | `32` / `2` |
| **Optimizer** | `Adam(lr=3e-4, betas=(0.9, 0.999))` | `Adam(lr=3e-4, betas=(0.9, 0.999))` |
| **LR Scheduler** | `CosineAnnealingLR(T_max=50, eta_min=1e-6)` | `CosineAnnealingLR(T_max=50, eta_min=1e-6)` |
| **Augmentation** | Random Horizontal Flip ($p = 0.5$) | Random Horizontal Flip ($p = 0.5$) |
| **Precision / Grad Clip** | AMP FP16 (`GradScaler`) / `clip_grad_norm = 1.0` | AMP FP16 (`GradScaler`) / `clip_grad_norm = 1.0` |
| **Tài nguyên thực tế (T4)** | ~878s/epoch (~11.95h) · Peak VRAM `6.01–11.64 GB` | ~880s/epoch (~12.22h) · Peak VRAM `6.01 GB` |
| **Kết quả Best Val (3.279)** | MSE `59.28` · fMSE `556.07` · PSNR `34.77` | MSE `39.64` · fMSE `384.83` · PSNR `36.11` |
| **Kết quả Test (7.404)** | MSE `60.59` · fMSE `586.10` · PSNR `34.61` · SSIM `0.9815` | MSE `41.67` · fMSE `401.16` · PSNR `35.95` · SSIM `0.9845` |

---

### 2. U-Net + Spectral-Norm PatchGAN — Branch B (1-Scale) & Branch C (3-Scale Multi-Scale)

- **Mã nguồn:** `src/iharmony/models/unet.py` + `src/iharmony/models/discriminators.py` | `scripts/train_patchgan.py`
- **Thư mục run:** `runs/unet_branchB_1patchgan_50ep_all4_bs32_seed42/` và `runs/unet_branchC_3patchgan_50ep_all4_bs32_seed42/`
- **Quy mô:** Generator `UNet` (**7.76M**) + Discriminator `SN-PatchGAN` (1 scale ở Branch B / 3 scales ở Branch C)

![Kiến trúc U-Net PatchGAN Branch B và Branch C](figures/arch_02_unet_patchgan.png)

#### Luồng kiến trúc (Mermaid)
```mermaid
flowchart LR
    IN["Input [C, M] (4x256x256)"] --> G["Generator: UNet (from best_30ep.pt)"]
    G --> BLEND["comp_out = raw*M + C*(1-M)"]
    BLEND --> CATF["Concat [comp_out, M] (4x256x256)"]
    REAL["Real [Y, M] (4x256x256)"] --> D1["Branch B: 1x SN-PatchGAN (256x256)"]
    CATF --> D1
    REAL --> D3["Branch C: 3x Multi-Scale SN-PatchGAN (256, 128, 64)"]
    CATF --> D3
    D1 --> LD["LSGAN D Loss (real=0.9, fake=0.0)"]
    D3 --> LD
    G --> L1["100 * Masked L1 (raw, Y, M)"]
    L1 --> LG["Total G Loss = 100*L1 + LSGAN_G"]
    LD -.-> LG
```

#### Kiến trúc Discriminator & Cơ chế ổn định GAN
- **70×70 Spectral-Norm PatchGAN (`ndf=64`, input 4 kênh `[RGB, Mask]`):**
  1. `SN-Conv2d(4 → 64, k=4, s=2, p=1) → LeakyReLU(0.2)`
  2. `SN-Conv2d(64 → 128, k=4, s=2, p=1) → InstanceNorm2d → LeakyReLU(0.2)`
  3. `SN-Conv2d(128 → 256, k=4, s=2, p=1) → InstanceNorm2d → LeakyReLU(0.2)`
  4. `SN-Conv2d(256 → 512, k=4, s=1, p=1) → InstanceNorm2d → LeakyReLU(0.2)`
  5. `SN-Conv2d(512 → 1, k=4, s=1, p=1)` (bản đồ phán đoán patch-level).
- **Branch C (Multi-scale `num_D = 3`):** Dùng 3 bộ PatchGAN độc lập trên độ phân giải gốc `256²`, downsample `128²` và `64²` qua `AvgPool2d(3, stride=2, padding=1)`, lấy trung bình loss trên 3 scale.
- **Ổn định huấn luyện:** Khởi tạo Generator từ mốc `best_30ep.pt` (Val fMSE `397.88`), **đóng băng G tại Epoch 31** (D-Warmup), áp dụng `n_critic = 2` ở 3 epoch đầu (Ep 31–33), label smoothing `real_label = 0.9` với LSGAN:
  $$\mathcal{L}_D = \frac{1}{2K}\sum_{k=1}^{K}\Big[\text{MSE}(D_k(Y, M), 0.9) + \text{MSE}(D_k(I_{\text{out}}, M), 0)\Big], \qquad \mathcal{L}_G = 100\,\mathcal{L}_{\text{masked}} + \frac{1}{K}\sum_{k=1}^{K}\text{MSE}(D_k(I_{\text{out}}, M), 1)$$

#### Cấu hình huấn luyện
| Tham số | Run 3: Branch B (1-PatchGAN) | Run 4: Branch C (3-Multi-Scale PatchGAN) |
|---|---|---|
| **Checkpoint khởi tạo** | `unet_masked_l1/best_30ep.pt` (Ep 29, Val fMSE `397.88`) | `unet_masked_l1/best_30ep.pt` |
| **Giai đoạn fine-tune** | Epoch `31 → 50` (20 epochs) | Epoch `31 → 49` (19/20 epochs) |
| **Discriminator** | `1× SN-PatchGAN` (`ndf=64`) | `3× Multi-Scale SN-PatchGAN` (`num_D=3, ndf=64`) |
| **Optimizer G / D** | `Adam(lr_G=5e-5, betas=(0.5, 0.999))` / `Adam(lr_D=1e-4)` | `Adam(lr_G=5e-5, betas=(0.5, 0.999))` / `Adam(lr_D=1e-4)` |
| **Scheduler G / D** | `CosineAnnealingLR` (`T_max=19` / `20`, `eta_min=1e-6`) | `CosineAnnealingLR` (`T_max=19` / `20`, `eta_min=1e-6`) |
| **Cân bằng GAN** | `lambda_l1=100`, `real_label=0.9`, D-Warmup Ep 31, `n_critic=2` (Ep 31–33) | Giống Branch B |
| **Tài nguyên thực tế (T4)** | ~1.618s/epoch (~8.99h) · Peak VRAM `6.13 GB` | ~1.795s/epoch (~9.48h) · Peak VRAM `6.24 GB` |
| **Kết quả Best Val (3.279)** | Ep **38**: MSE `40.20` · fMSE `388.14` · PSNR `36.08` | Ep **38**: MSE `40.05` · fMSE `386.84` · PSNR `36.10` |

---

### 3. Improved U-Net v2 (Hybrid)

- **Mã nguồn:** `src/iharmony/models/improved_unet_v2.py` + `src/iharmony/models/blocks.py` | `scripts/train_improved_unet.py --version v2`
- **Thư mục run:** `runs/improved_unet_v2_hybrid_probe10ep_bs32_seed42/`
- **Quy mô:** **15.54M tham số** (98 tensors)

![Kiến trúc Improved U-Net v2 Hybrid](figures/arch_03_improved_unet_v2.png)

#### Luồng kiến trúc (Mermaid)
```mermaid
flowchart LR
    IN["Input [C, M] (4x256x256)"] --> ENC["GatedDoubleConv Encoder x4 (32->64->128->256)"]
    ENC --> BOT["GatedDoubleConv Bottleneck (512x16x16)"]
    BOT --> ATTB["BottleneckFGBGCrossAttention (8 heads, K/V=BG only)"]
    ATTB --> DEC["BilinearUp + AttentionGate + StdDoubleConv x4"]
    ENC -.->|"Gated Skips (att1..att4)"| DEC
    DEC --> HEAD["GlobalLocalColorAffineHead: (I + dW)*C + db"]
    IN -.->|"Composite C"| HEAD
```

#### 4 Cải tiến cốt lõi so với Baseline U-Net
1. **Gated Convolution (`GatedDoubleConv`) ở toàn bộ Encoder & Bottleneck:** Nhân đôi kênh đầu ra rồi tách thành nhánh đặc trưng và nhánh cổng $\text{LeakyReLU}(\text{GN}(\text{feat})) \odot \sigma(\text{gate})$, giúp học tách bạch ranh giới foreground/background ngay từ các tầng thấp.
2. **Bottleneck FG-BG Cross-Attention (8 heads ở độ phân giải `16×16`):** Dùng `key_padding_mask = (m_low > 0.5)` để ép các token foreground truy vấn thông tin ánh sáng **chỉ từ vùng background**, sau đó cập nhật chọn lọc vùng foreground (`m_low > 0.2`) với hệ số residual học được $\gamma_1, \gamma_2$ khởi tạo bằng 0.
3. **Bilinear Upsampling + Attention Gate:** Thay `ConvTranspose2d` bằng `Upsample(bilinear) + Conv3×3` để loại bỏ nhiễu ô bàn cờ (checkerboard artifacts), kết hợp `AttentionGate` lọc nhiễu trên cả 4 đường skip connection.
4. **Global + Local Color Affine Head:** Thay vì dự đoán trực tiếp pixel RGB (dễ làm mờ kết cấu vật thể), đầu ra dự đoán trường biến đổi màu affine 12 kênh (ma trận $3\times 3$ $\Delta W$ + độ lệch $\Delta b$, cộng nhánh Local 3×3 và Global GAP-FC):
   $$\hat{I}_{\text{raw}} = \text{clip}\big((I + \Delta W) \cdot I_{\text{comp}} + \Delta b,\ 0,\ 1\big)$$

#### Cấu hình huấn luyện
| Tham số | Giá trị thực tế (`v2_best.pt`) |
|---|---|
| **Kiến trúc** | `ImprovedHarmonizationUNet_v2` (`base_channels=32`, `num_heads=8`) |
| **Hàm Loss** | $\mathcal{L} = \mathcal{L}_{\text{masked\_L1}} + 0.1 \cdot \mathcal{L}_{\text{bg\_L1}}$ (`bg_weight = 0.1`) |
| **Epochs** | 50 (Best Val fMSE tại Epoch **22**, Best Val MSE tại Epoch **29**; có dấu hiệu overfit nhẹ dần tới Ep 50 `404.45`) |
| **Batch Size / LR** | `32` · `lr = 3e-4 → min_lr = 1e-6` (CosineAnnealing) |
| **Tài nguyên thực tế (T4)** | ~1.728s/epoch (~24.00h / 50 ep) · Peak VRAM `8.27–12.61 GB` |
| **Kết quả Best Val (3.279)** | Ep **22**: MSE `39.71` · fMSE `382.23` · PSNR `36.24` · fPSNR `24.67` |

---

### 4. Improved U-Net v3 (`dinov2_lab`) — Mô hình Mạnh Nhất Hiện Tại

- **Mã nguồn:** `src/iharmony/models/improved_unet_v3.py` | `scripts/train_improved_unet.py --version v3`
- **Thư mục run:** `runs/improved_unet_v3_dinov2_lab_30ep_bs32_seed42/`
- **Quy mô:** **33.29M tổng tham số** (**11.23M Trainable** + **22.06M Frozen DINOv2 ViT-S/14**, 287 tensors)

![Kiến trúc Improved U-Net v3 DINOv2 Lab](figures/arch_04_improved_unet_v3_dinov2.png)

#### Luồng kiến trúc (Mermaid)
```mermaid
flowchart LR
    IN["Input [C, M] (4x256x256)"] --> E12["GatedDoubleConv enc1, enc2 (32, 64)"]
    E12 --> E34["StdDoubleConv enc3, enc4, bot (128, 256, 512)"]
    IN -->|"Comp RGB -> 224x224"| DINO["Frozen DINOv2 ViT-S/14 (22.06M)"]
    DINO --> PROJ["Conv1x1(384->512) + GN(8)"]
    E34 -->|"Value V (Spatial 512)"| SATT["SemanticGuidedFGBGCrossAttention (8 heads)"]
    PROJ -->|"Query Q, Key K (+PosEmb)"| SATT
    SATT --> DEC["BilinearUp + AttentionGate + StdDoubleConv x4"]
    E12 -.->|"Skip att1, att2"| DEC
    E34 -.->|"Skip att3, att4"| DEC
    DEC --> HEAD["HybridColorAffineResidualHead"]
    IN -.->|"Comp C, Mask M"| HEAD
```

#### 4 Đột phá kỹ thuật vượt trội so với v2
1. **Nhánh ngữ nghĩa đóng băng `FrozenDINOv2Guide` (`dinov2_vits14`):** Trích xuất đặc trưng ngữ nghĩa giàu thông tin vật liệu/bối cảnh trên ảnh `224×224` (đóng băng hoàn toàn trọng số, luôn giữ ở `.eval()`), chiếu về `512 × 16 × 16` qua `Conv1×1 + GroupNorm`.
2. **Semantic-Guided FG-BG Cross-Attention:** Dùng đặc trưng ngữ nghĩa DINOv2 + `pos_emb` làm **Query & Key** (tìm các vùng nền có cùng ngữ nghĩa/vật liệu chiếu sáng), trong khi lấy **Value** từ đặc trưng không gian/màu sắc ở Bottleneck CNN — giải quyết tận gốc nhược điểm overfit và khớp sai nền của v2!
3. **Hybrid Encoder + Hybrid Color Affine Residual Head:** Chỉ giữ `GatedDoubleConv` ở 2 tầng nông (`enc1, enc2`) để giảm tham số thừa (từ 15.54M xuống 11.23M trainable), đồng thời bổ sung nhánh **Chroma Residual** (`Conv3×3 → GN → LReLU → Conv3×3 → Tanh` nhân hệ số `res_scale` học được trên vùng foreground) lên trên ảnh Affine để tinh chỉnh các hiệu ứng phi tuyến (bóng đổ, chuyển sắc mềm).
4. **Hàm Loss cảm nhận màu CIE Lab + Foreground Color Augmentation:**
   $$\mathcal{L}_{\text{v3}} = \mathcal{L}_{\text{masked\_L1}} + 0.10\,\mathcal{L}_{\text{bg\_L1}} + 0.15\,\mathcal{L}_{\text{masked\_Lab\_L1}}$$
   kết hợp tăng cường màu sắc ngẫu nhiên trên vùng foreground (`fg_color_aug_prob = 0.35`) giúp tổng quát hóa cực tốt trên tập Test.

#### Cấu hình huấn luyện
| Tham số | Giá trị thực tế (`config.json`) |
|---|---|
| **Kiến trúc** | `improved_unet_v3_dinov2_lab` (`base_channels=32`, `num_heads=8`) |
| **Tham số** | `11.23M` trainable / `33.29M` total (DINOv2 ViT-S/14 frozen) |
| **Epochs** | `30` (Best tại Epoch **30** — vẫn còn dư địa giảm thêm nếu train tiếp!) |
| **Optimizer / Weight Decay** | `AdamW(lr=2.5e-4, weight_decay=1e-4)` |
| **Scheduler** | `CosineAnnealingLR(T_max=30, eta_min=1e-6)` |
| **Trọng số Loss** | `lambda_bg = 0.10`, `lambda_lab = 0.15` |
| **Data Augmentation** | Horizontal Flip ($0.5$) + Foreground Color Augmentation (`fg_color_aug_prob = 0.35`) |
| **Tài nguyên thực tế (T4)** | ~1.899s/epoch (~15.82h / 30 ep) · Peak VRAM `8.86 GB` |
| **Kết quả Best Val (3.279)** | Ep **30**: MSE `36.89` · fMSE `361.33` · PSNR `36.38` · fPSNR `24.82` |
| **Kết quả Test (7.404)** | **MSE `37.01` · fMSE `369.64` · PSNR `36.29` · fPSNR `24.58` · SSIM `0.9854`** |

---

## Phần II — Các Kiến trúc Pre-trained Benchmarks

### 5. RainNet Pretrained (Ling et al., CVPR 2021)

- **Paper:** *"Region-aware Adaptive Instance Normalization for Image Harmonization"*, CVPR 2021
- **Mã nguồn tích hợp:** `src/iharmony/models/rainnet.py` (`RAIN`, `UnetBlockCodec`, `RainNet`)
- **Checkpoint trên Drive:** `runs/RainNet_Pretrained/RainNet_net_G_last.pth` (**54.76M tham số**, 51 tensors)

![Kiến trúc RainNet Pretrained CVPR 2021](figures/arch_05_rainnet_pretrained.png)

#### Luồng kiến trúc (Mermaid)
```mermaid
flowchart LR
    IN["Comp [-1,1] (3x256x256) + Mask {0,1}"] --> ENC["8-Level Strided Conv4x4 Encoder + InstanceNorm (64->512, 256->1)"]
    ENC --> BOT["Innermost Bottleneck (1x1 -> 2x2) + RAIN"]
    BOT --> UDEC["3x UnetBlock Decoder + RAIN + Dropout (4->16)"]
    UDEC --> L11["layer11..13 + RAIN + 1x1 Sigmoid Skip Attention (32->128)"]
    ENC -.->|"Skip Connections"| UDEC
    ENC -.->|"Skip Connections"| L11
    L11 --> OUT["ConvT(128->3) + Tanh [-1,1] -> Rescale [0,1] + BG Blend"]
```

#### Cơ chế RAIN (Region-aware Adaptive Instance Normalization)
Ở cả 7 tầng Decoder, mô-đun `RAIN` tách riêng vùng foreground ($M$) và background ($1 - M$) trên bản đồ đặc trưng, chuẩn hóa foreground rồi chuyển giao trung bình $\mu_{\text{bg}}$ và độ lệch chuẩn $\sigma_{\text{bg}}$ của background sang foreground trước khi qua tham số affine học được $(\gamma_{\text{fg}}, \beta_{\text{fg}}, \gamma_{\text{bg}}, \beta_{\text{bg}})$:
$$x_{\text{fg}}' = \left(\frac{x - \mu_{\text{fg}}}{\sigma_{\text{fg}}}\,\sigma_{\text{bg}} + \mu_{\text{bg}}\right)(1 + \gamma_{\text{fg}}) + \beta_{\text{fg}}$$

#### Cấu hình Pre-train gốc & Thiết lập suy luận
| Hạng mục | Cấu hình |
|---|---|
| **Repo & Checkpoint gốc** | `junleen/RainNet` — bản chính thức 256×256 `net_G_last.pth` |
| **Cấu hình mạng** | `input_nc=3`, `output_nc=3`, `ngf=64`, `norm_type_indicator=[0]*7 + [1]*7` (IN ở Encoder, RAIN ở Decoder), `use_attention=True`, `use_dropout=True` |
| **Cấu hình pre-train gốc** | Tập iHarmony4 (256×256), `WGAN-GP` + `OrgDiscriminator` + `100·L1`, `Adam(lr=2e-4, beta1=0.5)`, `niter=100` |
| **Tiền/Hậu xử lý đánh giá** | Chuẩn hóa ảnh vào `[-1, 1]`, mask `{0, 1}`, chạy **float32** (tránh sai số fp16 trong phương sai RAIN), quy đổi về `[0, 1]` và ghép nền gốc |
| **Kết quả Test (7.404)** | MSE `46.76` · fMSE `531.44` · PSNR `35.61` · fPSNR `22.74` |

---

### 6. Harmony Transformer — HT+ (`HT_2H9L_allihd`, Pretrained)

- **Paper:** Guo et al., *"Image Harmonization with Transformer"* (ICCV 2021) / *"Transformer for Image Harmonization and Beyond"* (TPAMI)
- **Mã nguồn tích hợp:** `src/iharmony/models/ht_plus.py` (`PatchRearrange`, `HTTransformerEncoder*`, `HTGenerator`)
- **Checkpoint trên Drive:** `runs/ht_plus_pretrained/HT_2H9L_allihd_latest_net_G.pth` (**4.77M tham số**, 112 tensors)

![Kiến trúc Harmony Transformer HT+ 2H9L](figures/arch_06_ht_plus_pretrained.png)

#### Luồng kiến trúc (Mermaid)
```mermaid
flowchart LR
    IN["Input [C, M] in [0,1] (4x256x256)"] --> PE["PatchRearrange 4x4 (4096x64) -> Linear(64->256)"]
    POS["Fixed 2D Sine PosEmb (64x64 -> 4096x256)"] --> ADD["Add to Q, K"]
    PE --> ADD
    ADD --> TR["9x Transformer Encoder Layers (nhead=2, d=256, FFN=512, GELU, Post-LN)"]
    TR --> DEC["Reshape (256x64x64) -> ConvT2d(256->3, k=4, s=4) + Tanh"]
    DEC --> BLEND["Clamp [0,1] -> out = raw*M + C*(1-M)"]
```

#### Cấu hình Pre-train gốc & Thiết lập suy luận
| Hạng mục | Cấu hình |
|---|---|
| **Repo & Checkpoint gốc** | `zhenglab/HarmonyTransformer` — `FCHT_2H9L_allihd/latest_net_G.pth` |
| **Cấu hình mạng** | `--model ht`, `ksize=stride=4`, `dim=256`, `tr_r_enc_head=2`, `tr_r_enc_layers=9`, `dim_forward=2` (`FFN=512`), `tr_act=gelu`, `dropout=0.0`, `pos_type=sincos` |
| **Cấu hình pre-train gốc** | Toàn bộ iHarmony4 (`IHD`), `L1 loss (lambda_L1=100)`, `Adam(lr=1e-4, beta1=0.5, beta2=0.999)`, `niter=30 + niter_decay=30` (60 epochs, linear decay) |
| **Lưu ý chuẩn hóa quan trọng** | Input và Output đều nằm trực tiếp trong **`[0, 1]`** (`Normalize((0,0,0), (1,1,1))`, không dùng `[-1, 1]`), đầu ra `Tanh` được cắt `clamp(0, 1)` rồi ghép nền gốc |
| **Thiết lập suy luận đánh giá** | Chạy **float32**, `batch_size=12..16` (tối ưu VRAM cho 4.096 tokens trên T4 với `need_weights=False`) |
| **Kết quả Test (7.404)** | MSE `36.99` · fMSE `387.83` · PSNR `36.17` · fPSNR `24.42` · SSIM `0.9593` · fSSIM `0.8645` |

---

## Phần III — Nhóm Mô hình Diffusion Pre-trained (Mở rộng — `checkpoints/`)

Hai checkpoint Diffusion đã được tải sẵn về thư mục `ImageHarmonization/checkpoints/` trên Drive cho nhánh nghiên cứu mở rộng (chưa chạy benchmark 7.404 ảnh):

| Mô hình | Paper / Nguồn | File Checkpoint trên Drive | Kiến trúc & Ghi chú sử dụng |
|---|---|---|---|
| **DiffHarmony** | Zhou et al., ICMR 2024 (`nicecv/DiffHarmony`) | `checkpoints/DiffHarmonyCheckpoints.zip` (`9.36 GB`) | **Latent Diffusion Model (LDM)** fine-tune từ Stable Diffusion chuyên cho Image Harmonization chuẩn iHarmony4; có sẵn script inference/evaluate. |
| **IC-Light (`fbc`)** | Zhang, Rao, Agrawala, ICLR 2025 (`lllyasviel/IC-Light`) | `checkpoints/iclight_sd15_fbc.safetensors` (`536.9 MB`) | **Stable Diffusion 1.5** relighting có điều kiện foreground + background + text (`fbc`); chỉnh chiếu sáng tự nhiên mạnh, chạy được trên T4. |

---

## Phần IV — Bảng So sánh Tổng hợp Toàn bộ Kiến trúc

| # | Mô hình | Loại | Tham số (Train / Tot) | Độ phân giải / Precision | Loss chính | Best Val fMSE ↓ | Test fMSE ↓ | Test PSNR ↑ |
|---|---|---|---:|---|---|---:|---:|---:|
| 1 | **Baseline U-Net (Full L1)** | Tự train (50 ep) | 7.76M | 256² / AMP FP16 | Full L1 | 556.07 *(ep 39)* | 586.10 | 34.61 |
| 2 | **Baseline U-Net (Masked L1)** | Tự train (50 ep) | 7.76M | 256² / AMP FP16 | Masked L1 | 384.83 *(ep 39)* | 401.16 | 35.95 |
| 3 | **U-Net + 1 PatchGAN (Branch B)** | Fine-tune (ep 31–50) | 7.76M (+D) | 256² / AMP FP16 | 100·Masked L1 + LSGAN | 388.14 *(ep 38)* | *(chạy eval)* | *(chạy eval)* |
| 4 | **U-Net + 3 PatchGAN (Branch C)** | Fine-tune (ep 31–49) | 7.76M (+3D) | 256² / AMP FP16 | 100·Masked L1 + 3-Scale LSGAN | 386.84 *(ep 38)* | *(chạy eval)* | *(chạy eval)* |
| 5 | **Improved U-Net v2 (Hybrid)** | Tự train (50 ep) | 15.54M | 256² / AMP FP16 | Masked L1 + 0.1·Bg L1 | 382.23 *(ep 22)* | *(chạy eval)* | *(chạy eval)* |
| 6 | **Improved U-Net v3 (DINOv2+Lab)** | Tự train (30 ep) | **11.23M / 33.29M** | 256² / AMP FP16 | Masked L1 + 0.1·Bg + 0.15·Lab | **361.33 *(ep 30)*** | **369.64** | **36.29** |
| 7 | **RainNet (CVPR 2021)** | Pre-trained | 54.76M | 256² / FP32 | WGAN-GP + 100·L1 *(pre-train)* | — | 531.44 | 35.61 |
| 8 | **HT+ (`HT_2H9L_allihd`)** | Pre-trained | **4.77M** | 256² / FP32 | 100·L1 *(pre-train)* | — | 387.83 | 36.17 |

### Kết luận kỹ thuật chính
- **Tác động lớn nhất ở tầng Baseline:** Chuyển từ `Full L1` sang `Masked L1` giảm Test fMSE tới **31.6%** (`586.10 → 401.16`), vượt cả `RainNet` pre-trained (`531.44`) dù chỉ dùng 1/7 số tham số (`7.76M` vs `54.76M`).
- **Hiệu quả của PatchGAN (Branch B & C):** Thêm adversarial loss giúp sắc nét cục bộ nhưng không cải thiện fMSE so với Masked L1 thuần (`388.14` / `386.84` vs `384.83` trên Val), phù hợp lý thuyết đánh đổi Distortion–Perception.
- **Đột phá từ Ngữ nghĩa & Không gian màu (v3 DINOv2 + Lab):** Chỉ với 30 epochs và 11.23M tham số huấn luyện, `Improved U-Net v3` đạt Test fMSE **`369.64`** và SSIM **`0.9854`**, vượt qua cả mô hình Transformer pre-trained `HT+` (`387.83`).
