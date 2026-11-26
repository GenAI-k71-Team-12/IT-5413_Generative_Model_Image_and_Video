# Cấu Trúc Thư Mục Google Drive & Quy Tắc Làm Việc Nhóm — Image Harmonization

Tài liệu này mô tả **cấu trúc chuẩn của thư mục Google Drive dùng chung** (`ImageHarmonization`) và **bộ quy tắc bắt buộc** về lưu trữ, chia dữ liệu, huấn luyện, đánh giá công bằng và quản lý mã nguồn GitHub để cả nhóm (và các thành viên mới sau này) phối hợp đồng bộ, không ghi đè dữ liệu và không xảy ra rò rỉ dữ liệu (*data leakage*).

- **Google Drive dùng chung:** [ImageHarmonization (Google Drive)](https://drive.google.com/drive/folders/1oLmarz5jaqyxyRg762QeynYq4dQy-6mm?usp=sharing)
- **Tài liệu liên quan:** [Kiến trúc & Cấu hình các mô hình](architectures_and_configs.md) · [Tổng hợp Paper & Pretrained Links](image_harmonization_models_links.md)

---

## 1. Cấu Trúc Thư Mục Google Drive (`ImageHarmonization/`)

Tổng dung lượng hiện tại trên Drive khoảng **~68 GB**. Mọi đường dẫn trong Colab đều giả định thư mục gốc là:
```text
/content/drive/MyDrive/ImageHarmonization/
```

```text
ImageHarmonization/
├── iHarmony4_dataset_offical/                  # [READ-ONLY] Bộ dữ liệu gốc full-res (~50.3 GB)
│   ├── HCOCO.zip                               #   7.4 GB
│   ├── HAdobe5k.zip                            #  37.2 GB
│   ├── HFlickr.zip                             #   2.4 GB
│   ├── Hday2night.zip                          #  56.3 MB
│   └── ccHarmony.zip                           #   3.3 GB (tập mở rộng có color checker)
│
├── iHarmony4_dataset_resize_256/               # [READ-ONLY] Bộ chuẩn 256×256 để train/test (~3.5 GB)
│   ├── HCOCO_256.zip                           #   2.4 GB (38.545 train / 4.283 test)
│   ├── HAdobe5k_256.zip                        # 642.6 MB (19.437 train / 2.160 test)
│   ├── HFlickr_256.zip                         # 401.2 MB ( 7.449 train /   828 test)
│   └── Hday2night_256.zip                      #  13.7 MB (   311 train /   133 test)
│
├── iHarmony4_groups_train_test/                # [READ-ONLY] Phân tầng 6 nhóm độ khó (~66 MB)
│   ├── summary.json                            #   Ngưỡng median mask_ratio (4.72%) & q1/q2 DeltaE76 (11.87 / 18.15)
│   ├── group_counts.csv                        #   Thống kê số lượng từng nhóm độ khó
│   ├── train.csv                               #   27.0 MB — nhãn độ khó tập train
│   ├── test.csv                                #    3.0 MB — nhãn độ khó 7.404 ảnh test (dùng trong eval)
│   ├── metadata.csv                            #   30.1 MB — metadata đầy đủ 73.146 mẫu
│   ├── iHarmony4_splits_groups.zip             #    5.8 MB — bản nén sao lưu toàn bộ split/group
│   ├── groups/                                 #   Danh sách ảnh chia theo 6 nhóm chung
│   ├── groups_by_dataset/                      #   Danh sách ảnh chia theo dataset × 6 nhóm
│   └── manifests/                              #   Manifest đường dẫn chuẩn hóa
│
├── checkpoints/                                # [APPEND-ONLY] Checkpoint Pretrained Diffusion cỡ lớn (~9.2 GB)
│   ├── DiffHarmonyCheckpoints.zip              #   8.7 GB (DiffHarmony — ICMR 2024)
│   └── iclight_sd15_fbc.safetensors            # 512.0 MB (IC-Light SD1.5 FBC — ICLR 2025)
│
├── runs/                                       # [APPEND-ONLY] Kết quả train & eval của từng mô hình
│   ├── unet_full_l1_all4_bs32_seed42/          #   Run 1: Baseline U-Net Full L1
│   ├── unet_masked_l1_50ep_all4_bs32_seed42/   #   Run 2: Baseline U-Net Masked L1 (+ best_30ep.pt)
│   ├── unet_branchB_1patchgan_50ep_all4_bs32_seed42/  # Run 3: U-Net + 1 SN-PatchGAN
│   ├── unet_branchC_3patchgan_50ep_all4_bs32_seed42/  # Run 4: U-Net + 3 Multi-Scale SN-PatchGAN
│   ├── improved_unet_v2_hybrid_probe10ep_bs32_seed42/ # Run 5: Improved U-Net v2 (Hybrid)
│   ├── improved_unet_v3_dinov2_lab_30ep_bs32_seed42/  # Run 6: Improved U-Net v3 (DINOv2 + Lab)
│   ├── RainNet_Pretrained/                     #   Run 7: RainNet CVPR 2021 (256×256)
│   ├── ht_plus_pretrained/                     #   Run 8: Harmony Transformer HT+ (HT_2H9L)
│   └── evaluation_all_models/                  #   Tổng hợp kết quả chấm đồng bộ tất cả mô hình
│       ├── all_models_test_metrics.json        #     JSON đầy đủ toàn bộ 8+1 mô hình
│       ├── summary_overall.csv                 #     Bảng 1: Tổng hợp 7.404 ảnh Test
│       ├── summary_by_dataset.csv              #     Bảng 2: Theo 4 sub-dataset
│       ├── summary_by_difficulty_group.csv     #     Bảng 3: Theo 6 nhóm độ khó
│       ├── comparison_all_models_4datasets.png #     Lưới ảnh so sánh ngẫu nhiên 4 dataset
│       ├── comparison_all_models_hardest_large_high.png # Lưới ảnh nhóm khó nhất (large_high)
│       ├── charts/                             #     Biểu đồ PNG độ phân giải cao (bar, heatmap, scatter)
│       ├── previews/                           #     Ảnh preview 6 mẫu / mô hình
│       ├── per_image_metrics/                  #     CSV chi tiết từng ảnh (7.404 dòng / mô hình)
│       └── rainnet_audit/                      #     Kiểm định riêng RainNet (R1–R4: Blind vs GT-BG*)
│           ├── rainnet_audit_metrics.json
│           ├── rainnet_audit_summary.csv
│           ├── rainnet_audit_by_dataset.csv
│           ├── rainnet_audit_by_group.csv
│           ├── rainnet_blind_vs_official_chart.png
│           ├── rainnet_blind_vs_official_visual_grid.png
│           └── per_image_metrics/
│
├── notebooks/                                  # [WORKING] Bản sao lưu Colab notebooks đang chạy
└── docs/                                       # [WORKING] Tài liệu tham khảo nhanh trên Drive
    └── image_harmonization_models_links.md
```

### Quyền thao tác theo phân loại thư mục

| Phân loại | Thư mục áp dụng | Quy tắc thao tác |
|---|---|---|
| **`READ-ONLY` (Cấm sửa)** | `iHarmony4_dataset_offical/`, `iHarmony4_dataset_resize_256/`, `iHarmony4_groups_train_test/` | Chỉ đọc/giải nén sang `/content/` trên Colab. **Nghiêm cấm** xóa, đổi tên, ghi đè ảnh hoặc sửa file `*_train.txt` / `*_test.txt`. |
| **`APPEND-ONLY` (Chỉ thêm mới)** | `runs/`, `checkpoints/` | Mỗi thử nghiệm mới tạo **một thư mục con riêng** theo chuẩn đặt tên. Không ghi đè checkpoint `best.pt` hay `test_metrics.json` của các run đã chốt. |
| **`WORKING` (Cập nhật có kiểm soát)** | `runs/evaluation_all_models/`, `notebooks/`, `docs/` | Khi có mô hình mới, chạy lại notebook đánh giá để cập nhật `evaluation_all_models/`. Code chuẩn luôn quản lý chính trên GitHub. |

---

## 2. Quy Tắc Lưu Trữ & Phân Tách GitHub vs Google Drive

Để repository GitHub luôn gọn nhẹ (~ vài MB), dễ clone và review code:

| Loại file | Lưu trên GitHub? | Lưu trên Google Drive? | Ghi chú |
|---|:---:|:---:|---|
| Mã nguồn Python (`src/`, `scripts/`), `pyproject.toml`, `requirements.txt` | **CÓ** | Không bắt buộc | Nguồn chân lý (source of truth) là GitHub |
| Notebooks sạch (`notebooks/01..09_*.ipynb`) | **CÓ** | Bản chạy Colab | Xóa `metadata.widgets` & ảnh base64 nặng trước khi commit |
| Tài liệu `.md` & hình vẽ sơ đồ nhẹ (`docs/figures/*.png`) | **CÓ** | Bản sao tùy chọn | Nên giữ ảnh sơ đồ < 500 KB/ảnh |
| Log nhẹ: `config.json`, `history.json`, `summary_*.csv`, `all_models_test_metrics.json` | **CÓ** (`results/`) | **CÓ** (`runs/`) | Tổng dung lượng chỉ ~ vài trăm KB |
| Bộ dữ liệu ảnh & file `.zip` (`iHarmony4`, `ccHarmony`) | **KHÔNG** | **CÓ** | Đã chặn bởi `.gitignore` |
| Trọng số mô hình (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`) | **KHÔNG** | **CÓ** (`runs/`, `checkpoints/`) | Đã chặn bởi `.gitignore` |
| CSV theo từng ảnh (> 1 MB/file) & ảnh preview/grid nặng (> 2 MB) | **KHÔNG** | **CÓ** (`runs/evaluation_all_models/`) | Chứa 7.404 dòng/file hoặc lưới PNG 10–15 MB |

---

## 3. Quy Chuẩn Đặt Tên & Cấu Trúc Bên Trong `runs/`

### 3.1. Quy ước đặt tên thư mục run
- **Mô hình tự huấn luyện:**
  ```text
  <arch>_<variant>_<epochs>ep_all4_bs<batch_size>_seed<seed>
  ```
  - Ví dụ tốt: `improved_unet_v4_swin_lab_50ep_all4_bs32_seed42`, `unet_focal_l1_50ep_all4_bs32_seed42`.
  - Tránh: đặt tên chung chung (`test1`, `new_model`, `final_v2_real`), chứa dấu cách hoặc tiếng Việt có dấu.
- **Mô hình pre-trained:**
  ```text
  <model_name>_pretrained
  ```
  - Ví dụ: `RainNet_Pretrained`, `ht_plus_pretrained`, `diffharmony_pretrained`, `iclight_pretrained`.

### 3.2. Các file bắt buộc trong mỗi thư mục run tự huấn luyện
Mỗi thư mục `runs/<run_name>/` **bắt buộc** có đầy đủ các file sau:
1. `best.pt` — Checkpoint có **Val fMSE thấp nhất**. Dictionary lưu bên trong **phải có đủ**:
   - `'model'` (`state_dict`), `'epoch'`, `'cfg'` (dict cấu hình), `'val'` (dict kết quả Val), `'fingerprint'` (mã SHA-256 split).
2. `last.pt` — Checkpoint của epoch gần nhất (kèm `'optimizer'`, `'scheduler'`, `'scaler'`) để **resume** khi Colab bị ngắt kết nối.
3. `config.json` — Toàn bộ siêu tham số huấn luyện.
4. `history.json` — Lịch sử loss và chỉ số Val theo từng epoch (để vẽ learning curve).
5. `test_metrics_unified.json` — Kết quả đánh giá chuẩn hóa trên 7.404 ảnh Test (do notebook/script đánh giá tự sinh).
6. *(Tùy chọn)* `val_001.png ... val_0NN.png`, `test_preview_ep<N>_unified.png`.

### 3.3. Quy tắc an toàn khi chạy trên Google Colab
- **Ghi checkpoint nguyên tử (Atomic Save):** Luôn lưu vào file tạm `best.pt.tmp` rồi gọi `os.replace("best.pt.tmp", "best.pt")` (đã cài sẵn trong `iharmony.utils.atomic_torch_save`) để tránh hỏng file `.pt` nếu Colab mất kết nối đúng lúc đang ghi lên Drive.
- **Giải nén về SSD cục bộ `/content/`:** Luôn giải nén `*_256.zip` vào `/content/iharmony4_256/` trước khi tạo `DataLoader`. **Tuyệt đối không** trỏ `DataLoader` đọc trực tiếp hàng chục nghìn ảnh từ `/content/drive/MyDrive/` (chậm gấp 10–20 lần và dễ bị Google Drive giới hạn quota I/O).
- **Tránh tạo thư mục trùng:** Không lưu run mới vào `iHarmony4_dataset_resize_256/runs/` (nếu còn sót); mọi run đều tập trung về `ImageHarmonization/runs/`.
- **Dọn tên notebook trên Drive:** Không để tích tụ nhiều file `"Bản sao của ..."` hoặc `"Copy of ..."`; chốt tên gọn theo số thứ tự `01_...` đến `09_...` khớp với thư mục `notebooks/` trên GitHub.

---

## 4. Quy Tắc Khoa Học: Chia Dữ Liệu, Tái Lập & Đánh Giá Công Bằng

Đây là các nguyên tắc **bất di bất dịch** để mọi kết quả trong báo cáo/khóa luận có thể bảo vệ vững chắc trước hội đồng:

### 4.1. Chia dữ liệu cố định & Xác minh 0% Data Leakage
- **Tập dữ liệu chuẩn:** iHarmony4 resize `256 × 256` (`BICUBIC` cho ảnh RGB, `NEAREST` cho mask nhị phân giữ nguyên `{0, 255}`).
- **Quy tắc chia Train / Val / Test:**
  - Tập Test giữ nguyên **7.404 ảnh** chính thức từ 4 file `*_test.txt`.
  - Tập Train gốc (65.742 ảnh) được chia `95% Train (62.463)` / `5% Val (3.279)` với `seed = 42`, **nhóm theo `sourceID` của ảnh nền gốc** (không bao giờ chia ngẫu nhiên theo ảnh composite, vì một ảnh nền gốc có nhiều phiên bản ghép màu khác nhau).
- **Mã xác minh SHA-256 bắt buộc (`fingerprint`):**
  ```text
  aad71bde27817a76a2ab959e795b0798dff7fe526c5fe16ec6663e43e79de6be
  ```
  - Mọi script train và eval đều gọi `build_records(seed=42, val_fraction=0.05)` và **assert khớp tuyệt đối** mã `fingerprint` này.
- **Nguyên tắc mù Test (No Test Peeking):** Chỉ dùng tập **Val (3.279 ảnh)** để chọn `best.pt`, điều chỉnh learning rate, hàm loss hay kiến trúc. Tập **Test (7.404 ảnh)** chỉ chạy chấm điểm cuối cùng sau khi đã chốt mô hình.

### 4.2. Quy tắc Đánh Giá Công Bằng 100% Blind Test (Bài học từ RainNet & HT+)
1. **Tuyệt đối KHÔNG dùng ảnh Ground-Truth `real` ở đầu vào hoặc bước ghép nền đầu ra:**
   - Đầu vào mạng **chỉ được phép** lấy từ `composite_images/*.jpg` (`comp`) và `masks/*.png` (`mask`).
   - Hậu xử lý ghép nền chuẩn chung cho **tất cả** mô hình:
     $$I_{\text{out}} = \text{clip}\big(\hat{I}_{\text{raw}} \odot M + I_{\text{comp}} \odot (1 - M),\ 0,\ 1\big)$$
   - *Cảnh báo thực nghiệm đã kiểm chứng:* Repo gốc `junleen/RainNet` (`data/iharmony4_dataset.py`, dòng 72) có đoạn `comp = comp * mask + real * (1 - mask)` gây rò rỉ nền Ground-Truth, làm giảm ảo `fMSE` tới **-87.74** (`619.18 → 531.44`), đẩy ảo `PSNR` thêm **+1.35 dB** (`34.27 → 35.62`) và đẩy `SSIM` toàn ảnh lên ảo **0.9868** (dù `fSSIM` chỉ `0.8742`).
2. **Chấm điểm trực tiếp trên Float32 trong `[0, 1]`:**
   - Không lưu ảnh kết quả ra JPEG rồi mở lại để chấm (như `ih_evaluation.py` của HT+, gây lệch nén sắc độ 4:2:0 và tính sai thang SSIM `[0, 255]`).
   - Không cắt cụt `floor` sang `uint8` (như `util.tensor2im` của RainNet, gây lệch `-0.5` mức xám).
3. **Bộ chỉ số & 3 bảng kết quả bắt buộc khi công bố:**
   - Đủ **7 chỉ số định lượng**: `MSE ↓`, `fMSE ↓`, `PSNR ↑`, `fPSNR ↑`, `SSIM ↑`, `fSSIM ↑`, `fΔE76 ↓` (CIE Lab D65) + **tham số (Train/Total)** + **tốc độ suy luận (`ms/ảnh` trên cùng GPU)**.
   - Xuất đủ **3 góc nhìn**:
     - **Bảng 1:** Toàn tập Test (`7.404` ảnh) + `% ΔfMSE` so với `0. Input Composite`.
     - **Bảng 2:** Chi tiết 4 sub-dataset (`HCOCO`, `HAdobe5k`, `HFlickr`, `Hday2night`).
     - **Bảng 3:** Chi tiết 6 nhóm độ khó (`small_low` → `large_high` theo `iHarmony4_groups_train_test/test.csv`).
4. **Quy tắc ghi nhận mô hình Pre-trained có protocol đặc thù:**
   - Bảng so sánh chính **luôn** dùng con số **Blind Test** công bằng.
   - Nếu mô hình pre-trained có protocol gốc khác biệt (như RainNet GT-BG* hay HT+ JPEG reload), đưa đối chứng vào **bảng phụ / chú thích có dấu `*`** để minh bạch tuyệt đối.

---

## 5. Quy Trình GitHub & Checklist Khi Thêm Mô Hình Mới

### 5.1. Quy ước nhánh Git & Commit
- `main`: Luôn ở trạng thái sạch, ổn định; không đẩy trực tiếp thử nghiệm dở dang.
- Nhánh phát triển:
  - `feat/<ten-mo-hinh>`: thêm kiến trúc hoặc tính năng mới (vd: `feat/diffharmony-eval`, `feat/unet-v4-swin`).
  - `fix/<mo-ta-ngan>`: sửa lỗi pipeline hoặc metric.
  - `docs/<mo-ta-ngan>`: cập nhật báo cáo, hình vẽ, bảng số liệu.
- Trước khi commit, **luôn kiểm tra** `git status` và `du -sh .` để đảm bảo **không có file ảnh dataset hoặc checkpoint nặng** lọt vào commit.

### 5.2. Checklist 5 bước khi thành viên thêm một mô hình mới
1. **Định nghĩa kiến trúc:** Thêm module tự chứa vào `src/iharmony/models/<ten_model>.py` và export trong `src/iharmony/models/__init__.py`.
2. **Đăng ký vào Factory đánh giá:** Cập nhật `load_harmonization_model()`, `infer_raw_and_out()` và `build_default_models_spec()` trong `src/iharmony/models/registry.py` — ghi rõ:
   - Dải chuẩn hóa đầu vào (`[0, 1]` hay `[-1, 1]`),
   - Yêu cầu độ chính xác (`AMP FP16` hay bắt buộc `FP32` như RAIN / Transformer),
   - Khớp `strict=True` với `state_dict` checkpoint.
3. **Huấn luyện / Nạp checkpoint:** Lưu vào `ImageHarmonization/runs/<ten_run>/` đầy đủ `best.pt`, `last.pt`, `config.json`, `history.json` (khớp đúng SHA-256 `fingerprint`).
4. **Chạy đánh giá thống nhất:** Chạy `scripts/evaluate_all.py` hoặc `notebooks/09_evaluate_all_models.ipynb` trên toàn bộ **7.404 ảnh Test** (Blind Test).
5. **Đồng bộ kết quả & tài liệu:**
   - Copy các file tổng hợp nhẹ (`*.json`, `summary_*.csv`) vào `results/evaluation_all_models/`.
   - Cập nhật bảng số liệu trong `README.md` và thêm sơ đồ + cấu hình vào `docs/architectures_and_configs.md`.
   - Mở Pull Request mô tả rõ kiến trúc, tham số, tài nguyên GPU và bảng cải thiện `fMSE / PSNR / fSSIM / fΔE76`.
