# Image Harmonization — Tổng hợp link paper + code + checkpoint

Ngày tạo: 2026-11-13. Folder Drive BTL: ImageHarmonization (`1oLmarz5jaqyxyRg762QeynYq4dQy-6mm`).

## 4 mô hình chính của BTL (đã có / đang làm)

| Mô hình | Paper | Code / Checkpoint | Ghi chú |
|---|---|---|---|
| U-Net (baseline) | — (mạng generic, tự train) | Đã train xong full_l1 (60 epoch, best val fMSE 0,007972) và masked_l1 đang train | Checkpoint trong `runs/unet_full_60ep_v1/` trên Drive |
| RainNet | Ling et al., CVPR 2021 — "RainNet: Region-aware Adaptive Instance Normalization for Image Harmonization" | Checkpoint chính thức `net_G.pth` (54.759.043 params) — đã có, đã chạy evaluate.py (fMSE 556,96 thang 255, PSNR 36,65) | U-Net + module RAIN |
| cGAN (DoveNet) | Cong et al., CVPR 2020 — "DoveNet: Deep Image Harmonization via Domain Verification" | Thiết kế đã chốt (G = U-Net masked L1, D = PatchGAN), CHƯA train | U-Net + discriminator |
| HT+ | Transformer backbone cho harmonization | Checkpoint pretrained đã có, đã đánh giá (fMSE 325,25 trên test 4.283 — tốt nhất) | CNN đấu transformer, độc lập với U-Net |

## Mô hình Diffusion (mở rộng / mô hình thứ 5)

### 1. IC-Light — CÓ checkpoint, chạy được ngay (khuyên dùng)
- **Paper (ICLR 2025):** https://proceedings.iclr.cc/paper_files/paper/2025/file/1cce374fec7829c24a97ae0e91cd0999-Paper-Conference.pdf
  - Tên đầy đủ: "Scaling In-The-Wild Training for Diffusion-Based Illumination Harmonization and Editing by Imposing Consistent Light Transport" (Zhang, Rao, Agrawala).
- **GitHub:** https://github.com/lllyasviel/IC-Light
- **Checkpoint (HuggingFace):** https://huggingface.co/lllyasviel/ic-light
  - Lấy file `iclight_sd15_fbc.safetensors` (text + foreground + background) cho harmonization.
  - Hai file khác: `iclight_sd15_fc.safetensors`, `iclight_sd15_fcon.safetensors`.
- **Demo thử nhanh:** https://huggingface.co/spaces/lllyasviel/IC-Light
- **Ghi chú:** Model relighting (đổi ánh sáng) trên SD 1.5, không phải harmonization chuẩn iHarmony4. Dùng bản fbc (background-conditioned) để chỉnh sáng foreground theo nền. Chạy được trên T4.

### 2. DiffHarmony — CÓ checkpoint, latent diffusion cho harmonization
- **Paper:** https://arxiv.org/pdf/2404.06139
  - "DiffHarmony: Latent Diffusion Model Meets Image Harmonization" (Zhou, Feng, Wang, ICMR 2024).
- **GitHub:** https://github.com/nicecv/DiffHarmony
- **Checkpoint (Google Drive):** https://drive.google.com/file/d/1rezNdcuZbwejbC9rH9S1SUuaWTGTz_wG/view?usp=drive_link
  - Ngoài ra có Baidu Pan (code: `aqqd`) trong README của repo.
- **Ghi chú:** Adapt Stable Diffusion có sẵn cho harmonization, đánh giá trên iHarmony4. Repo có sẵn script inference (`scripts/inference.sh`) và evaluation (`scripts/evaluate.sh`). Có cả DiffHarmony++ (ACM MM 2024).

### 3. Image Harmonization with Diffusion Model (Li et al., 2023) — bài diffusion đầu tiên, KHÔNG có checkpoint public
- **Paper (arXiv):** https://arxiv.org/pdf/2306.10441
  - "Image Harmonization with Diffusion Model" (Jiajie Li, Jian Wang, Chen Wang, Jinjun Xiong; Univ. at Buffalo, Snap Inc., IBM Research).
  - Dùng DDPM train từ đầu và LDM + ControlNet fine-tune Stable Diffusion. Đánh giá trên iHarmony4.
- **Checkpoint:** Không tìm thấy link public (chỉ có paper). Muốn dùng thì xem DiffHarmony (mục 2) thay thế.

### 4. Training-free (không cần checkpoint riêng, dùng SD có sẵn)
- **FreePIH** — Paper: https://ira.lib.polyu.edu.hk/bitstream/10397/116815/1/3664647.3680780.pdf
  - "FreePIH: Training-Free Painterly Image Harmonization with Diffusion Model" (MM 2024). Dùng SD v1.5 có sẵn.
- **Diff-Harmonization (Zero-Shot)** — GitHub: https://github.com/WindVChen/Diff-Harmonization ; Paper: https://arxiv.org/html/2307.08182v2
  - "Zero-Shot Image Harmonization with Generative Model Prior" (TOG). Dùng VLM + diffusion có sẵn, không cần train.
- **FreeCompose** — GitHub: https://github.com/aim-uofa/FreeCompose ; Paper: https://pdfs.assets.alphaxiv.org/2407.04947v1.pdf
  - "FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior" (ECCV 2024). Dùng SD V2.1 có sẵn.
- **TF-GPH** — GitHub: https://github.com/BlueDyee/TF-GPH
  - "Training-and-Prompt-Free General Painterly Harmonization" (AAAI 2025). Dùng SD có sẵn.
- **ControlCom** — GitHub: https://github.com/bcmi/ControlCom-Image-Composition
  - "ControlCom: Controllable Image Composition using Diffusion Model". Unify 4 task (blending, harmonization, view synthesis, generative composition) trong một diffusion model.

## Ghi chú sử dụng cho BTL
- Dùng checkpoint người ta là cách làm chuẩn (RainNet và HT+ trong BTL cũng dùng checkpoint chính thức, không train lại). Nhớ trích dẫn nguồn và đánh giá trên cùng test set (HCOCO 4.283 ảnh) với cùng metric (fMSE, PSNR) thì mới so sánh công bằng.
- IC-Light và DiffHarmony là 2 mô hình diffusion thực tế nhất để thử làm mô hình thứ 5. IC-Light là relighting (cần tự chạy trên HCOCO rồi đo), DiffHarmony là harmonization chuẩn iHarmony4 (có sẵn script inference/evaluate).
- Không gắn trực tiếp U-Net đã train vào UNet của Stable Diffusion được (pixel vs latent, thiếu timestep/text conditioning, 7,85M vs 860M params). Muốn kết hợp thì đi đường pipeline (U-Net sửa màu → SD refine) hoặc ControlNet fine-tune.
