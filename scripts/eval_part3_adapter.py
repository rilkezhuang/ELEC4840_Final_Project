import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.data_io import load_brats_slice
# from src.prompts import extract_bbox_from_mask  # 我们将直接使用 np.where 提取精确紧密框
from src.eval_metrics import calculate_dice
from src.models.adapters import PromptRefinementAdapter
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")
ADAPTER_CKPT = os.path.join(PROJECT_ROOT, "weights", "pra_adapter_best.pth")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Test")

def main():
    print(f"[Info] Part III evaluation pipeline, device: {DEVICE}")
    
    # 核心修改 1：锁定随机数种子，确保学术评估的完全可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    print("[Info] Random seed locked to 42 for Deterministic A/B Testing.")

    sam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()

    adapter = PromptRefinementAdapter(embed_dim=256, roi_size=7).to(DEVICE)
    adapter.load_state_dict(torch.load(ADAPTER_CKPT, map_location=DEVICE))
    adapter.eval()

    results = []
    patient_ids = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    print(f"[Info] Evaluating {len(patient_ids)} test patients...")

    for pid in tqdm(patient_ids, desc="Adapter Evaluation"):
        patient_folder = os.path.join(TEST_DIR, pid)
        mask_path = os.path.join(patient_folder, f"{pid}_seg.nii")

        tasks = {
            "Whole_Tumor": {"img_path": os.path.join(patient_folder, f"{pid}_flair.nii"), "labels": None},
            "Tumor_Core": {"img_path": os.path.join(patient_folder, f"{pid}_t2.nii"), "labels": [1, 4]}
        }

        for task_name, task_info in tasks.items():
            if not os.path.exists(task_info["img_path"]) or not os.path.exists(mask_path):
                continue

            try:
                img_slice, gt_mask = load_brats_slice(task_info["img_path"], mask_path, slice_idx=None, target_labels=task_info["labels"])
            except Exception:
                continue

            if np.sum(gt_mask) == 0:
                continue

            original_h, original_w = img_slice.shape[:2]

            img_1024 = cv2.resize(img_slice, (1024, 1024))
            img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
            img_tensor = (img_tensor - pixel_mean) / pixel_std

            # ---------------------------------------------------------
            # 核心修改 2：非对称随机扰动 (Asymmetric Random Perturbation)
            # ---------------------------------------------------------
            # 提取完美的 GT Tight Box
            gt_y, gt_x = np.where(gt_mask > 0)
            x_min, x_max = np.min(gt_x), np.max(gt_x)
            y_min, y_max = np.min(gt_y), np.max(gt_y)

            # 在 10 到 30 像素之间，为四条边生成独立的随机外扩量
            noise_x_min = np.random.randint(10, 30)
            noise_y_min = np.random.randint(10, 30)
            noise_x_max = np.random.randint(10, 30)
            noise_y_max = np.random.randint(10, 30)

            # 生成 Loose Box，确保不越界
            loose_box = np.array([
                max(0, x_min - noise_x_min),
                max(0, y_min - noise_y_min),
                min(original_w - 1, x_max + noise_x_max),
                min(original_h - 1, y_max + noise_y_max)
            ])

            scale_x, scale_y = 1024 / original_w, 1024 / original_h
            loose_box_1024 = loose_box * np.array([scale_x, scale_y, scale_x, scale_y])
            box_tensor = torch.tensor(loose_box_1024).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # --- 通用图像特征编码 ---
                image_embedding = sam_model.image_encoder(img_tensor)
                
                # ---------------------------------------------------------
                # 核心机制 3：A/B 共享输入 (Baseline 和 Adapter 吃同一个 box_tensor)
                # ---------------------------------------------------------
                
                # === A. 基线测试 (Baseline) ===
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None, boxes=box_tensor, masks=None
                )
                low_res_baseline, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                pred_base = F.interpolate(low_res_baseline, size=(original_h, original_w), mode="bilinear", align_corners=False)
                pred_base_np = (torch.sigmoid(pred_base).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                dice_base = calculate_dice(pred_base_np, gt_mask)

                # === B. PRA 适配器测试 (Adapted) ===
                refined_box_tensor = adapter(image_embedding, box_tensor) # 传入同一个带随机噪声的 box_tensor
                sparse_adapted, dense_adapted = sam_model.prompt_encoder(
                    points=None, boxes=refined_box_tensor, masks=None
                )
                low_res_adapted, _ = sam_model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_adapted,
                    dense_prompt_embeddings=dense_adapted,
                    multimask_output=False,
                )
                pred_adapt = F.interpolate(low_res_adapted, size=(original_h, original_w), mode="bilinear", align_corners=False)
                pred_adapt_np = (torch.sigmoid(pred_adapt).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                dice_adapt = calculate_dice(pred_adapt_np, gt_mask)

            results.append({
                "Task": task_name,
                "Baseline_LooseBox": dice_base,
                "Adapted_LooseBox": dice_adapt,
                "Improvement": dice_adapt - dice_base
            })

    df = pd.DataFrame(results)
    print("\n=============================================")
    print("Part III: Prompt-Refinement Adapter final results summary")
    print("=============================================")
    summary = df.groupby("Task")[["Baseline_LooseBox", "Adapted_LooseBox", "Improvement"]].mean()
    print(summary)
    print("=============================================")
    OUTPUT_CSV_PART3 = os.path.join(PROJECT_ROOT, "outputs", "part3_adapter_results.csv")
    df.to_csv(OUTPUT_CSV_PART3, index=False)
    print(f"[Info] Results saved to: {OUTPUT_CSV_PART3}")
    print("[Info] Part III evaluation completed.")

if __name__ == "__main__":
    main()