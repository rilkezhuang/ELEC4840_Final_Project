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
from src.prompts import extract_bbox_from_mask, sample_points_from_mask
from src.eval_metrics import calculate_dice
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Test")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "brats_benchmark_results.csv")

PROMPT_SETTINGS = ["tight_box", "loose_box", "1_point", "5_points"]

def run_medsam_decoder(sam_model, image_embedding, original_h, original_w, box_1024=None, points_1024=None):
    """Run mask decoder reusing a precomputed image embedding."""
    with torch.no_grad():
        box_tensor = None
        if box_1024 is not None:
            box_tensor = torch.tensor(box_1024).float().unsqueeze(0).to(DEVICE)

        point_coords_tensor, point_labels_tensor = None, None
        if points_1024 is not None:
            coords, labels = points_1024
            point_coords_tensor = torch.tensor(coords).unsqueeze(0).float().to(DEVICE)
            point_labels_tensor = torch.tensor(labels).unsqueeze(0).to(DEVICE)

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=(point_coords_tensor, point_labels_tensor) if points_1024 is not None else None,
            boxes=box_tensor,
            masks=None,
        )

        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_mask_tensor = F.interpolate(low_res_masks, size=(original_h, original_w), mode="bilinear", align_corners=False)
        pred_mask_np = (torch.sigmoid(pred_mask_tensor).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    return pred_mask_np

def main():
    np.random.seed(42)
    print(f"[Info] Part II benchmark, device: {DEVICE}")
    sam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()

    results = []
    patient_ids = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]

    print(f"[Info] Found {len(patient_ids)} patients, starting evaluation...")

    for pid in tqdm(patient_ids, desc="BraTS-2018 Test Set Progress"):
        patient_folder = os.path.join(TEST_DIR, pid)
        mask_path = os.path.join(patient_folder, f"{pid}_seg.nii")

        tasks = {
            "Whole_Tumor": {"img_path": os.path.join(patient_folder, f"{pid}_flair.nii"), "labels": None},
            "Tumor_Core": {"img_path": os.path.join(patient_folder, f"{pid}_t2.nii"), "labels": [1, 4]}
        }

        for task_name, task_info in tasks.items():
            if not os.path.exists(task_info["img_path"]) or not os.path.exists(mask_path):
                continue

            img_slice, gt_mask = load_brats_slice(task_info["img_path"], mask_path, slice_idx=None, target_labels=task_info["labels"])
            original_h, original_w = img_slice.shape[:2]

            if np.sum(gt_mask) == 0:
                continue

            img_1024 = cv2.resize(img_slice, (1024, 1024))
            img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
            img_tensor = (img_tensor - pixel_mean) / pixel_std

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(img_tensor)

            scale_x, scale_y = 1024 / original_w, 1024 / original_h

            tight_box = extract_bbox_from_mask(gt_mask, perturbation=0, img_shape=(original_h, original_w))
            tight_box_1024 = tight_box * np.array([scale_x, scale_y, scale_x, scale_y])
            pred_tight = run_medsam_decoder(sam_model, image_embedding, original_h, original_w, box_1024=tight_box_1024)
            dice_tight = calculate_dice(pred_tight, gt_mask)

            loose_box = extract_bbox_from_mask(gt_mask, perturbation=10, img_shape=(original_h, original_w))
            loose_box_1024 = loose_box * np.array([scale_x, scale_y, scale_x, scale_y])
            pred_loose = run_medsam_decoder(sam_model, image_embedding, original_h, original_w, box_1024=loose_box_1024)
            dice_loose = calculate_dice(pred_loose, gt_mask)

            coords_1, labels_1 = sample_points_from_mask(gt_mask, num_points=1)
            coords_1_1024 = coords_1 * np.array([scale_x, scale_y])
            pred_1pt = run_medsam_decoder(sam_model, image_embedding, original_h, original_w, points_1024=(coords_1_1024, labels_1))
            dice_1pt = calculate_dice(pred_1pt, gt_mask)

            coords_5, labels_5 = sample_points_from_mask(gt_mask, num_points=5)
            coords_5_1024 = coords_5 * np.array([scale_x, scale_y])
            pred_5pt = run_medsam_decoder(sam_model, image_embedding, original_h, original_w, points_1024=(coords_5_1024, labels_5))
            dice_5pt = calculate_dice(pred_5pt, gt_mask)

            results.append({
                "Patient_ID": pid,
                "Task": task_name,
                "Tight_Box_Dice": dice_tight,
                "Loose_Box_Dice": dice_loose,
                "1_Point_Dice": dice_1pt,
                "5_Points_Dice": dice_5pt
            })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n[Success] Benchmark completed.")
    print(f"[Info] Results saved to: {OUTPUT_CSV}")

    print("\n--- Mean Dice scores ---")
    summary = df.groupby("Task")[["Tight_Box_Dice", "Loose_Box_Dice", "1_Point_Dice", "5_Points_Dice"]].mean()
    print(summary)

if __name__ == "__main__":
    main()
