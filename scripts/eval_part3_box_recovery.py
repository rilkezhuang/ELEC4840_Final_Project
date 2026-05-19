import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.data_io import load_brats_slice
from src.prompts import extract_bbox_from_mask
from src.eval_metrics import calculate_dice
from src.box_metrics import (
    calculate_box_iou,
    calculate_box_l1,
    scale_box_to_1024,
    unscale_box_from_1024,
)
from src.models.adapters import PromptRefinementAdapter
from segment_anything import sam_model_registry

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")
ADAPTER_CKPT = os.path.join(PROJECT_ROOT, "weights", "pra_adapter_best.pth")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Test")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "outputs", "part3_box_recovery.csv")

BENCHMARK_LOOSE_PERTURBATION = 10


def run_medsam_with_box(sam_model, image_embedding, box_1024, original_h, original_w):
    box_tensor = torch.tensor(box_1024).float().unsqueeze(0).to(DEVICE)
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None, boxes=box_tensor, masks=None
    )
    low_res, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    pred = F.interpolate(
        low_res, size=(original_h, original_w), mode="bilinear", align_corners=False
    )
    return (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def main():
    print(f"[Info] Part III box recovery eval (benchmark loose box +{BENCHMARK_LOOSE_PERTURBATION}px)")
    print(f"[Info] Device: {DEVICE}")

    sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()
    adapter = PromptRefinementAdapter(embed_dim=256, roi_size=7).to(DEVICE)
    adapter.load_state_dict(torch.load(ADAPTER_CKPT, map_location=DEVICE))
    adapter.eval()

    results = []
    patient_ids = sorted(
        d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))
    )

    for pid in tqdm(patient_ids, desc="Box Recovery"):
        patient_folder = os.path.join(TEST_DIR, pid)
        mask_path = os.path.join(patient_folder, f"{pid}_seg.nii")

        tasks = {
            "Whole_Tumor": {
                "img_path": os.path.join(patient_folder, f"{pid}_flair.nii"),
                "labels": None,
            },
            "Tumor_Core": {
                "img_path": os.path.join(patient_folder, f"{pid}_t2.nii"),
                "labels": [1, 4],
            },
        }

        for task_name, task_info in tasks.items():
            if not os.path.exists(task_info["img_path"]) or not os.path.exists(mask_path):
                continue

            try:
                img_slice, gt_mask = load_brats_slice(
                    task_info["img_path"],
                    mask_path,
                    slice_idx=None,
                    target_labels=task_info["labels"],
                )
            except Exception:
                continue

            if np.sum(gt_mask) == 0:
                continue

            original_h, original_w = img_slice.shape[:2]
            tight_box = extract_bbox_from_mask(
                gt_mask, perturbation=0, img_shape=(original_h, original_w)
            )
            loose_box = extract_bbox_from_mask(
                gt_mask,
                perturbation=BENCHMARK_LOOSE_PERTURBATION,
                img_shape=(original_h, original_w),
            )
            loose_box_1024 = scale_box_to_1024(loose_box, original_w, original_h)
            tight_box_1024 = scale_box_to_1024(tight_box, original_w, original_h)

            img_1024 = cv2.resize(img_slice, (1024, 1024))
            img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
            img_tensor = (img_tensor - pixel_mean) / pixel_std

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(img_tensor)
                box_tensor = torch.tensor(loose_box_1024).float().unsqueeze(0).to(DEVICE)
                refined_box_tensor = adapter(image_embedding, box_tensor)
                refined_box_1024 = refined_box_tensor.squeeze(0).cpu().numpy()
                refined_box = unscale_box_from_1024(
                    refined_box_1024, original_w, original_h
                )

                pred_loose = run_medsam_with_box(
                    sam_model, image_embedding, loose_box_1024, original_h, original_w
                )
                pred_refined = run_medsam_with_box(
                    sam_model, image_embedding, refined_box_1024, original_h, original_w
                )
                pred_tight = run_medsam_with_box(
                    sam_model, image_embedding, tight_box_1024, original_h, original_w
                )

            dice_loose = calculate_dice(pred_loose, gt_mask)
            dice_refined = calculate_dice(pred_refined, gt_mask)
            dice_tight = calculate_dice(pred_tight, gt_mask)

            iou_loose_tight = calculate_box_iou(loose_box, tight_box)
            iou_refined_tight = calculate_box_iou(refined_box, tight_box)
            l1_loose_tight = calculate_box_l1(loose_box, tight_box)
            l1_refined_tight = calculate_box_l1(refined_box, tight_box)

            results.append(
                {
                    "Patient_ID": pid,
                    "Task": task_name,
                    "Loose_Box_IoU_to_Tight": iou_loose_tight,
                    "Refined_Box_IoU_to_Tight": iou_refined_tight,
                    "Box_IoU_Recovery": iou_refined_tight - iou_loose_tight,
                    "Loose_Box_L1_to_Tight": l1_loose_tight,
                    "Refined_Box_L1_to_Tight": l1_refined_tight,
                    "Dice_Loose_Benchmark": dice_loose,
                    "Dice_PRA_Refined": dice_refined,
                    "Dice_Tight_Oracle": dice_tight,
                    "Dice_Recovery": dice_refined - dice_loose,
                    "Gap_to_Tight_Remaining": dice_tight - dice_refined,
                }
            )

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=============================================")
    print("Part III: Recovery toward Tight Box (benchmark loose +10px)")
    print("=============================================")

    box_summary = df.groupby("Task")[
        [
            "Loose_Box_IoU_to_Tight",
            "Refined_Box_IoU_to_Tight",
            "Box_IoU_Recovery",
            "Loose_Box_L1_to_Tight",
            "Refined_Box_L1_to_Tight",
        ]
    ].mean()
    print("\n--- Box geometry vs. GT tight box ---")
    print(box_summary.round(4).to_string())

    dice_summary = df.groupby("Task")[
        [
            "Dice_Loose_Benchmark",
            "Dice_PRA_Refined",
            "Dice_Tight_Oracle",
            "Dice_Recovery",
            "Gap_to_Tight_Remaining",
        ]
    ].mean()
    print("\n--- Segmentation Dice (same benchmark loose input) ---")
    print(dice_summary.round(3).to_string())

    improved = (df["Box_IoU_Recovery"] > 0).mean() * 100
    print(f"\n[Info] Cases with higher box IoU to tight after PRA: {improved:.1f}%")
    print(f"[Info] Per-case results saved to: {OUTPUT_CSV}")
    print("[Info] Box recovery evaluation completed.")


if __name__ == "__main__":
    main()
