import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.data_io import load_brats_slice
from src.prompts import extract_bbox_from_mask
from src.utils import plot_comparison
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")

TEST_NII_IMAGE = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Test", "Brats18_2013_7_1", "Brats18_2013_7_1_flair.nii")
TEST_NII_MASK = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Test", "Brats18_2013_7_1", "Brats18_2013_7_1_seg.nii")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

def main():
    print(f"[Info] 3D MRI inference pipeline, device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[Info] Loading MedSAM...")
    sam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT)
    sam_model = sam_model.to(DEVICE).eval()

    print(f"[Info] Loading NIfTI volume: {TEST_NII_IMAGE}")
    image_slice_np, gt_mask_slice_np = load_brats_slice(TEST_NII_IMAGE, TEST_NII_MASK, slice_idx=None)
    original_h, original_w = image_slice_np.shape[:2]

    if np.sum(gt_mask_slice_np) == 0:
        print("[Warning] Selected slice has no tumor annotation.")
        return

    box_prompt = extract_bbox_from_mask(gt_mask_slice_np, perturbation=2, img_shape=(original_h, original_w))
    print(f"[Info] Bounding box prompt: {box_prompt}")

    print("[Info] Preprocessing...")
    img_1024 = cv2.resize(image_slice_np, (1024, 1024))
    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
    img_tensor = (img_tensor - pixel_mean) / pixel_std

    scale_x, scale_y = 1024 / original_w, 1024 / original_h
    box_1024 = box_prompt * np.array([scale_x, scale_y, scale_x, scale_y])
    box_tensor = torch.tensor(box_1024).float().unsqueeze(0).to(DEVICE)

    print("[Info] Running MedSAM forward pass...")
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img_tensor)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None, boxes=box_tensor, masks=None
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

    save_path = os.path.join(OUTPUT_DIR, "part1_3d_brats_flair_result.png")
    print(f"[Info] Saving visualization to: {save_path}")
    plot_comparison(
        image=image_slice_np,
        gt_mask=gt_mask_slice_np,
        pred_mask=pred_mask_np,
        box_prompt=box_prompt,
        save_path=save_path
    )
    print("[Success] 3D MRI slice inference completed.")

if __name__ == "__main__":
    main()
