"""
Part II: Qualitative segmentation examples (tight vs loose box; 1 vs 5 points).
Saves two multi-panel figures for one representative BraTS-2018 test case.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.data_io import load_brats_slice
from src.prompts import extract_bbox_from_mask, sample_points_from_mask
from src.utils import show_mask, show_box, show_points
from segment_anything import sam_model_registry

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

VIS_PATIENT = "Brats18_2013_7_1"
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Test")
RANDOM_SEED = 42


def predict(sam_model, emb, h, w, box_1024=None, points_1024=None):
    box_t = None
    pt_coords_t, pt_labels_t = None, None
    if box_1024 is not None:
        box_t = torch.tensor(box_1024).float().unsqueeze(0).to(DEVICE)
    if points_1024 is not None:
        coords, labels = points_1024
        pt_coords_t = torch.tensor(coords).unsqueeze(0).float().to(DEVICE)
        pt_labels_t = torch.tensor(labels).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sparse, dense = sam_model.prompt_encoder(
            points=(pt_coords_t, pt_labels_t) if points_1024 else None,
            boxes=box_t,
            masks=None,
        )
        low_res, _ = sam_model.mask_decoder(
            image_embeddings=emb,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        pred = F.interpolate(low_res, size=(h, w), mode="bilinear", align_corners=False)
    return (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def main():
    import matplotlib.pyplot as plt

    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    folder = os.path.join(TEST_DIR, VIS_PATIENT)
    img_path = os.path.join(folder, f"{VIS_PATIENT}_flair.nii")
    mask_path = os.path.join(folder, f"{VIS_PATIENT}_seg.nii")

    print(f"[Info] Part II qualitative examples, patient={VIS_PATIENT}, device={DEVICE}")
    sam = sam_model_registry["vit_b"](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()

    img_np, gt = load_brats_slice(img_path, mask_path, slice_idx=None, target_labels=None)
    h, w = img_np.shape[:2]
    scale = np.array([1024 / w, 1024 / h, 1024 / w, 1024 / h])

    img_1024 = cv2.resize(img_np, (1024, 1024))
    img_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
    img_t = (img_t - mean) / std
    with torch.no_grad():
        emb = sam.image_encoder(img_t)

    tight_box = extract_bbox_from_mask(gt, 0, (h, w))
    loose_box = extract_bbox_from_mask(gt, 10, (h, w))
    tight_1024 = tight_box * scale
    loose_1024 = loose_box * scale

    pred_tight = predict(sam, emb, h, w, box_1024=tight_1024)
    pred_loose = predict(sam, emb, h, w, box_1024=loose_1024)

    coords_1, labels_1 = sample_points_from_mask(gt, 1)
    coords_5, labels_5 = sample_points_from_mask(gt, 5)
    pred_1 = predict(sam, emb, h, w, points_1024=(coords_1 * scale[:2], labels_1))
    pred_5 = predict(sam, emb, h, w, points_1024=(coords_5 * scale[:2], labels_5))

    # Figure 1: tight vs loose box
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["Tight Box Prompt", "Loose Box Prompt", "Ground Truth", "Tight Box Pred", "Loose Box Pred"]
    for ax, title in zip(axes, titles):
        ax.imshow(img_np)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    show_box(tight_box, axes[0], edgecolor="#3498db", lw=2)
    show_box(loose_box, axes[1], edgecolor="#e67e22", lw=2)
    show_mask(gt, axes[2], color_rgba=(0.2, 0.8, 0.2, 0.55))
    show_mask(pred_tight, axes[3], color_rgba=(0.1, 0.45, 0.9, 0.55))
    show_mask(pred_loose, axes[4], color_rgba=(0.9, 0.35, 0.2, 0.55))
    fig.suptitle(f"Part II: Tight vs. Loose Box (FLAIR / Whole Tumor) — {VIS_PATIENT}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_box = os.path.join(OUTPUT_DIR, "part2_qualitative_tight_vs_loose.png")
    plt.savefig(path_box, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Info] Saved: {path_box}")

    # Figure 2: 1 vs 5 points
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["1 Point Prompt", "5 Points Prompt", "Ground Truth", "1 Point Pred", "5 Points Pred"]
    for ax, title in zip(axes, titles):
        ax.imshow(img_np)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    show_points(coords_1, labels_1, axes[0])
    show_points(coords_5, labels_5, axes[1])
    show_mask(gt, axes[2], color_rgba=(0.2, 0.8, 0.2, 0.55))
    show_mask(pred_1, axes[3], color_rgba=(0.55, 0.35, 0.85, 0.55))
    show_mask(pred_5, axes[4], color_rgba=(0.1, 0.45, 0.9, 0.55))
    fig.suptitle(f"Part II: 1 vs. 5 Points (FLAIR / Whole Tumor) — {VIS_PATIENT}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_pt = os.path.join(OUTPUT_DIR, "part2_qualitative_1pt_vs_5pt.png")
    plt.savefig(path_pt, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Info] Saved: {path_pt}")
    print("[Success] Part II qualitative figures completed.")


if __name__ == "__main__":
    main()
