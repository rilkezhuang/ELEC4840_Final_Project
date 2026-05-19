"""
Part I: MedSAM prompt-based inference on all five TestBench datasets.
For each dataset, runs bounding-box and point prompts and saves qualitative figures.
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

from src.data_io import load_2d_data, load_brats_slice, load_video_frame
from src.prompts import extract_bbox_from_mask, sample_points_from_mask
from src.utils import plot_comparison
from segment_anything import sam_model_registry

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")
TESTBENCH_DIR = os.path.join(PROJECT_ROOT, "data", "TestBench")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")

RANDOM_SEED = 42
NUM_POINT_PROMPTS = 1

# One example per TestBench dataset (image/mask paths relative to dataset folder)
DATASETS = [
    {
        "name": "covidquex",
        "modality": "2D X-ray",
        "type": "2d",
        "image": os.path.join("covidquex", "images", "0010.png"),
        "mask": os.path.join("covidquex", "masks", "0010.png"),
    },
    {
        "name": "polypgen",
        "modality": "2D endoscopy",
        "type": "2d",
        "image": os.path.join("polypgen", "images", "0040.png"),
        "mask": os.path.join("polypgen", "masks", "0040.png"),
    },
    {
        "name": "skincancer",
        "modality": "2D dermoscopy",
        "type": "2d",
        "image": os.path.join("skincancer", "images", "0000.png"),
        "mask": os.path.join("skincancer", "masks", "0000.png"),
    },
    {
        "name": "liverseg",
        "modality": "3D CT (slice)",
        "type": "nii",
        "image": os.path.join("liverseg", "scans", "liver-orig001.nii.gz"),
        "mask": os.path.join("liverseg", "labels", "liver-seg001.nii.gz"),
    },
    {
        "name": "echodynamic",
        "modality": "Video echo (frame)",
        "type": "video",
        "video": os.path.join("echodynamic", "videos", "0X171B390B95FD0FAE.avi"),
        "mask": os.path.join("echodynamic", "masks", "0X171B390B95FD0FAE_77.png"),
        "frame_idx": 77,
    },
]


def load_sample(entry: dict):
    if entry["type"] == "2d":
        img_path = os.path.join(TESTBENCH_DIR, entry["image"])
        mask_path = os.path.join(TESTBENCH_DIR, entry["mask"])
        return load_2d_data(img_path, mask_path)
    if entry["type"] == "nii":
        img_path = os.path.join(TESTBENCH_DIR, entry["image"])
        mask_path = os.path.join(TESTBENCH_DIR, entry["mask"])
        return load_brats_slice(img_path, mask_path, slice_idx=None)
    if entry["type"] == "video":
        video_path = os.path.join(TESTBENCH_DIR, entry["video"])
        mask_path = os.path.join(TESTBENCH_DIR, entry["mask"])
        return load_video_frame(video_path, mask_path, frame_idx=entry.get("frame_idx", 0))
    raise ValueError(f"Unknown dataset type: {entry['type']}")


def medsam_predict(sam_model, image_np, gt_mask, prompt_type: str):
    h, w = image_np.shape[:2]
    scale = np.array([1024 / w, 1024 / h, 1024 / w, 1024 / h])

    box_prompt = None
    point_coords = None
    point_labels = None
    box_tensor = None
    point_coords_tensor = None
    point_labels_tensor = None

    if prompt_type == "box":
        box_prompt = extract_bbox_from_mask(gt_mask, perturbation=0, img_shape=(h, w))
        box_1024 = box_prompt * scale
        box_tensor = torch.tensor(box_1024).float().unsqueeze(0).to(DEVICE)
    elif prompt_type == "point":
        coords, labels = sample_points_from_mask(gt_mask, num_points=NUM_POINT_PROMPTS)
        point_coords = coords
        point_labels = labels
        coords_1024 = coords * np.array([scale[0], scale[1]])
        point_coords_tensor = torch.tensor(coords_1024).unsqueeze(0).float().to(DEVICE)
        point_labels_tensor = torch.tensor(labels).unsqueeze(0).to(DEVICE)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    img_1024 = cv2.resize(image_np, (1024, 1024))
    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
    img_tensor = (img_tensor - pixel_mean) / pixel_std

    with torch.no_grad():
        image_embedding = sam_model.image_encoder(img_tensor)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=(point_coords_tensor, point_labels_tensor) if prompt_type == "point" else None,
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
        pred_mask_tensor = F.interpolate(
            low_res_masks, size=(h, w), mode="bilinear", align_corners=False
        )
        pred_mask = (torch.sigmoid(pred_mask_tensor).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    return pred_mask, box_prompt, point_coords, point_labels


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Info] Part I TestBench pipeline, device: {DEVICE}")

    sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()

    manifest = []
    for entry in DATASETS:
        name = entry["name"]
        print(f"\n[Info] Dataset: {name} ({entry['modality']})")

        try:
            image_np, gt_mask_np = load_sample(entry)
        except Exception as e:
            print(f"[Error] Failed to load {name}: {e}")
            continue

        if np.sum(gt_mask_np) == 0:
            print(f"[Warning] No foreground in mask for {name}, skipping.")
            continue

        for prompt_type in ("box", "point"):
            print(f"  [Info] Running {prompt_type} prompt...")
            pred_mask, box_prompt, point_coords, point_labels = medsam_predict(
                sam_model, image_np, gt_mask_np, prompt_type
            )
            out_name = f"part1_{name}_{prompt_type}.png"
            save_path = os.path.join(OUTPUT_DIR, out_name)
            plot_comparison(
                image=image_np,
                gt_mask=gt_mask_np,
                pred_mask=pred_mask,
                box_prompt=box_prompt if prompt_type == "box" else None,
                point_coords=point_coords if prompt_type == "point" else None,
                point_labels=point_labels if prompt_type == "point" else None,
                save_path=save_path,
            )
            manifest.append(
                {
                    "dataset": name,
                    "modality": entry["modality"],
                    "prompt": prompt_type,
                    "path": save_path,
                }
            )
            print(f"  [Info] Saved: {save_path}")

    print("\n[Success] Part I TestBench completed.")
    print(f"[Info] Generated {len(manifest)} figures in {OUTPUT_DIR}")
    for m in manifest:
        print(f"  - {m['dataset']} | {m['prompt']} | {os.path.basename(m['path'])}")


if __name__ == "__main__":
    main()
