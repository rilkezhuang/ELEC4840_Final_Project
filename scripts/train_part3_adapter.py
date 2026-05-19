import os
import sys
import csv
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.data_io import load_brats_slice
from src.prompts import extract_bbox_from_mask
from src.models.adapters import PromptRefinementAdapter
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
MEDSAM_CKPT = os.path.join(PROJECT_ROOT, "weights", "sam_vit_b_01ec64.pth")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "BraTS-2018", "Train")
ADAPTER_SAVE_PATH = os.path.join(PROJECT_ROOT, "weights", "pra_adapter_best.pth")
LOSS_LOG_PATH = os.path.join(PROJECT_ROOT, "outputs", "part3_training_loss.csv")

EPOCHS = 10
LEARNING_RATE = 1e-3

def main():
    print(f"[Info] Part III adapter training, device: {DEVICE}")
    sam_model = sam_model_registry['vit_b'](checkpoint=MEDSAM_CKPT).to(DEVICE).eval()

    adapter = PromptRefinementAdapter(embed_dim=256, roi_size=7).to(DEVICE)
    adapter.train()

    optimizer = optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)

    patient_ids = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    best_loss = float('inf')
    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        random.shuffle(patient_ids)
        pbar = tqdm(patient_ids, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for pid in pbar:
            patient_folder = os.path.join(TRAIN_DIR, pid)
            img_path = os.path.join(patient_folder, f"{pid}_flair.nii")
            mask_path = os.path.join(patient_folder, f"{pid}_seg.nii")

            if not os.path.exists(img_path) or not os.path.exists(mask_path): continue
            try:
                img_slice, gt_mask = load_brats_slice(img_path, mask_path, slice_idx=None)
            except: continue
            if np.sum(gt_mask) == 0: continue

            original_h, original_w = img_slice.shape[:2]
            scale_x, scale_y = 1024 / original_w, 1024 / original_h

            tight_box = extract_bbox_from_mask(gt_mask, perturbation=0, img_shape=(original_h, original_w))
            target_box_1024 = torch.tensor(tight_box * np.array([scale_x, scale_y, scale_x, scale_y])).float().unsqueeze(0).to(DEVICE)

            noise = random.randint(5, 20)
            noisy_box = extract_bbox_from_mask(gt_mask, perturbation=noise, img_shape=(original_h, original_w))
            noisy_box_1024 = torch.tensor(noisy_box * np.array([scale_x, scale_y, scale_x, scale_y])).float().unsqueeze(0).to(DEVICE)

            img_1024 = cv2.resize(img_slice, (1024, 1024))
            img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(DEVICE)
            pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(DEVICE)
            img_tensor = (img_tensor - pixel_mean) / pixel_std

            optimizer.zero_grad()
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(img_tensor)

            refined_box = adapter(image_embedding, noisy_box_1024)
            loss = F.smooth_l1_loss(refined_box, target_box_1024)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Box_Loss': f"{loss.item():.2f}"})

        avg_loss = epoch_loss / len(patient_ids)
        loss_history.append({"epoch": epoch + 1, "avg_box_loss": avg_loss})
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Avg Box Loss: {avg_loss:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(adapter.state_dict(), ADAPTER_SAVE_PATH)
            print("[Info] Checkpoint saved.")

    os.makedirs(os.path.dirname(LOSS_LOG_PATH), exist_ok=True)
    with open(LOSS_LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "avg_box_loss"])
        writer.writeheader()
        writer.writerows(loss_history)
    print(f"[Info] Training loss log saved to: {LOSS_LOG_PATH}")
    print("\n[Success] Adapter training completed.")

if __name__ == "__main__":
    main()
