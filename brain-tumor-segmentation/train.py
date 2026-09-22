import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import os
import cv2
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import UNet

# ==== Custom Dataset ====
class BrainTumorDataset(Dataset):
    print("Loading dataset...")
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()  # Add channel dim for mask

# ==== Transformations ====
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
])

# ==== Load Data ====
train_dataset = BrainTumorDataset("dataset/images", "dataset/masks", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# ==== Load Predefined UNet ====
model = UNet.get_model()

# ==== Move to GPU ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# ==== Loss and Optimizer ====
loss_fn = smp.losses.DiceLoss(mode='binary')  # or BCEWithLogitsLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ==== Training Loop ====
num_epochs = 30
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    print(f"Epoch {epoch+1}/{num_epochs}")
    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
print("Training complete.")

# Save model weights
torch.save(model.state_dict(), "unet_brain_segmentation.pth")
print("Model saved to unet_brain_segmentation.pth")
