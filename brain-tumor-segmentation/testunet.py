import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import LoadUNet
import os

model = LoadUNet.load_model()

if model is None:
    raise ValueError("Error: Model1 not loaded properly!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
])

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    augmented = transform(image=img)
    img_tensor = augmented["image"].unsqueeze(0).to(device)  # Add batch dimension
    return img_tensor, img

def predict_unet(model, image_path, output_path):
    input_tensor, original_img = preprocess_image(image_path)

    with torch.no_grad():
        pred_mask = model(input_tensor)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float().cpu().squeeze().numpy()

    # Resize prediction to original size
    pred_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]))
    pred_resized_uint8 = (pred_resized * 255).astype(np.uint8)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(pred_resized_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale image to BGR for visualization
    display_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # If there are contours, get the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Create an empty mask and draw only the largest contour on it
        largest_mask = np.zeros_like(pred_resized_uint8)
        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Draw bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box
    else:
        largest_mask = np.zeros_like(pred_resized_uint8)

    # Overlay mask onto image
    colored_mask = np.zeros_like(display_img)
    colored_mask[:, :, 2] = largest_mask  # Red channel

    overlayed_img = cv2.addWeighted(display_img, 1.0, colored_mask, 0.4, 0)

    # Resize the output image to 70% of its original size
    height, width = overlayed_img.shape[:2]
    new_width = int(width * 0.7)
    new_height = int(height * 0.7)
    resized_img = cv2.resize(overlayed_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the image
    cv2.imwrite(output_path, overlayed_img)
    print(f"Saved: {output_path}")

