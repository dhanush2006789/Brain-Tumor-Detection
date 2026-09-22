import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import LoadSegNet
import os

model = LoadSegNet.load_model()
if model is None:
    raise ValueError("Error: Model2 not loaded properly!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
])

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    augmented = transform(image=img)
    img_tensor = augmented["image"].unsqueeze(0).to(device)  # Add batch dim
    return img_tensor, img

def predict_segnet(model, image_path, output_path):
    input_tensor, original_img = preprocess_image(image_path)

    with torch.no_grad():
        pred_mask = model(input_tensor)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = (pred_mask > 0.5).float().cpu().squeeze().numpy()

    # Resize prediction to original size
    pred_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]))
    pred_resized_uint8 = (pred_resized * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(pred_resized_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale image to BGR
    display_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box

    # Create red mask overlay
    colored_mask = np.zeros_like(display_img)
    colored_mask[:, :, 2] = pred_resized_uint8  # Red channel

    # Blend mask with the original image
    overlayed_img = cv2.addWeighted(display_img, 1.0, colored_mask, 0.4, 0)

    # Resize the output image to 70% of its original size
    height, width = overlayed_img.shape[:2]
    new_width = int(width * 0.7)
    new_height = int(height * 0.7)
    resized_img = cv2.resize(overlayed_img, (new_width, new_height), interpolation=cv2.INTER_AREA)


    # Save the image
    cv2.imwrite(output_path, overlayed_img)
    print(f"Saved: {output_path}")

