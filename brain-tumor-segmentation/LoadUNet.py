import torch
import UNet
import segmentation_models_pytorch as smp

# Recreate the same model architecture
def load_model():
    model = UNet.get_model()

    if model is None:
        print("Error: Model not loaded properly!")
    else:
        print("Model loaded successfully!")

    # Load saved weights
    model.load_state_dict(torch.load("unet_brain_segmentation.pth", map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    # Set to evaluation mode
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model.eval() 
    print("Model loaded and ready for inference.")
    return model
