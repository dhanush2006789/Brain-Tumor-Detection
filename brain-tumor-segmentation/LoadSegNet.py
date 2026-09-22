import torch
import SegNet as sn

def load_model():
    print("Loading SegNet model...")
    model = sn.SegNet()
    if model is None:
        print("Error: Model not loaded properly!")
    else:
        print("Model loaded successfully!")
    
    model.load_state_dict(torch.load("segnet_brain_segmentation.pth", map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    print("Model loaded and ready for inference.")
    return model
    