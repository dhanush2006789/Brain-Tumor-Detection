import segmentation_models_pytorch as smp
def get_model():
    
    print("Loading UNet model...")
    
    # Define the model
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=1,                  
        classes=1                      
    )

    return model
