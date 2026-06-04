# model_files/model_architecture.py

import torch
import torch.nn as nn
import os
from torchvision.models.video import r3d_18, R3D_18_Weights 

def load_detector_model(model_path, device):
    """
    Loads a pretrained R3D-18 model and modifies its input layer 
    to accept 4 channels (RGB + Mask).
    """
    # 1. Load the R3D-18 model with standard ImageNet weights
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)

    # 2. Modify the first convolutional layer (conv1) to accept 4 channels
    original_conv1 = model.stem[0]
    
    new_conv1 = nn.Conv3d(
        in_channels=4, # CHANGED FROM 3 TO 4
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias
    )
    
    # Transfer weights from the original 3 channels to the new 4-channel layer
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :, :] = original_conv1.weight.clone()
        # Initialize the 4th channel (index 3, the mask channel) weights to zero
        new_conv1.weight[:, 3, :, :, :].zero_()

    # Replace the original conv1 layer
    model.stem[0] = new_conv1

    # 3. Modify the final classification layer (fc) to output 2 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 

    # 4. Load the trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model weights not found at {model_path}. Using uninitialized weights.")

    model = model.to(device)
    model.eval()
    return model