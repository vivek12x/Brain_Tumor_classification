# gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch import nn
from PIL import Image
from captum.attr import LayerGradCam, LayerAttribution
import inspect

# helper to find last conv2d layer if user does not provide one
def find_last_conv(module):
    last = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found in model")
    return last

class GradCAMWrapper:
    def __init__(self, model, target_layer=None, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        if target_layer is None:
            self.target_layer = find_last_conv(model)
        else:
            self.target_layer = target_layer
        self.lgc = LayerGradCam(self.model, self.target_layer)

        # standard transform used in this repo
        self.transform = transforms.Compose([
            transforms.Resize((300,300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor expected to be a float tensor shaped 1xCxxHxxW already normalized.
        returns cam numpy H x W in range 0..1 before upsample if small
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        if target_class is None:
            with torch.no_grad():
                out = self.model(input_tensor)
                target_class = int(out.argmax().item())
        # LayerGradCam returns attribution with same spatial size as layer output
        attr = self.lgc.attribute(input_tensor, target=target_class)
        # convert to numpy and upsample to input size
        attr = attr.detach().cpu().numpy()[0,0]
        # relu
        attr = np.maximum(attr, 0)
        # normalize
        if attr.max() > 0:
            attr = attr / (attr.max() + 1e-9)
        # upsample to input H W
        import cv2
        H = input_tensor.shape[-2]
        W = input_tensor.shape[-1]
        attr_up = cv2.resize(attr, (W, H), interpolation=cv2.INTER_LINEAR)
        return attr_up
