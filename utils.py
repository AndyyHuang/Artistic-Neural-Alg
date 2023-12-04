import torchvision.models as models
import torchvision.utils as tvutils
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
    
    def forward(self, content, pred_content):
        return 0.5 * torch.sum(torch.sub(content, pred_content)**2)

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
    
    def forward(self, style, pred_style):
        return 0.5 * torch.sum(torch.sub(style, pred_style)**2)
    
class TotalLoss(nn.Module):
    def __init__(self, w_c, w_s, content_loss_fn, style_loss_fn):
        super(TotalLoss, self).__init__()
        self.w_c, self.w_s = w_c, w_s
        self.content_loss_fn, self.style_loss_fn = content_loss_fn, style_loss_fn
    
    def forward(self, content, style, pred_content, pred_style):
        return self.w_c * self.content_loss_fn(content, pred_content) + self.w_s * self.style_loss_fn(style, pred_style)
    
def get_activation(feature_type, feature_maps):
  # The hook signature
  def hook(model, input, output):
    feature_maps[f"{feature_type}"] = output.detach()
  return hook
    
def get_content_and_style(content_im, style_im, vgg19):
    # Indices for the layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', and 'conv5_1'.
    conv_layers = [0, 5, 10, 19, 28]

    vgg19.eval()
    feature_maps = {}

    # Register hooks for content
    for i in range(len(conv_layers)):
        layer = vgg19.features[conv_layers[i]]
        layer._forward_hooks.clear() # Unregister previous hook
        # Register new hook
        layer.register_forward_hook(get_activation(f"content_{i + 1}_1", feature_maps))

    # Forward pass of content
    with torch.no_grad():
        vgg19(content_im)

    # Register hooks for style
    for i in range(len(conv_layers)):
        layer = vgg19.features[conv_layers[i]]
        layer._forward_hooks.clear() # Unregister previous hook
        # Register new hook
        layer.register_forward_hook(get_activation(f"style_{i + 1}_1", feature_maps))

    # Forward pass of content
    with torch.no_grad():
        vgg19(style_im)

    return feature_maps