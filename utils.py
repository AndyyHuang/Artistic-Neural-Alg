from torchvision import transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

class ContentLoss(nn.Module):
    """ Calculates content loss of previous layer.
    """
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    """ Calculates style loss of previous layer.
    """
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input
    
class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    
    def forward(self, input):
        # im: B x C x H x W
        return (input - torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)) / torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    
def gram_matrix(input):
    # a = batch, b = feature maps, c,d = dims of feature map
    b, fms, m, n = input.size()

    # Resize into vector and compute activations
    activations = input.view(b * fms, m * n)
    gram = torch.mm(activations, activations.t())
    return gram.div(b * fms * m * n)

def create_neural_model(vgg19, content, style):
    """ Creates model and gets content and style activations for loss calculations.
    """
    relu_indicies = [1, 6, 11, 20, 29]
    content_losses, style_losses = [], []
    layers = list(vgg19.features.eval().children())

    # Initialize new model
    model = nn.Sequential(Normalize())

    for i in range(len(layers)):
        # Add in original modules
        layer = layers[i]
        if isinstance(layer, nn.ReLU):
            model.add_module(f'{i + 1}', nn.ReLU(inplace=False))
        elif isinstance(layer, nn.MaxPool2d):
            model.add_module(f'{i + 1}', nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding))
        else:
            model.add_module(f'{i + 1}', layer)

        # Get activations at layers of interest and add losses
        if i in relu_indicies:
            content_out = model(content).detach()
            style_out = model(style).detach()
            
            content_loss = ContentLoss(content_out)
            content_losses.append(content_loss)
            style_loss = StyleLoss(style_out)
            style_losses.append(style_loss)

            model.add_module(f"content_{i + 1}", content_loss)
            model.add_module(f"style_{i + 1}", style_loss)

        if i == 29:
            break
        i += 1
    return model, content_losses, style_losses

def import_images(content_path, style_path, height=400):
    content_pil = Image.open(content_path)
    style_pil = Image.open(style_path)
    resized_content = transforms.Resize(height)(content_pil)

    content = transforms.ToTensor()(resized_content) # Content
    resized_style = transforms.Resize((content.shape[1], content.shape[2]))(style_pil)
    style = transforms.ToTensor()(resized_style) # Style
    input = torch.rand(content.shape) # Model input

    return content.unsqueeze(0), style.unsqueeze(0), input.unsqueeze(0) # Add batch dim

def train_image(input, model, optimizer, epochs, w_c, w_s, content_losses, style_losses, content_layer, style_layer):
    epoch = 0
    step = 50
    while epoch < epochs:
        def train_loop():
            nonlocal epoch
            nonlocal step
            with torch.no_grad():
                input.clamp_(0, 1)

            optimizer.zero_grad()
            model(input)

            total_style_loss = 0
            content_loss = content_losses[content_layer - 1].loss
            for style_loss in style_losses[:style_layer]:
                total_style_loss += style_loss.loss

            # Calculate loss and backprop
            loss = w_c * content_loss + w_s * total_style_loss
            loss.backward()

            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] Total loss: {loss} Content loss: {w_c * content_loss} Style loss: {w_s * total_style_loss}")
            
            # Save results
            if epoch + 1 == 1 or epoch + 1 == step:
                pil_im = transforms.ToPILImage()(input.squeeze(0))
                pil_im.save(f"output/dancing_picasso_{epoch + 1}.jpg")
                step *= 2
            epoch += 1

            return loss
        
        optimizer.step(train_loop)
    return input