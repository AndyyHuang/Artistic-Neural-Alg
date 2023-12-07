import torch
import torchvision.models as models
from utils import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f"Running on device: {device}")

# Load VGG19.
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Import images
content, style, input = import_images("content/dog_with_stick.jpg", "style/femme.jpg")
output_dir_path = "output_32131/picasso_dog"
# input = torch.clone(content)
input.requires_grad_(True)
optimizer = torch.optim.LBFGS([input])

# Hyper Params
epochs = 600
w_c, w_s = 1, 10000
content_layer = 1
style_layer = 5

# Get model and losses
model, content_losses, style_losses = create_neural_model(vgg19, content, style)
model.eval()
model.requires_grad_(False)

# Logger
writer = SummaryWriter(f'stats/{output_dir_path}')

# Save final result
output = train_image(input, model, optimizer, epochs, w_c, w_s, content_losses, style_losses, content_layer, style_layer, output_dir_path, writer=None, log=True)
store_image(f"{output_dir_path}/{epochs}.jpg", output)
