import torch
import torchvision.models as models
from utils import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f"Running on device: {device}")

# Load VGG19.
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Hyper Params
epochs = 600
w_c, w_s = 1, 1_000_000
content_layer = 4
style_layer = 5

# Import images
content, style, input = import_images("content/small_dog.jpg", "style/starry_night.jpg")
output_dir_name = "starry_dog"
input.requires_grad_(True)

# Get model and losses
model, content_losses, style_losses = create_neural_model(vgg19, content, style)
model.eval()
model.requires_grad_(False)
optimizer = torch.optim.LBFGS([input])

# Logger
writer = SummaryWriter(f'stats/run_1/{output_dir_name}')

# Save final result
output = train_image(input, model, optimizer, epochs, w_c, w_s, content_losses, style_losses, content_layer, style_layer, writer, output_dir_name).clamp(0, 1)
pil_im = transforms.ToPILImage()(output.squeeze(0))
pil_im.save(f"output/{output_dir_name}/{epochs}.jpg")
