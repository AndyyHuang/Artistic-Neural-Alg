import torchvision.models as models
import torchvision.utils as tvutils
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import torch
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load VGG19 and freeze weights.
vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
for p in vgg19.parameters():
    p.requires_grad = False

# Import images
starry_night = torch.from_numpy(skio.imread("assets/starry_night.jpg"))
neckarfront = torch.from_numpy(skio.imread("assets/Tuebingen_Neckarfront.jpg"))
input_im = torch.rand(neckarfront.shape)
input_im.requires_grad = True

print(starry_night.shape)
print(neckarfront.shape)
# plt.imshow(input_im)
# plt.show()

# Hyper Params
epochs = 1000
step = 50
w_c, w_s = 1, .001

optimizer = torch.optim.Adam([input_im])
content_loss_fn, style_loss_fn = utils.ContentLoss(), utils.StyleLoss()
total_loss_fn = utils.TotalLoss(w_c, w_s, content_loss_fn, style_loss_fn)

# Train Image
for epoch in range(epochs):
    vgg19.train()
    pred = vgg19(input_im)
    content_loss = content_loss_fn(pred, y)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Visualize image
    if epoch == 1 or epoch == step:
        tvutils.save_image(input_im, f"pred_{epoch}")
        if epoch != 1:
            step *= 2

# output_that_you_want = ...
# actual_output = net(input)
# some_loss = SomeLossFunction(output_that_you_want, actual_output)