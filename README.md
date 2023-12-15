# A Neural Algorithm of Artistic Style Implementation

<p align="center">
  <img width="250" alt="Screenshot 2023-12-14 at 8 35 56 PM" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/bf37aed1-294b-4181-a154-9f87379d8039">
  <img width="250" alt="Screenshot 2023-12-14 at 8 35 56 PM" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/8a557e10-295c-4cd2-a2c7-d71f24158d62">
  <img width="250" alt="Screenshot 2023-12-14 at 8 35 56 PM" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/6605097b-fd24-4d14-906f-4c5a967328e0">
</p>

In this project, I implemented "A Neural Algorithm of Artistic Style" by Leon A. Gatys et al. and explored the results from varying hyper parameter configurations and methods.

## Overview
The artistic neural algorithm generates an image by propogating gradients down to the input image. Thus, the input image is learned as opposed to the network's weights by doing a gradient update with respect to the layer activations. Because only the input is trained, the weights of the pretrained model, VGG19, are frozen during training. The loss function is a weighted sum of the content and style losses.

<p align="center">
  <img width="400" alt="Screenshot 2023-12-14 at 8 35 56 PM" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/c9446b32-cb3a-4e02-aca0-37d28303ccef">
</p>

<p align="center">Total Loss</p>

Thus, two forward passes must be done before training to retrieve the style and content components from each of the model's convolutional layers to calculate the loss for the learned image during training.

## Implementation Details
In my implementation, I mostly followed what was on the original paper, however I made some adjustments which I will state here. I found that using MSE loss for both the content and style yielded the best results. Furthermore, for the style calculation, I normalized the gram matrix by dividing by the number of elements in it before it is passed into MSE. When using the loss functions (SSD) and without Gram matrix normalization (as outlined in the paper), my total loss exploded upwards to the hundred thousands for many iterations and had difficulty converging to a small loss. However, when using MSE for loss and gram matrix normalization, I was able to drive the loss down to <= 1 for all tests. The image seemed to converge around 500-600 training epochs. Another notable difference is that I omitted weights applied to each style loss layer when computing the total loss, as it helped with the issue stated previously.

As for the way I obtained the losses from each layer, I added content and style loss modules after the RELU layer that succeeds each convolutional layer of interest in VGG19 (which was imported with preinitialized weights using pytorch) in order to probe activation losses. It is necessary to put the loss modules after the RELU layer for the loss gradients to match the loss gradients presented in the paper. See the paper for more detail about the equations below.

<p align="center">
  <img width="350" alt="Screenshot 2023-12-14 at 8 27 48 PM" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/685121e7-94fb-40a8-8d69-ac5e86ddfb7f">
</p>
<p align="center">Gradient of Content Loss</p>

<p align="center">
  <img width="350" alt="Screenshot 2023-12-14 at 8 27 54 PM" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/e2823678-45e3-4272-9104-91a1a55496f0">
</p>
<p align="center">Gradient of Style Loss</p>

The convolutional layers of interest in my implementation were conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1. Additionally, I changed all max pool layers inside VGG19 to avg pool as the authors noted that avg pool seemed to work better. After each forward pass, I would calculate the appropriate loss using each module I added to VGG19. The optimizer I used was LBFGS. It is also worth noting that I froze the weights of VGG19 and put the model on eval mode while running gradient descent on the input image (as also mentioned in the paper). Finally, input normalization is not mentioned in the paper, however I found that adding a normalization layer as the first layer helped

Below are my best results. The gifs visualize the training process over 600 epochs. Each frame (aside from the first initial frame) is generated every 10 epochs. For the full report, see index.html.

<p align="center">
  <img alt="" src="content/orange_cat.jpg" width="30%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="style/der_schrei.jpg" width="30%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/66a0c542-a2ef-426b-8047-56040ee50f23" width="30%">
</p>

<p align="center">
  <img alt="" src="content/dog_with_stick.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="style/femme.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/c025ea26-5e76-4c57-9cac-fa97f9d97276" width="30%">
</p>

<p align="center">
  <img alt="" src="content/small_dog.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="style/starry_night.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/fae11f65-d092-495f-a0f7-cefd1a087380" width="30%">
</p>

<p align="center">
  <img alt="" src="content/tuebingen.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="style/composition_7.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="" src="https://github.com/AndyyHuang/Artistic-Neural-Alg/assets/76765795/b9e523c6-d01d-46b1-9746-83a7041f7a2a" width="30%">
</p>

Credits to Alison Friend for the cute animal portraits.
