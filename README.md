# An Implementation of "A Neural Algorithm of Artistic Style" by Leon A. Gatys et al

In this project, I implemented "A Neural Algorithm of Artistic Style" by Leon A. Gatys et al. and explored results generated using different hyper parameter configurations and methods.

## Implementation Details
In my implementation, I found that using MSE loss for both the content and style yielded the best results. Furthermore, for the style calculation, I normalized the gram matrix by dividing by the number of elements in it before it is passed into MSE. When using the loss functions (SSD) and without Gram matrix normalization (as outlined in the paper), my total loss exploded upwards to the hundred thousands for many iterations and had difficulty converging to a small loss. However, when using MSE for loss and gram matrix normalization, I was able to drive the loss down to <= 1 for all tests. The image seemed to converge around 500-600 training epochs. Another notable difference is that I omitted weights applied to each style loss layer when computing the total loss, as it helped with the issue stated previously. Finally, input normalization is not mentioned in the paper, however I found that adding a normalization layer as the first layer helped.

As for the way I obtained the losses from each layer, I added content and style loss modules after the RELU layer that succeeds each convolutional layer of interest in order to probe activation losses. It is necessary to put the loss modules after the RELU layer for the gradients to match what was stated in the paper. The convolutional layers of interest in my implementation were conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1. In the paper, the authors used conv4_2 to generate their results. Additionally, I changed all max pool layers inside VGG19 to avg pool as the authors noted that avg pool seemed to work better. After each forward pass, I would calculate the appropriate loss using each module I added to VGG19. The optimizer I used was LBFGS. It is also worth noting that I froze the weights of VGG19 and put the model on eval mode while running gradient descent on the input image (as also mentioned in the paper).

## Initial Results
For my initial results below, I used the following setup:
<ul>
  <li>Input: Randomly Generated Image (unseeded)</li>
  <li>Content Loss Weight: 1</li>
  <li>Style Loss Weight: 1_000_000</li>
  <li>Content Layer: conv4_1</li>
  <li>Style Layers: conv1_1, conv2_1, conv3_1, conv4_1, and conv5_1</li>
  <li>Optimizer: LBFGS</li>
  <li>Epochs: 600</li>
</ul>

Below are visualizations of the training process over 600 epochs. Each frame (aside from the first initial frame) is generated every 10 epochs.
