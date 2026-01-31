"""Exercise 2: Image Batch Transformation
You're working on a computer vision model and have a batch of 4 grayscale images, each of size 3x3 pixels. The data is currently in a tensor with the shape [4, 3, 3], which represents [batch_size, height, width].

For processing with certain deep learning frameworks, you need to transform this data into the [batch_size, channels, height, width] format. Since the images are grayscale, you'll need to:

Add a new dimension of size 1 at index 1 to represent the color channel.
After adding the channel, you realize the model expects the shape [batch_size, height, width, channels]. Transpose the tensor to swap the channel dimension with the last dimension.
"""
# A dump of 
import  torch
import numpy as np
import pandas as pd

image_batch = torch.rand(4, 3, 3)

print("ORIGINAL BATCH SHAPE:", image_batch.shape)
print("-" * 45)

### START CODE HERE ###

# 1. Add a channel dimension at index 1.
image_batch_with_channel = None

# 2. Transpose the tensor to move the channel dimension to the end.
# Swap dimension 1 (channels) with dimension 3 (the last one).
image_batch_transposed = None

### END CODE HERE ###


print("\nSHAPE AFTER UNSQUEEZE:", image_batch_with_channel.shape)
print("SHAPE AFTER TRANSPOSE:", image_batch_transposed.shape)