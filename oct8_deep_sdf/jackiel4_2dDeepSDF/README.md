# DeepSDF Hacker 2: 2D SDF implementation

I implemented a simple 2D DeepSDF NN on the [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) dataset. This demo is implemented off the original DeepSDF codebase: https://github.com/facebookresearch/DeepSDF/tree/main

## Training
This section trains a 2D DeepSDF NN.

Run `train.ipynb`

Saved model weights and embedding layer weights in `./experiments_`.

## Reconstruction 
This section is to reconstruct SDFs from unseen, partial SDFs. 

Run `inference.ipynb`

Examples of reconstructed SDFs in `./experiments_/Reconstructions`.

## Dependencies
Standard `torch` and `torchvision` libraries were used. 

Additionally, install `numpy`, `matplotlib`, `easydict`.
