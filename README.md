# CIFAR-10 Neural Network Classifier from scratch

## What I’ve Done

- Built a two-layer neural network with ReLU activation and softmax loss using only NumPy, no deep learning frameworks.
- Implemented core neural components fully-connected layers, ReLU, softmax cross-entropy loss, and L2 regularization.
- Wrote custom forward and backward passes for all layers and loss functions.
- Trained the model on 20,000 CIFAR-10 images using mini-batch SGD and monitored performance over 10+ epochs.
- Developed a training pipeline with live loss/accuracy tracking and model checkpoint saving.
- Visualized training progress by plotting accuracy and loss per epoch.

  ## How to Run

Download CIFAR-10 dataset:
bash download_cifar.sh

Run gradient checks for layers.py:
python gradcheck_layers.py

Run gradient checks for two_layer_net.py:
python gradcheck_classifier.py

Test the best model:
python test.py
