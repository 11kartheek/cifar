# CIFAR10 Training with PyTorch and GradCAM Visualization

This repository demonstrates the implementation of ResNet18 on the CIFAR10 dataset using PyTorch. Additionally, it includes visualization of GradCAM (Gradient-weighted Class Activation Mapping) on misclassified outputs.

## Files

### `utils.py`
- [Link to utils.py](https://github.com/11kartheek/cifar/blob/main/utils.py)
- **Description:** `utils.py` contains helper functions and variables such as:
  - Train transforms
  - Test transforms
  - Misclassified images helper function
  - Plot loss function
  - Visualize GradCAM function

### `main.py`
- [Link to main.py](https://github.com/11kartheek/cifar/blob/main/main.py)
- **Description:** `main.py` serves as the main script for training the ResNet18 model. It includes functions like:
  - Train-test split
  - Train loop

### `models Folder`
- [link to resnet] (https://github.com/11kartheek/cifar/blob/main/models/resnet.py)
- **Description:** This folder contains various models, including ResNet18.

### `s11.ipynb`
- [Link to s11 notebook](https://github.com/11kartheek/cifar/blob/main/KartheekB_s11.ipynb)
- **Description:** `s11.ipynb` is a Jupyter notebook that provides a step-by-step walkthrough of the functions, showcasing misclassified images and corresponding GradCAM images.

