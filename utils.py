

import math
import os
import sys
import time

import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from albumentations.pytorch import ToTensorV2

means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

train_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=means,
        ),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),
    ]
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def denormalize_image(image):
    # Denormalize the image using means and standard deviations
    for i in range(3):  # Assuming RGB image
        image[i] = (image[i] * stds[i]) + means[i]
    return image


def denormalize_images(misclassified_images):
    for image in misclassified_images:
        image = denormalize_image(image)


def plot_losses(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def misclassifiedimages(model, device, test_loader, count=10):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified_images = []  # List to store misclassified images
    misclassified_targets = []  # List to store misclassified targets
    misclassified_predictions = []  # List to store misclassified predictions

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Check for misclassified images
            misclassified_idx = ~pred.eq(target.view_as(pred)).squeeze()
            misclassified_data = data[misclassified_idx]
            misclassified_target = target[misclassified_idx]
            misclassified_pred = pred[misclassified_idx]
            misclassified_images.extend(misclassified_data.cpu().numpy())
            misclassified_targets.extend(misclassified_target.cpu().numpy())
            misclassified_predictions.extend(misclassified_pred.cpu().numpy())
            if len(misclassified_predictions) > count:
                break
    return misclassified_images, misclassified_targets, misclassified_predictions


def plot_misclassified_images(
    misclassified_images, misclassified_targets, misclassified_predictions
):
    num_misclassified = min(len(misclassified_images), 10)
    plt.figure(figsize=(10, 5))
    for i in range(num_misclassified):
        plt.subplot(2, 5, i + 1)
        img = misclassified_images[i]
        plt.imshow(img.transpose((1, 2, 0)))
        plt.title(f"Pred: {classes[int(misclassified_predictions[i])]}")
        plt.axis("off")
        plt.text(
            0,
            misclassified_images[i].shape[1] + 2,
            f"Target: {classes[int(misclassified_targets[i])]}",
            ha="left",
            va="top",
            color="blue",
        )
    plt.show()


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def visualize_gradcam(
    model, device, misclassified_images, misclassified_predictions, target_classes
):
    for i, (image, pred, target) in enumerate(
        zip(misclassified_images, misclassified_predictions, target_classes)
    ):
        if i > 9:
            break
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        target_class = target

        # Initialize GradCAM
        cam = GradCAM(model=model, target_layers=[model.module.module.layer3[-1]])

        # Compute Grad-CAM
        grayscale_cam = cam(input_tensor=image_tensor)

        # Visualize Grad-CAM
        visualization = show_cam_on_image(image.transpose(1, 2, 0), grayscale_cam[0])

        # Plot Grad-CAM
        plt.figure(figsize=(4, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(image.transpose(1, 2, 0))
        plt.title(f"Predicted Class: {classes[int(pred)]}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM on image {i}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
