"""
gradcam.py
----------
Grad-CAM (Gradient-weighted Class Activation Mapping): produces a heatmap
showing WHERE in the image the model focused to make its decision.

WHY THIS MATTERS FOR MEDICAL AI:
A model can be "right for the wrong reasons" -- e.g. it might learn to
detect a hospital's specific X-ray machine watermark instead of the actual
lung disease pattern. Grad-CAM lets you (and eventually a doctor) sanity
check that the model is looking at the LUNGS, not some irrelevant corner
of the image. This is considered essential for any credible medical AI
project, even a student/portfolio one.

HOW IT WORKS (intuition, not deep math):
The last convolutional layer of a CNN still retains spatial information
(which part of the image corresponds to which part of the feature map),
while being rich in "what is this" information. Grad-CAM looks at how
much each spatial location in that last layer contributed to the final
prediction, and turns that into a heatmap.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: a single preprocessed image, shape (1, 3, 224, 224)
        class_idx: which class to explain. If None, uses the model's
                   top predicted class.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        # Global-average-pool the gradients -> importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps -> class activation map
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only keep positive influence

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize to 0-1

        return cam, class_idx


def overlay_heatmap(original_image_np, cam, alpha=0.4):
    """
    original_image_np: the original image as a numpy array (H, W, 3), values 0-255
    cam: the Grad-CAM heatmap (H, W), values 0-1
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = heatmap * alpha + original_image_np * (1 - alpha)
    return np.uint8(overlayed)


def show_gradcam(model, image_tensor, original_image_np, target_layer, class_names, save_path=None):
    """
    Full pipeline: generate + overlay + display.
    target_layer for ResNet50 is typically model.layer4[-1]
    """
    gradcam = GradCAM(model, target_layer)
    cam, predicted_class = gradcam.generate(image_tensor.unsqueeze(0))
    overlayed = overlay_heatmap(original_image_np, cam)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(overlayed)
    axes[1].set_title(f"Grad-CAM: predicted '{class_names[predicted_class]}'")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return predicted_class
