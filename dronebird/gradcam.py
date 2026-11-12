"""Grad-CAM, written with tensor operations (no library).

Author: Mariah Habak
Reference: Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV.

For a class c and a chosen layer with activations A (1, K, h, w):
    weights  = mean of dScore_c / dA over h and w     -> (1, K, 1, 1)
    cam      = ReLU( sum over K of weights * A )        -> (1, 1, h, w)
    cam      = bilinear upscale to the input size, scaled to [0, 1]
"""

import torch
import torch.nn.functional as F
from torch import nn

from dronebird.data.datasets import normalise


class GradCAM:
    def __init__(self, network: nn.Module, layer: nn.Module):
        self.__net = network
        self.__layer = layer
        self.__activations = None
        self.__gradients = None

    def __forward_hook(self, module, inputs, output):
        self.__activations = output
        output.register_hook(self.__save_gradient)

    def __save_gradient(self, grad):
        self.__gradients = grad

    def __call__(self, image: torch.Tensor, target_class: int = None) -> tuple:
        """Float image (3, S, S) in [0, 1] -> (heat map (S, S) in [0, 1], class, scores)."""
        device = next(self.__net.parameters()).device
        handle = self.__layer.register_forward_hook(self.__forward_hook)
        try:
            self.__net.eval()
            x = normalise(image).unsqueeze(0).to(device)
            logits = self.__net(x)
            scores = logits.softmax(dim=1)[0].detach().cpu()
            cls = int(scores.argmax()) if target_class is None else int(target_class)
            self.__net.zero_grad()
            logits[0, cls].backward()
            weights = self.__gradients.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.__activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear",
                                align_corners=False)[0, 0]
            cam = cam - cam.min()
            cam = cam / cam.max().clamp(min=1e-8)
        finally:
            handle.remove()
        return cam.detach().cpu(), cls, scores
