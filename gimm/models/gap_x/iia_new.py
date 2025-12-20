import torch
import torch.nn.functional as F


def samples_linear_normalization(samples: torch.Tensor) -> torch.Tensor:
    mins = samples.flatten(start_dim=1).min(dim=1)[0].view(-1, 1, 1, 1)
    maxs = samples.flatten(start_dim=1).max(dim=1)[0].view(-1, 1, 1, 1)
    denominator = (maxs - mins).clamp_min(1e-12)
    normalized_imgs = (samples - mins) / denominator
    return normalized_imgs


def compute_gradients_for_label(discriminator: torch.nn.Module, activations: torch.Tensor, label: int):
    acts = activations.detach().clone().requires_grad_(True)
    logits = discriminator.forward_classifier(acts)

    one_hot = torch.zeros_like(logits)
    one_hot[:, label] = 1
    score = torch.sum(one_hot * logits)

    gradients = torch.autograd.grad(outputs=score, inputs=acts, create_graph=False)[0]
    return gradients


def compute_iia_heatmap(discriminator: torch.nn.Module, images: torch.Tensor, label: int):
    was_training = discriminator.training
    discriminator.eval()

    activations: torch.Tensor = discriminator.forward_features(images)
    grad = compute_gradients_for_label(discriminator, activations, label)

    gradsum = grad * F.relu(activations)
    integrated_heatmaps = torch.sum(gradsum, dim=1)

    raw_heatmaps = integrated_heatmaps.unsqueeze(1)
    scaled_heatmaps = F.interpolate(raw_heatmaps, images.shape[2:], mode='bicubic', align_corners=False)
    heatmaps = samples_linear_normalization(scaled_heatmaps)

    if was_training:
        discriminator.train()

    return heatmaps
