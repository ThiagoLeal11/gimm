import torch
import torch.nn.functional as F


def backward_class_score_and_get_gradients(discriminator, label, x, device):
    x_input = x.detach().clone()
    x_input.requires_grad = True
    preds = discriminator.forward_from_activations(x_input, hook=False)

    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1  # label deve ser int
    score = torch.sum(one_hot * preds)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=x_input,
        retain_graph=True,  # True para não deletar o grafo do treino principal
        create_graph=False,
        only_inputs=True
    )[0]

    return gradients

def make_resize_norm(act_grads, image_size):
    heatmap = F.interpolate(act_grads.unsqueeze(0).unsqueeze(0), (image_size, image_size), mode='bilinear', align_corners=False)

    # Normalização segura
    heatmap = heatmap - heatmap.min()
    maxv = heatmap.max()
    if maxv > 0:
        heatmap = heatmap / maxv

    return heatmap.squeeze()

def compute_iia_heatmap_single_new(discriminator, image, label, num_steps, device):
    was_training = discriminator.training
    discriminator.eval()

    image_size = image.shape[2]
    activations: torch.Tensor = discriminator.get_activations(image)

    # Calcula gradientes para cada passo de interpolação
    act = activations.type(torch.float32)
    grad = backward_class_score_and_get_gradients(discriminator, label, act, device)

    # -------------- Até aqui está igualzinho! --------------
    # Integra gradientes
    gradsum = grad.squeeze() * F.relu(activations.squeeze())
    integrated_heatmaps = torch.sum(gradsum, dim=[0])

    heatmap = make_resize_norm(integrated_heatmaps, image_size)

    # Restaurar modo de treinamento se necessário
    if was_training:
        discriminator.train()

    return heatmap


def compute_iia_heatmap(discriminator, images, label, num_steps, device):
    """
    Calcula o heatmap IIA para um batch de imagens e classe específica.
    Processa cada imagem individualmente (como na referência) e empilha os resultados.

    Args:
        discriminator: Discriminator model
        images: Input images tensor [batch_size, C, H, W]
        label: Label da classe alvo (int: 0 para real, 1 para fake)
        num_steps: Número de passos de interpolação
        device: Device to run computation on

    Returns:
        Heatmap tensor [batch_size, H, W]
    """
    batch_size = images.shape[0]

    heatmaps = []
    for i in range(batch_size):
        # Processa cada imagem individualmente (como na referência)
        single_image = images[i:i+1]  # [1, C, H, W]
        # label é passado como int (não tensor)
        new_heatmap = compute_iia_heatmap_single_new(discriminator, single_image, label, num_steps, device)
        heatmaps.append(new_heatmap)

    # Empilha os heatmaps: [batch_size, H, W]
    return torch.stack(heatmaps)