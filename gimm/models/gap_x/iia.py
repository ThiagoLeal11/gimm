import numpy as np
import torch
import torch.nn.functional as F

from gimm.models.gap_x.iia_new import compute_iia_heatmap_single_new


def get_interpolated_values(baseline, target, num_steps):
    """
    Gera valores interpolados entre baseline e target.
    Usado para integração no método IIA.

    Args:
        baseline: Tensor baseline (geralmente zeros ou mínimo)
        target: Tensor alvo
        num_steps: Número de passos de interpolação

    Returns:
        Array numpy com os valores interpolados
    """
    if num_steps <= 0:
        return np.array([])
    if num_steps == 1:
        return np.array([baseline, target])

    delta = target - baseline

    if baseline.ndim == 3:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis]
    elif baseline.ndim == 4:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    elif baseline.ndim == 5:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    else:
        scales = np.linspace(0, 1, num_steps + 1, dtype=np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta.detach().cpu().numpy(), shape)
    interpolated_activations = baseline.detach().cpu().numpy() + deltas

    return interpolated_activations


def backward_class_score_and_get_gradients(discriminator, label, x, device):
    """
    Realiza backpropagation para uma classe específica e retorna os gradientes das ativações.

    Usa torch.autograd.grad() em vez de backward() para ser compatível com treinamento,
    permitindo calcular gradientes sem interferir no grafo computacional principal.

    Args:
        discriminator: Discriminator model
        label: Label da classe alvo (int, não tensor)
        x: Tensor de entrada (ativações)
        device: Device to run computation on

    Returns:
        Gradientes das ativações
    """
    discriminator.zero_grad()
    x.requires_grad = True
    x.requires_grad_(True)
    preds = discriminator.forward_from_activations(x.to(device), hook=False)

    # Debug: verificar se preds tem valores diferentes para cada classe
    # print(f"Preds shape: {preds.shape}, values: {preds[0]}")

    one_hot = torch.zeros(preds.shape).to(device)
    one_hot[:, label] = 1  # label deve ser int

    score = torch.sum(one_hot * preds)

    gradients = torch.autograd.grad(
        outputs=score,
        inputs=x,
        retain_graph=True,  # True para não deletar o grafo do treino principal
        create_graph=False,
        only_inputs=True
    )[0]

    # Debug: verificar gradientes
    # print(f"Label: {label}, Score: {score.item()}, Grad mean: {gradients.mean().item()}, Grad std: {gradients.std().item()}")

    gradients = gradients.unsqueeze(1).detach().cpu()
    return gradients


def make_resize_norm(act_grads, image_size):
    """
    Redimensiona e normaliza os gradientes das ativações para o tamanho da imagem.

    Args:
        act_grads: Gradientes das ativações
        image_size: Tamanho da imagem de saída

    Returns:
        Heatmap normalizado como tensor
    """
    # Garantir que o tensor tenha entre 2 e 4 dims e transformar em (N, C, H, W)
    t = act_grads
    # Se for tensor com batch e canal (N, C, H, W)
    if t.dim() == 4:
        # Se tiver batch de 1, remover e somar canais (comportamento anterior)
        if t.size(0) == 1:
            heatmap = t.squeeze(0).sum(dim=0)
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        else:
            # múltiplos batches: somar canais, depois somar batches
            heatmap = t.sum(dim=1).sum(dim=0)
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        # (C, H, W) -> somar canais -> (1,1,H,W)
        heatmap = t.sum(dim=0).unsqueeze(0).unsqueeze(0)
    elif t.dim() == 2:
        # (H, W) -> apenas adicionar dimensões de batch e canal
        heatmap = t.unsqueeze(0).unsqueeze(0)
    else:
        # Se tiver mais de 4 dims (por exemplo (1,1,C,H,W)), agregue as dimensões
        # extras (batch/canais) somando/achando para reduzir para (1,1,H,W).
        # Isso evita erros com F.interpolate quando o tensor tem K>2 dimensões espaciais.
        # Colapsar todas as dimensões exceto as duas últimas e somar
        heatmap = t.reshape(-1, t.shape[-2], t.shape[-1]).sum(dim=0).unsqueeze(0).unsqueeze(0)

    # Agora heatmap está em (N, C, H, W) - realizar resize
    heatmap = F.interpolate(heatmap, size=(image_size, image_size), mode='bicubic', align_corners=False)

    # Normalização segura
    heatmap = heatmap - heatmap.min()
    maxv = heatmap.max()
    if maxv > 0:
        heatmap = heatmap / maxv

    heatmap = heatmap.squeeze()
    return heatmap


def compute_iia_heatmap_single(discriminator, image, label, num_steps, device):
    """
    Calcula o heatmap IIA para UMA imagem e classe específica.
    Copiado diretamente da implementação de referência (compute_iia_heatmap).

    Args:
        discriminator: Discriminator model
        image: Input image tensor [1, C, H, W] (single image with batch dim)
        label: Label da classe alvo (int)
        num_steps: Número de passos de interpolação
        device: Device to run computation on

    Returns:
        Heatmap tensor [H, W]
    """
    # Colocar em modo eval como na referência original
    was_training = discriminator.training
    discriminator.eval()

    image_size = image.shape[2]

    # Obtém ativações
    activations = discriminator.get_activations(image)
    activations_featmap = activations.unsqueeze(1)

    # Baseline: mínimo das ativações
    baseline, _ = torch.min(activations_featmap, dim=1)
    baseline = torch.ones_like(activations_featmap) * baseline.unsqueeze(1)

    # Interpola ativações
    ig_activations = get_interpolated_values(
        baseline.detach(),
        activations_featmap,
        num_steps
    )

    # Calcula gradientes para cada passo de interpolação
    grads = []
    for act in ig_activations:
        act_tensor = torch.tensor(act, dtype=torch.float32)
        act_tensor.requires_grad = True
        grad = backward_class_score_and_get_gradients(discriminator, label, act_tensor, device)
        grads.append(grad.detach())
        act_tensor.requires_grad = False

    # Integra gradientes
    with torch.no_grad():
        integrated_grads = torch.stack(grads).detach()
        ig_activations_tensor = torch.tensor(ig_activations)
        ig_activations_tensor[1:] = ig_activations_tensor[1:] - ig_activations_tensor[0]

        gradsum = torch.sum(
            integrated_grads.squeeze().detach() * F.relu(ig_activations_tensor.squeeze()),
            dim=[0]
        )
        integrated_heatmaps = torch.sum(gradsum, dim=[0])

    heatmap = make_resize_norm(integrated_heatmaps.unsqueeze(0).unsqueeze(0), image_size)

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
        # heatmap = compute_iia_heatmap_single(discriminator, single_image, label, num_steps, device)
        new_heatmap = compute_iia_heatmap_single_new(discriminator, single_image, label, num_steps, device)
        heatmaps.append(new_heatmap)

    # Empilha os heatmaps: [batch_size, H, W]
    return torch.stack(heatmaps)