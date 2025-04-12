import torch


def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    group_labels: torch.Tensor,
    l1_weight: float,
    group_weight: float
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    return (
        normalized_mean_squared_error(reconstruction, original_input)
        + normalized_L1_loss(latent_activations, original_input) * l1_weight 
        + group_steering_loss(latent_activations, group_labels) * group_weight
    )


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()


def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()

def group_steering_loss(latents: torch.Tensor, group_labels: torch.Tensor) -> torch.Tensor:
    """
    Encourage latent representations of different groups to be different.

    :param latents: [batch, latent_dim]
    :param group_labels: [batch] (e.g. int: 0, 1, 2,...)
    :return: negative average distance between group latent centers
    """
    unique_groups = group_labels.unique()
    group_means = []

    for group in unique_groups:
        mask = group_labels == group
        if mask.sum() == 0:
            continue
        group_latents = latents[mask]
        group_mean = group_latents.mean(dim=0)
        group_means.append(group_mean)

    if len(group_means) <= 1:
        return torch.tensor(0.0, device=latents.device)

    group_means = torch.stack(group_means)  # [num_groups, latent_dim]
    dist_matrix = torch.cdist(group_means, group_means, p=2)
    loss = -dist_matrix.triu(diagonal=1).mean()
    return loss
