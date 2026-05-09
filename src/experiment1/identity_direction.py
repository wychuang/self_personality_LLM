"""Identity direction vector (v_id) computation and analysis.

Computes v_id from extracted activations, provides analysis tools
for understanding the identity direction's properties.
"""
import torch
import torch.nn.functional as F


def compute_v_id(activations: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Compute identity direction from extracted activations.

    Args:
        activations: (num_texts, hidden_dim) tensor of extracted activations
        method: "mean" = simple average, "pca" = first principal component

    Returns:
        v_id: (hidden_dim,) unit-norm identity direction vector
    """
    if method == "mean":
        v_id = activations.mean(dim=0)
    elif method == "pca":
        # Center and compute first principal component
        centered = activations - activations.mean(dim=0, keepdim=True)
        _, _, V = torch.svd(centered.T)
        v_id = V[:, 0]
    else:
        raise ValueError(f"Unknown method: {method}")

    return F.normalize(v_id, dim=0)


def compute_id_loss(hidden_states: torch.Tensor, v_id: torch.Tensor) -> torch.Tensor:
    """Compute identity alignment loss: negative cosine similarity with v_id.

    Args:
        hidden_states: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        v_id: (hidden_dim,) identity direction vector

    Returns:
        scalar loss: -cos(hidden_states, v_id) averaged over batch
    """
    if hidden_states.dim() == 3:
        # (batch, seq_len, hidden_dim) — use last non-padding position
        hidden_states = hidden_states[:, -1, :]

    hidden_norm = F.normalize(hidden_states.float(), dim=-1)
    v_id_norm = F.normalize(v_id.float(), dim=-1)

    cosine_sim = (hidden_norm * v_id_norm).sum(dim=-1)
    return -cosine_sim.mean()


def analyze_v_id(
    v_id: torch.Tensor,
    activations: torch.Tensor,
) -> dict:
    """Analyze the identity direction's properties.

    Returns dict with:
      - norm: L2 norm of v_id before normalization
      - projection_spread: std of cosine similarities across seed texts
      - explained_variance_ratio: fraction of variance along v_id
    """
    v_id_unnormalized = activations.mean(dim=0)
    norm = float(v_id_unnormalized.norm().item())

    # Projection spread
    hidden_norm = F.normalize(activations.float(), dim=-1)
    v_id_norm = F.normalize(v_id.float(), dim=-1)
    cosines = (hidden_norm * v_id_norm).sum(dim=-1)
    projection_spread = float(cosines.std().item())

    # Explained variance along v_id
    centered = activations.float() - activations.float().mean(dim=0, keepdim=True)
    total_var = float((centered ** 2).sum().item())
    projections = (centered @ v_id.float())
    explained_var = float((projections ** 2).sum().item())
    explained_variance_ratio = explained_var / total_var if total_var > 0 else 0.0

    return {
        "norm": norm,
        "projection_spread": projection_spread,
        "explained_variance_ratio": explained_variance_ratio,
    }
