"""Experiment B3: Mechanistic interpretability probes.

Uses Sparse Autoencoders (SAEs), linear probes, and activation patching
to locate personality circuits in the model's weights/activations.

Core hypothesis: if deposition is real structural change, it should leave
detectable traces in weight/activation geometry — e.g., early-phase features
should be closer to principal components (capturing more variance),
while late-phase features grow in orthogonal complement spaces.
"""
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for discovering interpretable features in activations.

    Trained on residual stream activations, the SAE learns a sparse overcomplete
    basis where each latent dimension ideally corresponds to a monosemantic feature.
    """

    def __init__(self, input_dim: int, hidden_dim: int, l1_coefficient: float = 0.001):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.l1_coefficient = l1_coefficient

        # Initialize decoder weights orthogonal for better feature separation
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and reconstruct activations.

        Returns (reconstruction, latent_activations).
        """
        latent = torch.relu(self.encoder(x))
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def loss(
        self, x: torch.Tensor, reconstruction: torch.Tensor, latent: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """SAE loss = reconstruction MSE + L1 sparsity penalty."""
        recon_loss = nn.functional.mse_loss(reconstruction, x)
        sparsity_loss = self.l1_coefficient * latent.abs().mean()
        total = recon_loss + sparsity_loss
        return total, {"recon_loss": recon_loss.item(), "sparsity_loss": sparsity_loss.item()}


def train_sae_on_activations(
    sae: SparseAutoencoder,
    activations: torch.Tensor,  # (n_samples, hidden_dim)
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> SparseAutoencoder:
    """Train the SAE on extracted activations."""
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for epoch in range(epochs):
        perm = torch.randperm(activations.size(0))
        for i in range(0, activations.size(0), batch_size):
            batch = activations[perm[i:i + batch_size]]
            recon, latent = sae(batch)
            loss, metrics = sae.loss(batch, recon, latent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return sae
