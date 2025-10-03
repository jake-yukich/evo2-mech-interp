import torch
from torch import nn, Tensor
from jaxtyping import Float
from torch.utils.data import Dataset


class FlatSubspace(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = nn.Linear(input_dim, output_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def encode(self, x: Float[Tensor, "n input_dim"]) -> Float[Tensor, "n output_dim"]:
        return self.encoder(x)

    def decode(self, z: Float[Tensor, "n output_dim"]) -> Float[Tensor, "n input_dim"]:
        W_pinv = torch.linalg.pinv(self.encoder.weight)
        return z @ W_pinv.T

    def pairwise_angular_distance(
        self, z: Float[Tensor, "n output_dim"]
    ) -> Float[Tensor, "n n"]:
        """Compute all pairwise angular distances."""
        z_norm = torch.nn.functional.normalize(z, dim=-1)
        cos_sim = z_norm @ z_norm.T
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        return self.beta * torch.arccos(cos_sim)

    def forward(self, x: Float[Tensor, "n input_dim"]) -> Float[Tensor, "n n"]:
        z = self.encode(x)
        return self.pairwise_angular_distance(z)


class FlatSubspaceLoss(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        d_pred_matrix: Float[Tensor, "n n"],
        d_true_matrix: Float[Tensor, "n n"],
        x_recon: Float[Tensor, "n input_dim"],
        x_true: Float[Tensor, "n input_dim"],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        triu_mask = torch.triu(torch.ones_like(d_pred_matrix), diagonal=1).bool()

        distance_loss = torch.nn.functional.mse_loss(
            d_pred_matrix[triu_mask], d_true_matrix[triu_mask]
        )
        reconstruction_loss = torch.nn.functional.mse_loss(x_recon, x_true)
        total_loss = distance_loss + self.alpha * reconstruction_loss

        return total_loss, distance_loss, reconstruction_loss


class PhylogeneticPairDataset(Dataset):
    """Dataset that yields pairs of genome embeddings with their phylogenetic distances."""

    def __init__(
        self,
        embeddings: Float[Tensor, "n_genomes d_model"],
        phylo_distance_matrix: Float[Tensor, "n_genomes n_genomes"],
    ):
        self.embeddings = embeddings.to(torch.float32)
        self.phylo_distance_matrix = phylo_distance_matrix.to(torch.float32)
        self.n_genomes = embeddings.shape[0]

        # generate all pair indices (i, j) where i < j (lower triangular, no diagonal)
        pairs = []
        for i in range(self.n_genomes):
            for j in range(i):
                pairs.append((i, j))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i, j = self.pairs[idx]

        x_i = self.embeddings[i]
        x_j = self.embeddings[j]
        d_phylo = self.phylo_distance_matrix[i, j]

        return x_i, x_j, d_phylo
