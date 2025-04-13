import os
import torch

class SSMPCA:
    def __init__(self, correspondences):
        """
        Compute the SSM based on eigendecomposition.
        Args:
            correspondences:    Corresponded shapes as a torch.Tensor
        """
        self.device = correspondences.device
        self.mean = torch.mean(correspondences, dim=0)

        data_centered = correspondences - self.mean
        cov_dual = torch.matmul(data_centered, data_centered.T) / (
            data_centered.shape[0] - 1
        )

        evals, evecs = torch.linalg.eigh(cov_dual)
        evecs = torch.matmul(data_centered.t(), evecs)
        # Normalize the col-vectors
        evecs /= torch.sqrt(torch.sum(evecs ** 2, dim=0))

        # Sort
        idx = torch.argsort(evals, descending=True)
        evecs = evecs[:, idx]
        evals = evals[idx]

        # Remove the last eigenpair (it should have zero eigenvalue)
        self.variances = evals[:-1]
        self.modes_norm = evecs[:, :-1]
        # Compute the modes scaled by corresp. std. dev.
        self.modes_scaled = self.modes_norm * torch.sqrt(self.variances)
        self.length = evecs.shape[0]

    def generate_random_samples(self, n_samples=1, n_modes=None):
        """
        Generate random samples from the SSM.
        Args:
            n_samples:  number of samples to generate
            n_modes:    number of modes to use
        Returns:
            samples:    Generated random samples as torch.Tensor
        """
        if n_modes is None:
            n_modes = self.modes_scaled.shape[1]
        weights = torch.randn(n_samples, n_modes).to(self.device)
        samples = self.mean + torch.matmul(weights, self.modes_scaled.t()[:n_modes])
        return samples.squeeze()

    def get_reconstruction(self, shape, n_modes=None):
        """
        Project shape into the SSM to get a reconstruction
        Args:
            shape:      shape to reconstruct as torch.Tensor
            n_modes:    number of modes to use. If None, all relevant modes are used
        Returns:
            data_proj:  projected data as reconstruction as torch.Tensor
        """
        shape = shape.view(-1)
        data_proj = shape - self.mean
        if n_modes:
            # restrict to max number of modes
            if n_modes > self.length:
                n_modes = self.modes_scaled.shape[1]
            evecs = self.modes_norm[:, :n_modes]
        else:
            evecs = self.modes_norm
        evecs_t = evecs.t()
        data_proj_re = data_proj.view(-1, 1)
        weights = torch.matmul(evecs_t, data_proj_re)
        data_proj = self.mean + torch.matmul(weights.t(), evecs_t)
        data_proj = data_proj.view(-1, 3)
        return data_proj.float()