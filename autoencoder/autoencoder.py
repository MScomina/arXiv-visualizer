import torch
from torch import nn
import torch.distributions

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.map = nn.Linear(in_dim, out_dim, bias=False)
        self.bn  = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop= nn.Dropout(dropout)

        # If dims differ we need a projection so the skip can be added
        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.map(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)

        # residual shortcut
        res = self.proj(x)
        return out + res

class VariationalEncoder(nn.Module):
    def __init__(
        self,
        in_dimensions: int = 768,
        hidden_layers: tuple[int, ...] = (512, 256),
        latent_size: int = 128,
        dropout: float = 0.2,
        beta: float = 1.0,
    ):
        super().__init__()

        layers = []
        inp = in_dimensions
        if hidden_layers:
            for h in hidden_layers:
                layers.append(ResidualBlock(inp, h, dropout))
                inp = h

        self.encoder  = nn.Sequential(*layers)
        self.mean     = nn.Linear(inp, latent_size)
        self.logvar   = nn.Linear(inp, latent_size)

        self.N  = torch.distributions.Normal(torch.zeros(1), torch.ones(1))
        self.beta = beta

    def forward(self, x: torch.Tensor):
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)

        h   = self.encoder(x)
        mu  = self.mean(h)
        logv= self.logvar(h)

        sigma = torch.exp(0.5 * logv)
        eps   = torch.randn_like(mu)
        z     = mu + sigma * eps

        kl = self.beta * 0.5 * torch.sum(
            -logv + torch.exp(logv) + mu.pow(2) - 1,
            dim=1
        ).mean()
        return z, mu, logv, kl

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dims: int,
        hidden_layers: tuple[int, ...] = (512, 256),
        out_dims: int = 768,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        inp = latent_dims
        if hidden_layers:
            for h in reversed(hidden_layers):
                layers.append(ResidualBlock(inp, h, dropout))
                inp = h

        layers.append(nn.Linear(inp, out_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        return self.net(z)

class VAE(nn.Module):
    def __init__(
        self,
        in_dimensions: int = 768,
        hidden_layers: tuple[int, ...] = (512, 256),
        latent_size: int = 128,
        dropout: float = 0.2,
        beta: float = 1.0,
    ):
        super().__init__()
        self.beta = beta 
        self.var_encoder = VariationalEncoder(
            in_dimensions=in_dimensions,
            hidden_layers=hidden_layers,
            latent_size=latent_size,
            dropout=dropout,
            beta=beta,
        )
        self.decoder = Decoder(
            latent_dims=latent_size,
            hidden_layers=hidden_layers,
            out_dims=in_dimensions,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the VAE.

        Returns
        -------
        recon  : Tensor
            Reconstructed input.
        mu     : Tensor
            Mean of the approximate posterior.
        logvar : Tensor
            Log‑variance of the approximate posterior.
        kl     : Tensor
            KL divergence term.
        """
        z, mu, logv, kl = self.var_encoder(x)
        recon = self.decoder(z)
        return recon, mu, logv, kl