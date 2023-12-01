import numpy as np
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_filters: int, num_latent_var: int):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_latent_var = num_latent_var
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        encoder_output_dim = (num_filters, 7, 7)
        encoder_output_size = int(np.prod(encoder_output_dim))
        self.y_encoder = nn.Sequential(nn.Linear(encoder_output_size, 128), nn.ReLU(), nn.Linear(128, output_size))
        self.z_mean = nn.Linear(encoder_output_size, num_latent_var)
        self.log_z_var = nn.Linear(encoder_output_size, num_latent_var)

        self.y_decoder = nn.Sequential(
            nn.Linear(output_size, 128), nn.ReLU(), nn.Linear(128, encoder_output_size), nn.ReLU()
        )
        self.z_decoder = nn.Linear(num_latent_var, encoder_output_size)

        self.output_decoder = nn.Sequential(
            nn.Unflatten(1, encoder_output_dim),
            nn.ConvTranspose2d(in_channels=num_filters, out_channels=num_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=num_filters, out_channels=1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_encoder(x)
        y_pred = self.y_encoder(x)
        z_mean = self.z_mean(x)
        log_z_var = self.log_z_var(x)
        return y_pred, z_mean, log_z_var

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_pred, z_mean, log_z_var = self.encode(x)
        z = self.sample(z_mean, log_z_var)
        x_hat = self.decode(z, y_pred.softmax(dim=-1))
        return x_hat, z_mean, log_z_var, y_pred

    def sample(self, z_mean: torch.Tensor, log_z_var: torch.Tensor) -> torch.Tensor:
        std = log_z_var.mul(0.5).exp_()
        epsilon = torch.ones_like(std).normal_()
        return z_mean + std * epsilon

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.z_decoder(z) + self.y_decoder(y)
        return self.output_decoder(x)
