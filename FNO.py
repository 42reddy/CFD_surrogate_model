import torch
import torch.nn as nn


class FNO_block(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

        self.w = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(
            x_ft[:, :, :self.modes, :self.modes], self.weights
        )

        x_ifft = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        x_proj = self.w(x)

        return x_ifft + x_proj



class FNO_network(nn.Module):
    def __init__(self, in_channels, out_channels, modes, width=64, n_blocks=4, dropout=0.0):
        super().__init__()

        self.input_proj = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            FNO_block(width, width, modes) for _ in range(n_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)
            x = torch.relu(x)

        return self.output_proj(x)