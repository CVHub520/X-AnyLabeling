import torch
from torch import nn


class PositionalEncodingsFixed(nn.Module):

    def __init__(self, emb_dim, temperature=10000):

        super(PositionalEncodingsFixed, self).__init__()

        self.emb_dim = emb_dim
        self.temperature = temperature

    def _1d_pos_enc(self, mask, dim):
        temp = torch.arange(self.emb_dim // 2).float().to(mask.device)
        temp = self.temperature ** (2 * (temp.div(2, rounding_mode='floor')) / self.emb_dim)

        enc = (~mask).cumsum(dim).float().unsqueeze(-1) / temp
        enc = torch.stack([
            enc[..., 0::2].sin(), enc[..., 1::2].cos()
        ], dim=-1).flatten(-2)

        return enc

    def forward(self, bs, h, w, device):
        mask = torch.zeros(bs, h, w, dtype=torch.bool, requires_grad=False, device=device)
        x = self._1d_pos_enc(mask, dim=2)
        y = self._1d_pos_enc(mask, dim=1)

        return torch.cat([y, x], dim=3).permute(0, 3, 1, 2)
