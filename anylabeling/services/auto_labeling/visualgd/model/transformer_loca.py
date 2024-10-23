from .mlp_loca import MLP

from torch import nn


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
    ):

        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        output = src
        for layer in self.layers:
            output = layer(output, pos_emb, src_mask, src_key_padding_mask)
        return self.norm(output)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout
        )
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        if self.norm_first:
            src_norm = self.norm1(src)
            q = k = src_norm + pos_emb
            src = src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0])

            src_norm = self.norm2(src)
            src = src + self.dropout2(self.mlp(src_norm))
        else:
            q = k = src + pos_emb
            src = self.norm1(src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0]))

            src = self.norm2(src + self.dropout2(self.mlp(src)))

        return src
