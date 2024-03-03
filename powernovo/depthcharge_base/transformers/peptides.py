from typing import Union

import torch

from powernovo.depthcharge_base.encoders.sinusoidal import FloatEncoder, PositionalEncoder
from powernovo.depthcharge_base.tokenizers.peptides import PeptideTokenizer
from powernovo.depthcharge_base.utils.feedforward import FeedForward


class _PeptideTransformer(torch.nn.Module):
    def __init__(
            self,
            n_tokens: Union[int, PeptideTokenizer],
            d_model: int,
            positional_encoder: Union[PositionalEncoder, bool],
            max_charge: int,
    ) -> None:
        super().__init__()
        try:
            n_tokens = len(n_tokens)
        except TypeError:
            pass

        if callable(positional_encoder):
            self.positional_encoder = positional_encoder
        elif positional_encoder:
            self.positional_encoder = PositionalEncoder(d_model)
        else:
            self.positional_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge + 1, d_model)
        self.aa_encoder = torch.nn.Embedding(
            n_tokens + 1,
            d_model,
            padding_idx=0,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class PeptideTransformerEncoder(_PeptideTransformer):
    def __init__(
            self,
            n_tokens: Union[int, PeptideTokenizer],
            d_model: int = 128,
            nhead: int = 8,
            dim_feedforward: int = 1024,
            n_layers: int = 1,
            dropout: float = 0,
            positional_encoder: Union[PositionalEncoder, bool] = True,
            max_charge: int = 5,
    ) -> None:
        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            positional_encoder=positional_encoder,
            max_charge=max_charge,
        )

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
            self,
            tokens: torch.Tensor,
            charges: torch.Tensor,
    ) -> tuple:
        encoded = self.aa_encoder(tokens)
        charges = self.charge_encoder(charges)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).long()

        encoded = self.positional_encoder(encoded)

        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


class PeptideTransformerDecoder(_PeptideTransformer):
    def __init__(
            self,
            n_tokens: Union[int, PeptideTokenizer],
            tokenizer=None,
            d_model: int = 128,
            nhead: int = 8,
            dim_feedforward: int = 1024,
            n_layers: int = 1,
            dropout: float = 0,
            positional_encoder: Union[PositionalEncoder, bool] = True,
            max_charge: int = 5,
    ) -> None:
        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            positional_encoder=positional_encoder,
            max_charge=max_charge,
        )
        self.mass_encoder = FloatEncoder(d_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = FeedForward(
            in_features=d_model,
            layers=2,
            out_features=self.aa_encoder.num_embeddings,
        )

        self.tokenizer = tokenizer

    def forward(
            self,
            tokens: Union[torch.Tensor, None],
            precursors: torch.Tensor,
            memory: torch.Tensor,
            memory_key_padding_mask: torch.Tensor,
    ) -> tuple:
        if tokens is None:
            tokens = torch.tensor([[] for _ in range(memory.size(0))], dtype=torch.long).to(self.device)

        tokens = self.aa_encoder(tokens)
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        tgt = torch.cat([precursors, tokens], dim=1)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.positional_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(self.device)

        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )

        return self.final(preds)


def generate_tgt_mask(sz: int) -> torch.Tensor:
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
