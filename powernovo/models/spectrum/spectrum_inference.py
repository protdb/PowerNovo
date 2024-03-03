from typing import Tuple, List

import torch.nn as nn
import torch
from powernovo.pipeline_config.config import PWNConfig
from powernovo.depthcharge_base.transformers.peptides import PeptideTransformerDecoder
from powernovo.depthcharge_base.transformers.spectra import SpectrumTransformerEncoder


class SpectrumTransformer(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        config = PWNConfig()
        dmodel = config.spectrum_transformer.dmodel
        n_layers = config.spectrum_transformer.n_layers
        max_charge = config.spectrum_transformer.max_charge
        self.n_tokens = config.spectrum_transformer.n_tokens
        self.beam_size = config.hypotheses.beam_size
        self.device = device

        self.spectrum_encoder = SpectrumTransformerEncoder(
            d_model=dmodel,
            n_layers=n_layers,
        )

        self.peptide_decoder = PeptideTransformerDecoder(
            n_tokens=self.n_tokens,
            d_model=dmodel,
            n_layers=n_layers,
            max_charge=max_charge
        )

    def init_beam(self, spectrum: torch.FloatTensor, precursors: torch.Tensor):
        with torch.no_grad():
            batch_size = spectrum.size(0)
            spectrum_embedding, memory_mask = self.spectrum_encoder(spectrum)
            start_token = torch.LongTensor([[]]).to(self.device)
            start_token = start_token.repeat(batch_size, 1)
            decoder_output = self.peptide_decoder(
                tokens=start_token,
                precursors=precursors,
                memory=spectrum_embedding,
                memory_key_padding_mask=memory_mask
            )

            scores = decoder_output[:, -1, :]  # (s, vocab_size)
            scores = torch.softmax(scores, dim=-1)
        return (spectrum_embedding, memory_mask), scores

    def decode(self,
               tokens: torch.Tensor,
               precursors: torch.Tensor,
               spectrum_embedding: torch.FloatTensor,
               memory_mask: torch.FloatTensor):
        decoder_output = self.peptide_decoder(
            tokens=tokens,
            precursors=precursors,
            memory=spectrum_embedding,
            memory_key_padding_mask=memory_mask)

        scores = decoder_output[:, -1, :]  # (s, vocab_size)
        scores = torch.softmax(scores, dim=-1)
        return scores

    def embed_spectrum(self, batch: List[torch.Tensor], max_peaks: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            spectrum = batch[0].float()
            spectrum_embedding, memory_mask = self.spectrum_encoder(spectrum)
        return spectrum_embedding, memory_mask

    def decoder_step(self,
                     tokens: torch.Tensor,
                     precursors: torch.Tensor,
                     encoder_memory: torch.Tensor,
                     memory_mask: torch.Tensor,
                     hypotheses: torch.Tensor,
                     hypotheses_scores: torch.Tensor,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder = self.peptide_decoder

        with torch.no_grad():
            decoder_output = decoder(
                tokens=tokens,
                precursors=precursors,
                memory=encoder_memory,
                memory_key_padding_mask=memory_mask
            )

        scores = decoder_output[:, -1, :]  # (s, vocab_size)
        scores = torch.softmax(scores, dim=-1)
        scores = hypotheses_scores.unsqueeze(1) + scores
        top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(self.beam_size, 0, True, True)  # (k)

        # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
        prev_word_indices = unrolled_indices // (self.n_tokens + 1)  # (k)
        next_word_indices = unrolled_indices % (self.n_tokens + 1)  # (k)

        top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                     dim=1)

        return top_k_hypotheses, top_k_hypotheses_scores
