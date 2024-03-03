from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn.functional import one_hot

from powernovo.depthcharge_base.tokenizers.peptides import PeptideTokenizer
from powernovo.depthcharge_base.utils.constants import PRECURSOR_DIM, C13, H2O, MASS_SCALE, MIN_PEPTIDE_LEN
from powernovo.models.spectrum.spectrum_inference import SpectrumTransformer


@dataclass
class HypothesesBeam(object):
    sequences: torch.LongTensor
    scores: torch.FloatTensor
    remaining_masses: torch.LongTensor
    precursor_mass_charge: torch.FloatTensor
    spectrum_encoding: torch.FloatTensor
    spectrum_mask: torch.BoolTensor

    def is_empty(self) -> bool:
        if self.sequences is None:
            return True
        else:
            return False


@dataclass
class CompletedHypotheses(object):
    sequence: list[str]
    mass_error: float
    score: float
    aa_scores: list
    solved: bool


class AdaptiveBeamSearchDecoder(object):
    def __init__(self,
                 model: SpectrumTransformer,
                 tokenizer: PeptideTokenizer,
                 device: torch.device,
                 mass_scale: int = MASS_SCALE,
                 min_peptide_len: int = MIN_PEPTIDE_LEN
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.mass_scale = mass_scale
        self.device = device
        self.residue_masses = torch.zeros(len(self.tokenizer) + 1, dtype=torch.int64).to(self.device)
        self.min_peptide_len = min_peptide_len

        for k, v in self.tokenizer.index.items():
            if k in self.tokenizer.residues:
                mass = round(self.tokenizer.residues[k] * MASS_SCALE)
                self.residue_masses[v] = mass

        self.aa_n_term = torch.as_tensor(self.tokenizer.get_n_term_aa(), dtype=torch.long, device=self.device)
        self.special_tokens = torch.as_tensor([self.tokenizer.stop_int, self.tokenizer.start_int, 0],
                                              device=self.device)

    def detokenize(self, tokens: torch.Tensor, join: bool = True) -> str:
        try:
            tokens = self.post_filter(tokens)
            if len(tokens) < self.min_peptide_len:
                return ''
            sequence = self.tokenizer.detokenize(tokens.unsqueeze(0), join=join)[0]
        except (AttributeError, KeyError, Exception):
            return ''
        return sequence

    def __calc_mod_mask(self, tokens: torch.Tensor) -> np.ndarray:
        mod_mask = torch.isin(tokens, self.aa_n_term)
        return mod_mask.cpu().numpy()

    def post_filter(self, tokens) -> torch.Tensor:
        mask = torch.isin(tokens, self.special_tokens)
        special_indices = torch.where(mask)[0]
        mod_mask = torch.isin(tokens.__reversed__(), self.aa_n_term)

        if len(special_indices) == 0 and mod_mask[0]:
            tokens[-1] = self.tokenizer.stop_int
        elif len(special_indices) == 1 and mod_mask[1]:
            tokens[-2] = self.tokenizer.stop_int
            tokens = tokens[:-2]
        elif len(special_indices) > 1:
            tokens = tokens[:special_indices[0]]
            if len(tokens) < self.min_peptide_len:
                return torch.tensor([])
            i = 0
            mod_mask = torch.isin(tokens.__reversed__(), self.aa_n_term)
            while mod_mask[i]:
                i += 1
            tokens = tokens[:len(tokens) - i]

        mask = torch.isin(tokens, self.special_tokens)
        tokens = tokens[~mask]

        return tokens

    @staticmethod
    def unravel_index(
            indices: torch.LongTensor, outer_dim: int
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        rows = indices.div(outer_dim, rounding_mode="floor")
        columns = indices.remainder(outer_dim)
        return rows, columns

    def expand_hypotheses(
            self, hypotheses_beam: HypothesesBeam,
            residue_masses: torch.LongTensor
    ) -> torch.FloatTensor:
        assert hypotheses_beam.remaining_masses is not None
        remaining_masses = hypotheses_beam.remaining_masses.unsqueeze(-1) - residue_masses.unsqueeze(
            0
        ).unsqueeze(0)

        assert hypotheses_beam.sequences is not None
        sequence_length = hypotheses_beam.sequences.shape[-1]
        spectrum_length = hypotheses_beam.spectrum_encoding.shape[2]
        hidden_dim = hypotheses_beam.spectrum_encoding.shape[3]
        scores = self.model.decode(
            hypotheses_beam.sequences.reshape(-1, sequence_length),
            hypotheses_beam.precursor_mass_charge.reshape(-1, PRECURSOR_DIM),
            hypotheses_beam.spectrum_encoding.reshape(-1, spectrum_length, hidden_dim),
            hypotheses_beam.spectrum_mask.reshape(-1, spectrum_length),
        )

        assert hypotheses_beam.scores is not None
        batch_size = hypotheses_beam.scores.shape[0]
        beam_size = hypotheses_beam.scores.shape[1]
        candidate_scores = scores.reshape(
            batch_size, beam_size, -1
        ) + hypotheses_beam.scores.unsqueeze(-1)
        return candidate_scores, remaining_masses

    def filter_items(
            self,
            beam_state: HypothesesBeam,
            scores: torch.FloatTensor,
            remaining_masses: torch.LongTensor,
            mass_buffer: torch.LongTensor,
            max_isotope: int,
    ) -> tuple[list[list[CompletedHypotheses]], HypothesesBeam]:
        assert beam_state.remaining_masses is not None
        reshaped_mass_buffer = mass_buffer.unsqueeze(-1).unsqueeze(-1)

        batch_size, beam_size, num_residues = scores.shape

        completed_items: list[list[CompletedHypotheses]] = [[] for _ in range(batch_size)]

        item_is_complete = (reshaped_mass_buffer >= remaining_masses) & (
                remaining_masses >= -reshaped_mass_buffer
        )
        is_finish = (
            one_hot(
                torch.tensor([self.tokenizer.stop_int])
                .unsqueeze(0)
                .expand(batch_size, beam_size),
                num_classes=num_residues,
            )
            .bool()
            .to(scores.device)
        )

        if max_isotope > 0:
            for num_isotopes in range(1, max_isotope + 1):
                isotope_is_complete = (
                                              reshaped_mass_buffer
                                              >= remaining_masses - num_isotopes * round(self.mass_scale * C13)
                                      ) & (
                                              remaining_masses - num_isotopes * round(self.mass_scale * C13)
                                              >= -reshaped_mass_buffer
                                      )
                item_is_complete = item_is_complete | isotope_is_complete

        item_is_complete = item_is_complete & ~is_finish & scores.isfinite()

        local_variables = zip(
            item_is_complete, remaining_masses, scores, beam_state.sequences
        )

        for batch, (is_complete, mass_errors, local_scores, sequences) in enumerate(
                local_variables
        ):
            if is_complete.any().item():
                beam_index, residues = torch.where(is_complete)
                completed_sequences = torch.column_stack((sequences[beam_index], residues))

                eos_scores = self.model.decode(
                    completed_sequences,
                    beam_state.precursor_mass_charge[batch, beam_index],
                    beam_state.spectrum_encoding[batch, beam_index],
                    beam_state.spectrum_mask[batch, beam_index],
                )
                completed_scores = (
                        local_scores[beam_index, residues]
                        + eos_scores[:, self.tokenizer.stop_int]
                )
                completed_mass_errors = mass_errors[beam_index, residues]
                completed_items[batch].extend(
                    CompletedHypotheses(
                        sequence=self.detokenize(sequence),
                        solved=False,
                        aa_scores=[],
                        mass_error=mass_error.item() / self.mass_scale,
                        score=score,
                    )
                    for sequence, mass_error, score in zip(
                        completed_sequences,
                        completed_mass_errors,
                        completed_scores.tolist(),
                    )
                )

        scores = self.filter_hypotheses(
            scores=scores,
            remaining_masses=remaining_masses,
            mass_buffer=reshaped_mass_buffer,
        )

        beam_scores, beam_indices = scores.reshape(batch_size, -1).topk(
            k=beam_size
        )

        beam_sequences = self._append_next_token(
            indices=beam_indices, outer_dim=num_residues, sequences=beam_state.sequences
        )
        remaining_masses = remaining_masses.reshape(batch_size, -1)
        beam_remaining_masses = []
        for local_remaining_masses, local_indices in zip(remaining_masses, beam_indices):
            beam_remaining_masses.append(local_remaining_masses[local_indices])
        beam_remaining_masses = torch.stack(beam_remaining_masses)
        new_beam = HypothesesBeam(
            sequences=beam_sequences,
            scores=beam_scores,
            remaining_masses=beam_remaining_masses,
            precursor_mass_charge=beam_state.precursor_mass_charge,
            spectrum_encoding=beam_state.spectrum_encoding,
            spectrum_mask=beam_state.spectrum_mask,
        )
        return completed_items, new_beam

    def filter_hypotheses(
            self,
            scores: torch.FloatTensor,
            remaining_masses: torch.LongTensor,
            mass_buffer: torch.LongTensor,
    ) -> torch.FloatTensor:

        scores[:, :, self.tokenizer.stop_int] = -float("inf")
        scores[:, :, self.tokenizer.start_int] = -float("inf")
        scores[:, :, 0] = -float("inf")
        mass_is_invalid = remaining_masses < -mass_buffer
        scores[mass_is_invalid] = -float("inf")

        return scores

    def init_beam(
            self,
            spectra: torch.FloatTensor,
            precursor_mass_charge: torch.FloatTensor,
            residue_masses: torch.LongTensor,
            beam_size: int,
    ) -> HypothesesBeam:

        (spectrum_encoding, spectrum_mask), scores = self.model.init_beam(
            spectrum=spectra, precursors=precursor_mass_charge
        )

        precursor_masses = (
            torch.round(
                self.mass_scale * precursor_mass_charge[:, 0]
            )
            .type(torch.int64)
            .to(spectra.device)
        )
        precursor_masses = precursor_masses - round(self.mass_scale * H2O)

        beam_scores = scores.topk(k=beam_size)
        beam_masses = residue_masses.to(spectra.device).gather(-1, beam_scores.indices)
        remaining_masses = precursor_masses.unsqueeze(-1) - beam_masses

        beam_precursor_mass_charge = precursor_mass_charge.unsqueeze(1).expand(-1, beam_size, -1)
        beam_spectrum_encoding = spectrum_encoding.unsqueeze(1).expand(-1, beam_size, -1, -1)
        beam_spectrum_mask = spectrum_mask.unsqueeze(1).expand(-1, beam_size, -1)
        return HypothesesBeam(
            sequences=beam_scores.indices.unsqueeze(-1),
            scores=beam_scores.values,
            remaining_masses=remaining_masses,
            precursor_mass_charge=beam_precursor_mass_charge,
            spectrum_encoding=beam_spectrum_encoding,
            spectrum_mask=beam_spectrum_mask,
        )

    def decode(  # type:ignore
            self,
            spectra: torch.FloatTensor,
            precursors: torch.FloatTensor,
            beam_size: int,
            max_length: int,
            mass_tolerance: float = 5e-5,
            max_isotope: int = 1,
            return_all_beams: bool = False,
    ) -> list[Any]:
        with torch.no_grad():
            batch_size = spectra.shape[0]
            complete_items: list[list[CompletedHypotheses]] = [[] for _ in range(batch_size)]

            num_residues = self.residue_masses.shape[-1]
            mass_buffers = (
                (
                        self.mass_scale
                        * mass_tolerance
                        * precursors[:, 0]
                )
                .round()
                .long()
            )

            beam: HypothesesBeam = self.init_beam(
                spectra=spectra,
                precursor_mass_charge=precursors,
                residue_masses=self.residue_masses.unsqueeze(0).expand(batch_size, num_residues),
                beam_size=beam_size,
            )

            for _ in range(max_length):
                if beam.is_empty():
                    break

                assert beam.scores is not None
                if beam.scores.isinf().all():
                    break

                scores, remaining_masses = self.expand_hypotheses(
                    hypotheses_beam=beam, residue_masses=self.residue_masses
                )

                complete_candidates, beam = self.filter_items(
                    scores=scores,
                    beam_state=beam,
                    remaining_masses=remaining_masses,
                    mass_buffer=mass_buffers,
                    max_isotope=max_isotope,
                )

                for i, items in enumerate(complete_candidates):
                    complete_items[i].extend(items)

            for items in complete_items:
                items.sort(key=lambda item: item.score, reverse=True)

            if not return_all_beams:
                sequences = [items[0] if len(items) > 0 else [] for items in complete_items]
            else:
                sequences = [items if len(items) > 0 else [] for items in complete_items]

            return sequences

    def _append_next_token(
            self,
            indices: torch.LongTensor,
            outer_dim: int,
            sequences: torch.LongTensor,
    ) -> torch.LongTensor:
        beam_items, residues = self.unravel_index(indices, outer_dim)

        collected_sequences = []
        for beam_item, residue, sequence in zip(beam_items, residues, sequences):
            last_aa_n_term = torch.isin(sequence[beam_item][:, -1], self.aa_n_term)
            next_aa_n_term = torch.isin(residue, self.aa_n_term)
            discard = last_aa_n_term & next_aa_n_term
            residue[discard] = self.tokenizer.stop_int
            sequence[beam_item][:, -1] = 0
            collected_sequences.append(torch.column_stack((sequence[beam_item], residue)))
        return torch.stack(collected_sequences, 0)
