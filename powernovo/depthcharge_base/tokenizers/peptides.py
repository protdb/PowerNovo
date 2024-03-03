"""Tokenizers for peptides."""
from __future__ import annotations

import re
from collections.abc import Iterable

import numba as nb
import numpy as np
import torch
from numba.typed.typedlist import List
from pyteomics.proforma import GenericModification, MassModification

from powernovo.depthcharge_base.utils import utils
from powernovo.depthcharge_base.utils.constants import H2O, PROTON
from powernovo.depthcharge_base.utils.primitives import MASSIVE_KB_MOD, Peptide, PeptideIons
from powernovo.depthcharge_base.tokenizers.tokenizer import Tokenizer


class PeptideTokenizer(Tokenizer):

    residues = nb.typed.Dict.empty(
        nb.types.unicode_type,
        nb.types.float64,
    )
    residues.update(
        G=57.021463735,
        A=71.037113805,
        S=87.032028435,
        P=97.052763875,
        V=99.068413945,
        T=101.047678505,
        C=103.009184505,
        L=113.084064015,
        I=113.084064015,
        N=114.042927470,
        D=115.026943065,
        Q=128.058577540,
        K=128.094963050,
        E=129.042593135,
        M=131.040484645,
        H=137.058911875,
        F=147.068413945,
        R=156.101111050,
        Y=163.063328575,
        W=186.079312980,
    )

    # The peptide parsing function:
    _parse_peptide = Peptide.from_proforma

    def __init__(
            self,
            residues: dict[str, float] | None = None,
            replace_isoleucine_with_leucine: bool = False,
            reverse: bool = False,
    ) -> None:
        """Initialize a PeptideTokenizer."""
        self.replace_isoleucine_with_leucine = replace_isoleucine_with_leucine
        self.reverse = reverse
        self.residues = self.residues.copy()
        if residues is not None:
            self.residues.update(residues)

        if self.replace_isoleucine_with_leucine:
            del self.residues["I"]

        super().__init__(list(self.residues.keys()))

    def split(self, sequence: str) -> list[str]:
        """Split a ProForma peptide sequence.

        Parameters
        ----------
        sequence : str
            The peptide sequence.

        Returns
        -------
        list[str]
            The tokens that compprise the peptide sequence.
        """

        pep = self._parse_peptide(sequence)
        if self.replace_isoleucine_with_leucine:
            pep.sequence = pep.sequence.replace("I", "L")

        pep = pep.split()
        if self.reverse:
            pep.reverse()

        return pep

    def ions(  # noqa: C901
            self,
            sequences: str,
            precursor_charges: Iterable[int] | str,
            max_fragment_charge: int | None = None,
    ) -> tuple[torch.Tensor[float], list[torch.Tensor[float]]]:
        sequences = utils.listify(sequences)
        if max_fragment_charge is None:
            max_fragment_charge = np.inf

        if precursor_charges is None:
            precursor_charges = [None] * len(sequences)
        else:
            precursor_charges = utils.listify(precursor_charges)

        if len(sequences) != len(precursor_charges):
            raise ValueError(
                "The number of sequences and precursor charges did not match."
            )

        out = []
        for seq, charge in zip(sequences, precursor_charges):
            if isinstance(seq, str):
                if self.replace_isoleucine_with_leucine:
                    seq = seq.replace("I", "L")

                try:
                    pep = Peptide.from_proforma(seq)
                except ValueError:
                    pep = Peptide.from_massivekb(seq)

                tokens = pep.split()
                if charge is None:
                    charge = max(pep.charge - 1, 1)
            else:
                tokens = seq

            if charge is None:
                raise ValueError(
                    f"No charge was provided for {seq}",
                )

            try:
                prec = calculate_mass(
                    nb.typed.List(tokens),
                    charge,
                    self.residues,
                )
            except KeyError as err:
                raise ValueError(
                    f"Unrecognized token(s) in {''.join(tokens)}"
                ) from err

            frags = _calc_fragment_masses(
                nb.typed.List(tokens),
                min(charge, max_fragment_charge),
                self.residues,
            )

            ions = PeptideIons(
                tokens=tokens,
                precursor=prec,
                fragments=torch.tensor(frags),
            )

            out.append(ions)

        return out

    @classmethod
    def from_proforma(
            cls,
            sequences: Iterable[str],
            replace_isoleucine_with_leucine: bool = True,
            reverse: bool = True,
    ) -> PeptideTokenizer:
        if isinstance(sequences, str):
            sequences = [sequences]

        # Parse modifications:
        new_res = cls.residues.copy()
        for peptide in sequences:
            parsed = Peptide.from_proforma(peptide).split()
            for token in parsed:
                if token in new_res.keys():
                    continue

                if token == "-":
                    continue

                match = re.search(r"(.*)\[(.*)\]", token)
                try:
                    res, mod = match.groups()
                    if res and res != "-":
                        res_mass = new_res[res]
                    else:
                        res_mass = 0
                except (AttributeError, KeyError) as err:
                    raise ValueError("Unrecognized token {token}.") from err

                try:
                    mod = MassModification(mod)
                except ValueError:
                    mod = GenericModification(mod)

                new_res[token] = res_mass + mod.mass

        return cls(new_res, replace_isoleucine_with_leucine, reverse)

    @staticmethod
    def from_massivekb(
            replace_isoleucine_with_leucine: bool = True,
            reverse: bool = True,
    ) -> MskbPeptideTokenizer:
        return MskbPeptideTokenizer.from_proforma(
            [f"{mod}A" for mod in MASSIVE_KB_MOD.values()],
            replace_isoleucine_with_leucine,
            reverse,
        )

    def get_n_term_aa(self):
        n_term_aa = []
        for aa, mass in self.residues.items():
            aa_idx = self.index[aa]
            if aa in list(MASSIVE_KB_MOD.values()):
                n_term_aa.append(aa_idx)
        return n_term_aa



class MskbPeptideTokenizer(PeptideTokenizer):
    _parse_peptide = Peptide.from_massivekb


def calculate_mass(tokens: List,
                   charge: int,
                   masses: nb.typed.typedDict
                   ) -> float:
    tokens = nb.typed.List(tokens)
    return __calculate_mass(tokens, charge, masses)


@nb.njit
def __calculate_mass(
        tokens: List[str],
        charge: int,
        masses: nb.typed.Dict,
) -> float:

    mass = sum([masses[t] for t in tokens]) + H2O
    if charge is not None:
        mass = _mass2mz(mass, charge)

    return mass


@nb.njit
def _calc_fragment_masses(
        tokens: list[str],
        charge: int,
        masses: nb.typed.Dict,
) -> np.ndarray[float]:

    seq = np.empty(len(tokens))
    n_mod = False
    c_mod = False
    for idx, token in enumerate(tokens):
        if not idx and token.endswith("-"):
            n_mod = True

        if idx == (len(tokens) - 1) and token.startswith("-"):
            c_mod = True

        seq[idx] = masses[token]

    if n_mod:
        seq[1] += seq[0]
        seq = seq[1:]

    if c_mod:
        seq[-2] += seq[-1]
        seq = seq[:-1]

    # Calculate fragments:
    max_charge = min(charge, 2)
    n_ions = len(seq) - 1
    ions = np.empty((2, n_ions, max_charge))
    b_mass = 0
    y_mass = H2O
    for idx in range(n_ions):
        b_mass += seq[idx]
        y_mass += seq[-(idx + 1)]
        for cur_charge in range(1, max_charge + 1):
            z_idx = cur_charge - 1
            ions[0, idx, z_idx] = _mass2mz(b_mass, cur_charge)
            ions[1, idx, z_idx] = _mass2mz(y_mass, cur_charge)

    return ions


@nb.njit
def _mass2mz(mass: float, charge: int) -> float:
    return (mass / charge) + PROTON


def calc_mass_error(
        calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10 ** 6
