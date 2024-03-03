
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import torch
from sortedcontainers import SortedDict, SortedSet
from torch import nn

from powernovo.depthcharge_base.utils import utils


class Tokenizer(ABC):

    def __init__(self, tokens: Sequence[str], stop_token: str = "$", start_token='@') -> None:
        self.stop_token = stop_token
        self.start_token = start_token
        tokens = SortedSet(tokens)
        if (self.stop_token in tokens) or (self.start_token in tokens):
            raise ValueError(
                f"Stop/Start token {stop_token} already exists in tokens.",
            )

        tokens.add(self.stop_token)
        tokens.add(self.start_token)
        self.index = SortedDict({k: i + 1 for i, k in enumerate(tokens)})
        self.reverse_index = [None] + list(tokens)
        self.stop_int = self.index[self.stop_token]
        self.start_int = self.index[start_token]

    def __len__(self) -> int:
        """The number of tokens."""
        return len(self.index)

    @abstractmethod
    def split(self, sequence: str) -> list[str]:
        """Split a sequence into the constituent string tokens."""

    def tokenize(
            self,
            sequences: Iterable[str],
            to_strings: bool = False,
            add_stop: bool = False,
    ) -> torch.Tensor | list[list[str]]:
        try:
            out = []
            excluded_ids = []
            for idx, seq in enumerate(utils.listify(sequences)):
                try:
                    tokens = self.split(seq)
                except:
                    excluded_ids.append(idx)
                    continue

                if add_stop and tokens[-1] != self.stop_token:
                    tokens.append(self.stop_token)

                if to_strings:
                    out.append(tokens)
                    continue

                out.append(torch.tensor([self.index[t] for t in tokens]))

            if to_strings:
                return out, excluded_ids

            if isinstance(sequences, str):
                return out[0], excluded_ids

            if not out:
                return None, []

            return nn.utils.rnn.pad_sequence(out, batch_first=True), excluded_ids
        except KeyError as err:
            raise ValueError("Unrecognized token") from err

    def detokenize(
            self,
            tokens: torch.Tensor,
            join: bool = True,
            trim_stop_token: bool = True,
    ) -> list[str] | list[list[str]]:
        decoded = []
        for row in tokens:
            seq = [
                self.reverse_index[i]
                for i in row
                if self.reverse_index[i] is not None
            ]

            if trim_stop_token and seq[-1] == self.stop_token:
                seq.pop(-1)

            if join:
                seq = "".join(seq)

            decoded.append(seq)

        return decoded

