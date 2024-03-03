import re
from itertools import groupby
from typing import Tuple

from Bio.PDB.Polypeptide import aa1
from powernovo.depthcharge_base.utils.primitives import MASSIVE_KB_MOD_MAP


def to_canonical(seq: str) -> str:
    canonical_seq = re.sub(r"\[.*?\]", '', seq)
    canonical_seq = ''.join([s for s in canonical_seq if s in aa1])
    canonical_seq = canonical_seq.replace('-', '')
    return canonical_seq


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def from_proforma(sequence: str) -> Tuple[str, dict]:
    sequence_mod = []
    mod_dict = {}
    for k, v in MASSIVE_KB_MOD_MAP.items():
        sequence = sequence.replace(k, v).strip()
        if v:
            if v in sequence:
                sequence_mod.append(v)

    if sequence_mod:
        for mod in sequence_mod:
            mod_pos = [i for i in range(len(sequence)) if sequence.startswith(mod, i)]
            mod_dict.update({mod: mod_pos})
    return sequence, mod_dict


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
