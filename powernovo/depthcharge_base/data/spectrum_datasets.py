
from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dill
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from powernovo.depthcharge_base.utils.primitives import MassSpectrum
from powernovo.depthcharge_base.tokenizers.peptides import PeptideTokenizer
from powernovo.depthcharge_base.data import preprocessing
from powernovo.depthcharge_base.data.parsers import MgfParser, MzmlParser, MzxmlParser
from powernovo.depthcharge_base.utils.utils import check_positive_int, listify

logger = logging.getLogger('powernova')


class SpectrumDataset(Dataset):
    _annotated = False

    def __init__(
            self,
            ms_data_files: PathLike | Iterable[PathLike],
            ms_level: int = 2,
            preprocessing_fn: Callable | Iterable[Callable] | None = None,
            valid_charge: Iterable[int] | None = None,
            index_path: PathLike | None = None,
            overwrite: bool = False,
    ) -> None:
        self._tmpdir = None
        if index_path is None:
            # Create a random temporary file:
            self._tmpdir = TemporaryDirectory()
            index_path = Path(self._tmpdir.name) / f"{uuid.uuid4()}.hdf5"

        index_path = Path(index_path)
        if index_path.suffix not in [".h5", ".hdf5"]:
            index_path = Path(index_path)
            index_path = Path(str(index_path) + ".hdf5")

        self._path = index_path
        self._ms_level = check_positive_int(ms_level, "ms_level")
        self._valid_charge = valid_charge
        self._overwrite = bool(overwrite)
        self._handle = None
        self._file_offsets = np.array([0])
        self._file_map = {}
        self._locs = {}
        self._offsets = None

        if preprocessing_fn is not None:
            self._preprocessing_fn = listify(preprocessing_fn)
        else:
            self._preprocessing_fn = [
                preprocessing.set_mz_range(min_mz=140),
                preprocessing.filter_intensity(max_num_peaks=200),
                preprocessing.scale_intensity(scaling="root"),
                preprocessing.scale_to_unit_norm,
            ]

        # Create the file if it doesn't exist.
        if not self.path.exists() or self.overwrite:
            with h5py.File(self.path, "w") as index:
                index.attrs["ms_level"] = self.ms_level
                index.attrs["n_spectra"] = 0
                index.attrs["n_peaks"] = 0
                index.attrs["annotated"] = self.annotated
                index.attrs["preprocessing_fn"] = _hash_obj(
                    tuple(self.preprocessing_fn)
                )
        else:
            self._validate_index()

        # Now parse spectra.
        if ms_data_files is not None:
            ms_data_files = listify(ms_data_files)
            for ms_file in ms_data_files:
                self.add_file(ms_file)

    def _reindex(self) -> None:
        """Update the file mappings and offsets."""
        offsets = [0]
        for idx in range(len(self._handle)):
            grp = self._handle[str(idx)]
            offsets.append(grp.attrs["n_spectra"])
            self._file_map[grp.attrs["path"]] = idx

        self._file_offsets = np.cumsum([0] + offsets)

        # Build a map of 1D indices to 2D locations:
        grp_idx = 0
        for lin_idx in range(offsets[-1]):
            grp_idx += lin_idx >= offsets[grp_idx + 1]
            row_idx = lin_idx - offsets[grp_idx]
            self._locs[lin_idx] = (grp_idx, row_idx)

        self._offsets = None
        _ = self.offsets  # Reinitialize the offsets.

    def _validate_index(self) -> None:
        """Validate that the index is appropriate for this dataset."""
        preproc_hash = _hash_obj(tuple(self.preprocessing_fn))
        with self:
                try:
                    assert self._handle.attrs["ms_level"] == self.ms_level
                    assert self._handle.attrs["preprocessing_fn"] == preproc_hash
                    if self._annotated:
                        assert self._handle.attrs["annotated"]
                except (KeyError, AssertionError):
                    raise ValueError(
                        f"'{self.path}' already exists, but was created with "
                        "incompatible parameters. Use 'overwrite=True' to "
                        "overwrite it."
                    )

                self._reindex()

    def _get_parser(
            self,
            ms_data_file: PathLike,
    ) -> MzmlParser | MzxmlParser | MgfParser:
        kw_args = {
            "ms_level": self.ms_level,
            "valid_charge": self.valid_charge,
            "preprocessing_fn": self.preprocessing_fn,
        }

        if ms_data_file.suffix.lower() == ".mzml":
            return MzmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mzxml":
            return MzxmlParser(ms_data_file, **kw_args)

        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(ms_data_file, **kw_args)

        raise ValueError("Only mzML, mzXML, and MGF files are supported.")

    def _assemble_metadata(
            self,
            parser: MzmlParser | MzxmlParser | MgfParser,
    ) -> np.ndarray:

        meta_types = [
            ("precursor_mz", np.float32),
            ("precursor_charge", np.uint8),
            ("offset", np.uint64),
            ("scan_id", np.uint32),
        ]

        metadata = np.empty(parser.n_spectra, dtype=meta_types)
        metadata["precursor_mz"] = parser.precursor_mz
        metadata["precursor_charge"] = parser.precursor_charge
        metadata["offset"] = parser.offset
        metadata["scan_id"] = parser.scan_id
        return metadata

    def add_file(self, ms_data_file: PathLike) -> None:
        ms_data_file = Path(ms_data_file)
        if str(ms_data_file) in self._file_map:
            return

        # Invalidate current offsets:
        self._offsets = None

        # Read the file:
        parser = self._get_parser(ms_data_file)

        parser.read()



        # Create the tables:
        metadata = self._assemble_metadata(parser)

        spectrum_types = [
            ("mz_array", np.float64),
            ("intensity_array", np.float32),
        ]

        spectra = np.zeros(parser.n_peaks, dtype=spectrum_types)
        spectra["mz_array"] = parser.mz_arrays
        spectra["intensity_array"] = parser.intensity_arrays

        # Write the tables:
        with h5py.File(self.path, "a") as index:
            group_index = len(index)
            group = index.create_group(str(group_index))
            group.attrs["path"] = str(ms_data_file)
            group.attrs["n_spectra"] = parser.n_spectra
            group.attrs["n_peaks"] = parser.n_peaks
            group.attrs["id_type"] = parser.id_type

            # Update overall stats:
            index.attrs["n_spectra"] += parser.n_spectra
            index.attrs["n_peaks"] += parser.n_peaks

            # Add the datasets:
            group.create_dataset(
                "metadata",
                data=metadata,
            )

            group.create_dataset(
                "spectra",
                data=spectra,
            )

            try:
                group.create_dataset(
                    "annotations",
                    data=parser.annotations,
                    dtype=h5py.string_dtype(),
                )
            except (KeyError, AttributeError, TypeError):
                pass

            self._file_map[str(ms_data_file)] = group_index
            end_offset = self._file_offsets[-1] + parser.n_spectra
            self._file_offsets = np.append(self._file_offsets, [end_offset])

            # Update the locations:
            grp_idx = len(self._file_offsets) - 2
            for row_idx in range(parser.n_spectra):
                lin_idx = row_idx + self._file_offsets[-2]
                self._locs[lin_idx] = (grp_idx, row_idx)

    def get_spectrum(self, idx: int) -> MassSpectrum:
        group_index, row_index = self._locs[idx]
        if self._handle is None:
            raise RuntimeError("Use the context manager for access.")

        grp = self._handle[str(group_index)]
        metadata = grp["metadata"]
        spectra = grp["spectra"]
        offsets = self.offsets[str(group_index)][row_index: row_index + 2]

        start_offset = offsets[0]
        if offsets.shape[0] == 2:
            stop_offset = offsets[1]
        else:
            stop_offset = spectra.shape[0]

        spectrum = spectra[start_offset:stop_offset]
        precursor = metadata[row_index]
        return MassSpectrum(
            filename=grp.attrs["path"],
            scan_id=f"{grp.attrs['id_type']}={metadata[row_index]['scan_id']}",
            mz=np.array(spectrum["mz_array"]),
            intensity=np.array(spectrum["intensity_array"]),
            precursor_mz=precursor["precursor_mz"],
            precursor_charge=precursor["precursor_charge"],
        )

    def get_spectrum_id(self, idx: int) -> tuple[str, str]:
        group_index, row_index = self._locs[idx]
        if self._handle is None:
            raise RuntimeError("Use the context manager for access.")

        grp = self._handle[str(group_index)]
        ms_data_file = grp.attrs["path"]
        identifier = grp["metadata"][row_index]["scan_id"]
        prefix = grp.attrs["id_type"]
        return ms_data_file, f"{prefix}={identifier}"

    def loader(self, *args: tuple, **kwargs: dict) -> DataLoader:
        return DataLoader(self, *args, collate_fn=self.collate_fn, **kwargs)

    def __len__(self) -> int:
        return self.n_spectra

    def __del__(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def __getitem__(self, idx: int) -> MassSpectrum:
        if self._handle is None:
            with self:
                return self.get_spectrum(idx)

        return self.get_spectrum(idx)

    def __enter__(self) -> SpectrumDataset:
        if self._handle is None:
            self._handle = h5py.File(
                self.path,
                "r",
                rdcc_nbytes=int(3e8),
                rdcc_nslots=1024000,
            )
        return self

    def __exit__(self, *args: str) -> None:
        self._handle.close()
        self._handle = None

    @property
    def ms_files(self) -> list[str]:
        return list(self._file_map.keys())

    @property
    def path(self) -> Path:
        return self._path

    @property
    def ms_level(self) -> int:
        return self._ms_level

    @property
    def preprocessing_fn(self) -> list[Callable]:
        return self._preprocessing_fn

    @property
    def valid_charge(self) -> list[int]:
        return self._valid_charge

    @property
    def annotated(self) -> bool:
        return self._annotated

    @property
    def overwrite(self) -> bool:
        return self._overwrite

    @property
    def n_spectra(self) -> int:
        if self._handle is None:
            with self:
                return self._handle.attrs["n_spectra"]

        return self._handle.attrs["n_spectra"]

    @property
    def n_peaks(self) -> int:
        if self._handle is None:
            with self:
                return self._handle.attrs["n_peaks"]

        return self._handle.attrs["n_peaks"]

    @property
    def offsets(self) -> dict[str, np.array]:
        if self._offsets is not None:
            return self._offsets

        self._offsets = {
            k: v["metadata"]["offset"] for k, v in self._handle.items()
        }

        return self._offsets

    @staticmethod
    def collate_fn(
            batch: Iterable[MassSpectrum],
    ) -> tuple[torch.Tensor, torch.Tensor, np.array(str) | None]:
        return _collate_fn(batch)


class AnnotatedSpectrumDataset(SpectrumDataset):
    _annotated = True

    def __init__(
            self,
            tokenizer: PeptideTokenizer,
            add_stop_token=False,
            ms_data_files: PathLike | Iterable[PathLike] = None,
            ms_level: int = 2,
            preprocessing_fn: Callable | Iterable[Callable] | None = None,
            valid_charge: Iterable[int] | None = None,
            index_path: PathLike | None = None,
            overwrite: bool = False,

    ) -> None:
        self.tokenizer = tokenizer
        self. add_stop_token = add_stop_token
        super().__init__(
            ms_data_files=ms_data_files,
            ms_level=ms_level,
            preprocessing_fn=preprocessing_fn,
            valid_charge=valid_charge,
            index_path=index_path,
            overwrite=overwrite,
        )

    def _get_parser(self, ms_data_file: str) -> MgfParser:
        if ms_data_file.suffix.lower() == ".mgf":
            return MgfParser(
                ms_data_file,
                ms_level=self.ms_level,
                annotations=True,
                preprocessing_fn=self.preprocessing_fn,
            )

        raise ValueError("Only MGF files are currently supported.")

    def get_spectrum(self, idx: int) -> MassSpectrum:
        try:
            spectrum = super().get_spectrum(idx)
            group_index, row_index = self._locs[idx]
            grp = self._handle[str(group_index)]
            annotations = grp["annotations"]
            spectrum.label = annotations[row_index].decode()
        except KeyError:
            return None
        return spectrum

    @property
    def annotations(self) -> np.ndarray[str]:
        annotations = []
        for grp in self._handle.values():
            try:
                annotations.append(grp["annotations"])
            except KeyError:
                pass

        return np.concatenate(annotations)

    def collate_fn(
            self,
            batch: Iterable[MassSpectrum],
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray[str], np.ndarray[str]]:
        spectra, precursors, annotations_, usi = _collate_fn(batch)

        if spectra is None:
            return None, None, None

        tokens, excluded_ids = self.tokenizer.tokenize(annotations_, add_stop=self.add_stop_token)

        if tokens is None:
            return None, None, None

        if excluded_ids:
            #   print(f'Remove {annotations_} {excluded_ids}')
            spectra = self.__remove_item(spectra, excluded_ids)
            precursors = self.__remove_item(precursors, excluded_ids)

        assert spectra.size(0) == precursors.size(0) == tokens.size(0)
        return spectra, precursors, tokens, usi

    @staticmethod
    def __remove_item(batch, excluded_ids):
        batch_size = batch.size(0)
        batch_ids = set(range(batch_size))
        batch_ids = batch_ids - set(excluded_ids)
        batch = batch[list(batch_ids)]
        return batch


def _collate_fn(
        batch: Iterable[MassSpectrum],
) -> tuple[torch.Tensor, torch.Tensor, list[str | None], list[str | None]]:
    spectra = []
    precursor_masses = []
    precursor_charges = []
    precursor_mz = []
    annotations = []
    usi = []

    for spec in batch:
        if spec is None:
            return None, None, None

        spectra.append(spec.to_tensor())
        precursor_masses.append(spec.precursor_mass)
        precursor_charges.append(spec.precursor_charge)
        precursor_mz.append(spec.precursor_mz)
        annotations.append(spec.label)
        usi.append(spec.scan_id)

    precursors = torch.vstack([torch.tensor(precursor_masses),
                               torch.tensor(precursor_charges),
                               torch.tensor(precursor_mz),
                               ])
    spectra = torch.nn.utils.rnn.pad_sequence(
        spectra,
        batch_first=True,
    )
    return spectra, precursors.T.float(), annotations, usi


def _hash_obj(obj: Any) -> str:
    out = hashlib.sha1()
    out.update(dill.dumps(obj))
    return out.hexdigest()
