import os.path
from pathlib import Path
from typing import Union, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from powernovo.depthcharge_base.data import preprocessing
from powernovo.depthcharge_base.data.spectrum_datasets import AnnotatedSpectrumDataset, SpectrumDataset
from powernovo.depthcharge_base.tokenizers.peptides import PeptideTokenizer
from powernovo.depthcharge_base.utils.constants import MASS_SCALE
from powernovo.beam_search.adaptive_beam_search import AdaptiveBeamSearchDecoder
from powernovo.beam_search.bert_hypotheses_search import KnapsackBeamSearchDecoder
from powernovo.pipeline_config.config import PWNConfig
from powernovo.models.spectrum.spectrum_inference import SpectrumTransformer
import logging

from powernovo.peptides.peptide_aggregator import PeptideAggregator

logger = logging.getLogger("powernovo")
logger.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)


class PWNInference(object):
    def __init__(self, **kwargs):
        self.spectrum_model = None
        self.tokenizer = PeptideTokenizer.from_massivekb(reverse=False)
        self.config = PWNConfig()

        if self.config.device == 'auto':
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif self.config.device in ['cpu', 'cuda']:
            self.device = torch.device(self.config.device)
        else:
            raise AttributeError(f'Invalid device parameter in config.yaml: {self.config.device}')
        self.beam_size = self.config.hypotheses.beam_size
        self.beam_search = None

    def load_models(self):
        model_file = self.config.spectrum_transformer.trained_model
        model_path = self.config.models_folder / model_file
        try:
            assert os.path.exists(model_path), f"Error: model not found: {model_path}"
            self.spectrum_model = SpectrumTransformer(device=self.device)
            self.spectrum_model.to(self.device)
            model_state = torch.load(model_path, map_location=self.device)
            self.spectrum_model.load_state_dict(model_state['state_dict'])
            logger.info(f'Model loaded successfully {model_path}')
            logger.info(f'Use device: {self.device}')

        except (AssertionError, Exception) as e:
            logger.error(e)
            raise e

        if self.config.use_bert_inference:
            bert_model_path = self.config.models_folder / self.config.peptide_bert.trained_model
            n_gen = int(self.config.hypotheses.n_gen)
            cx_prob = float(self.config.hypotheses.cx_prob)
            lambda_ = int(self.config.hypotheses.lambda_)
            self.beam_search = KnapsackBeamSearchDecoder.from_bert_config(spectrum_model=self.spectrum_model,
                                                                          tokenizer=self.tokenizer,
                                                                          bert_model_path=bert_model_path,
                                                                          device=self.device,
                                                                          mass_scale=MASS_SCALE,
                                                                          cx_prob=cx_prob,
                                                                          n_gen=n_gen,
                                                                          lambda_=lambda_
                                                                          )
        else:
            self.beam_search = AdaptiveBeamSearchDecoder(model=self.spectrum_model,
                                                         tokenizer=self.tokenizer,
                                                         device=self.device,
                                                         mass_scale=MASS_SCALE,
                                                         )

    def run(self,
            input_file: Union[str, os.PathLike, Path],
            output_folder: Union[str, os.PathLike, Path] = None):
        annotated = self.config.annotated
        loader = self.preprocessing(input_file=input_file,
                                    annotated=annotated)
        output_filename = Path(input_file).stem
        if not os.path.exists(output_folder):
            output_folder = Path(input_file).parent / self.config.output_folder_name
            output_folder.mkdir(exist_ok=True)
            output_folder = output_folder / output_filename
            output_folder.mkdir(exist_ok=True)

        self.process_data(loader=loader,
                          output_folder=output_folder,
                          output_filename=output_filename,
                          annotated=annotated)

    def preprocessing(self, input_file: os.PathLike,
                      annotated) -> DataLoader:
        try:
            assert os.path.exists(input_file), f"Error: input file not found: {input_file}"
            preprocessing_folder = self.config.preprocessing_folder
            if not os.path.exists(preprocessing_folder):
                preprocessing_folder = Path(input_file).parent / preprocessing_folder
                preprocessing_folder.mkdir(exist_ok=True)

            preprocessing_file = f'{Path(input_file).stem}.hdf5'
            preprocessing_path = Path(preprocessing_folder) / preprocessing_file
            logger.info(f'Preprocessing input file: {input_file}')

            try:
                loader = self._create_loader(
                    input_file=input_file,
                    preprocessing_path=preprocessing_path,
                    overwrite=False,
                    annotated=annotated
                )
            except (ValueError, Exception) as e:
                logger.info('The datafile already exists, but the parameters are incompatible. Attempt to rebuild...')
                loader = self._create_loader(
                    input_file=input_file,
                    preprocessing_path=preprocessing_path,
                    overwrite=True,
                    annotated=annotated
                )

            logger.info('Preprocessing completed successfully')

        except (AssertionError, Exception) as e:
            logger.info(f'Unable to preprocessing input file: {input_file}')
            logger.error(e)
            raise e

        return loader

    def _create_loader(self,
                       input_file: os.PathLike,
                       preprocessing_path: os.PathLike,
                       overwrite: bool = False,
                       annotated: bool = False) -> DataLoader:
        try:
            if annotated:
                dataset = AnnotatedSpectrumDataset(tokenizer=self.tokenizer,
                                                   ms_data_files=input_file,
                                                   overwrite=overwrite,
                                                   index_path=preprocessing_path,
                                                   )
            else:
                dataset = SpectrumDataset(ms_data_files=input_file,
                                          overwrite=overwrite,
                                          preprocessing_fn=[
                                              preprocessing.set_mz_range(min_mz=140),
                                              preprocessing.scale_intensity(scaling="root"),
                                              preprocessing.scale_to_unit_norm,
                                          ],
                                          index_path=preprocessing_path)
        except (ValueError, Exception) as e:
            raise e

        num_workers = self.config.n_workers
        num_workers = num_workers if num_workers > 0 else os.cpu_count() // 2
        loader = dataset.loader(batch_size=int(self.config.inference_batch_size),
                                num_workers=num_workers,
                                pin_memory=True)
        return loader

    def process_data(self,
                     loader: DataLoader,
                     output_folder: Union[str, os.PathLike, Path],
                     output_filename: str,
                     annotated: bool = False):
        logger.info(f'Start processing. Total items: {len(loader)} ')

        peptide_aggregator = PeptideAggregator(output_folder=output_folder,
                                               output_filename=output_filename
                                               )

        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            if None in batch:
                continue

            scan_ids = batch[-1]
            meta_len = 2 - annotated
            batch = batch[:len(batch) - meta_len]
            batch = [b.to(self.device) for b in batch]
            precursors = batch[1]
            charges = precursors[:, 1].int()
            invalid_charges_mask = charges == 0

            if torch.sum(invalid_charges_mask, dim=-1) > 0:
                continue
            inference_records = self.inference(batch, scan_id=scan_ids)

            for i in range(len(inference_records)):
                predicted_ = inference_records[i]
                if predicted_:
                    try:
                        peptide_aggregator.add_record(
                            scan_id=scan_ids[i],
                            sequence=predicted_.sequence,
                            mass_error=predicted_.mass_error,
                            score=predicted_.score,
                            aa_scores=predicted_.aa_scores)
                    except (AttributeError, Exception):
                        continue

        peptide_aggregator.solve()

    def inference(self, batch: List[torch.Tensor], scan_id: list[str] | None) -> object:
        best_hypothesis = self.beam_search.decode(
            spectra=batch[0].float(),
            precursors=batch[1],
            beam_size=self.beam_size,
            max_length=self.config.hypotheses.max_peptide_len,
        )

        return best_hypothesis
