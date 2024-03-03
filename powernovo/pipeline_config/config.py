import os.path
from pathlib import Path
import yaml
import logging

from powernovo.utils.utils import Singleton

CONFIG_FILE = "config.yaml"

logger = logging.getLogger("powernovo")

DEFAULT_CONFIG_PARAMS = {'environments': [{'working_folder': ''},
                                          {'figshare_id': 44806387},
                                          {'temporary_folder': 'temp_'},
                                          {'models_folder': 'models'},
                                          {'database_folder': 'database'},
                                          {'preprocessing_folder': 'index'},
                                          {'output_folder_name': 'pwn_output'},
                                          {'database_folder': 'database'},
                                          {'assembler_folder': 'assembler'},
                                          {'annotated': False},
                                          {'use_bert_inference': True},
                                          {'inference_batch_size': 5},
                                          {'device': 'auto'},
                                          {'n_workers': 0}],
                         'models': [{'spectrum_transformer': [{'trained_model': 'pwn_spectrum.pt'},
                                                              {'dmodel': 512},
                                                              {'n_layers': 10},
                                                              {'max_charge': 10},
                                                              {'n_tokens': 32}]},
                                    {'peptide_bert': [{'trained_model': 'pwn_bert.pt'},
                                                      {'vocab_file': 'tokenizer_vocab.txt'},
                                                      {'classifier_input_dim': 8192}]},
                                    {'hypotheses': [{'beam_size': 5},
                                                    {'precursor_mass_tol': 10.0},
                                                    {'min_peptide_len': 7},
                                                    {'max_peptide_len': 64},
                                                    {'max_steps': 0},
                                                    {'lambda_': 100, 'cx_prob': 0.9, 'n_gen': 30}]}],
                         'assignment': [{'peptides': [{'use_alps': True},
                                                      {'running_file': 'ALPS.jar'},
                                                      {'n_contigs': 10}, {'kmers': [7, 8, 10]}]},
                                        {'proteins': [{'minIdentity': 0.75}, {'inference': True},
                                                      {'map_modifications': True},
                                                      {'fasta_path': ''}]}],
                         'spectrum_model_train': [{'train_dataset_path': None}, {'val_dataset_path': None},
                                                  {'checkpoint_folder': 'checkpoints'}, {'max_epoch': 256},
                                                  {'learning_rate': '5e-4'}, {'max_workers': 4}, {'batch_size': 256}]}


class ModelParams(object):
    pass


class PWNConfig(metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        self.working_folder = None
        self.models_folder = None
        self.preprocessing_folder = None
        self.assembler_folder = None
        self.output_folder_name = None
        self.database_folder = None
        self.temporary_folder = None
        self.figshare_id = None
        self.annotated = False
        self.n_workers = 0
        self.train_batch_size = 1
        self.inference_batch_size = 1
        self.device = 'auto'
        self.use_bert_inference = False

        config_path = Path(__file__).parent.resolve() / CONFIG_FILE
        try:
            with open(config_path, 'r') as fh:
                config_records = yaml.safe_load(fh)
        except FileNotFoundError:
            config_records = DEFAULT_CONFIG_PARAMS

        # environments
        for r in config_records['environments']:
            for k, v in r.items():
                self.__setattr__(k, v)

        try:
            assert self.working_folder is not None
            assert self.models_folder is not None
        except AssertionError as e:
            logger.error(f'Invalid pipeline_config records: {config_path}')
            raise e

        try:
            working_folder = kwargs['working_folder']
            self.working_folder = working_folder
            self.working_folder = Path(self.working_folder)
            self.working_folder.mkdir(exist_ok=True)
            logger.info(f'Current work folder is {self.working_folder}')
            self.models_folder = self.working_folder / self.models_folder
            self.models_folder.mkdir(exist_ok=True)
            self.assembler_folder = self.working_folder / self.assembler_folder
            self.assembler_folder.mkdir(exist_ok=True)
            self.database_folder = self.working_folder / self.database_folder
            self.database_folder.mkdir(exist_ok=True)
            self.temporary_folder = self.working_folder / self.temporary_folder
        except Exception as e:
            logger.error(f'Path not found: {e}')
            raise e

        for r in config_records['models']:
            for model in r:
                self.__setattr__(model, ModelParams())
                for params in r[model]:
                    m = self.__getattribute__(model)
                    for k, v in params.items():
                        m.__setattr__(k, v)

        for r in config_records['assignment']:
            for assignment in r:
                self.__setattr__(assignment, ModelParams())
                for params in r[assignment]:
                    m = self.__getattribute__(assignment)
                    for k, v in params.items():
                        m.__setattr__(k, v)

        for r in config_records['spectrum_model_train']:
            for k, v in r.items():
                self.__setattr__(k, v)


if __name__ == '__main__':
    config = PWNConfig()
