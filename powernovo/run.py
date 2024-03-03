import argparse
import glob
import logging
import os.path
import shutil
import sys
from pathlib import Path

import requests
from requests import HTTPError
import zipfile
from powernovo.inference import PWNInference
from powernovo.pipeline_config.config import PWNConfig

logger = logging.getLogger("powernovo")


def retrieve_data_from_figshare():
    config = PWNConfig()
    figshare_id = config.figshare_id
    temporary_folder = config.temporary_folder
    temporary_folder.mkdir(exist_ok=True)

    url = f"https://figshare.com/ndownloader/files/{figshare_id}"
    headers = {'Content-Type': 'application/json'}
    try:
        logger.info(f'Start downloading https://figshare.com/ndownloader/files/{figshare_id}')
        response = requests.request('GET', url, headers=headers, data=None)
        datafile = temporary_folder / 'model_data.zip'
        with open(datafile, 'wb') as fh:
            fh.write(response.content)
        assert os.path.exists(datafile)

        logger.info('Download complete')
        logger.info('Extract data...')
        datafile = temporary_folder / 'model_data.zip'
        with zipfile.ZipFile(datafile, 'r') as zip_fh:
            zip_fh.extractall(temporary_folder)

        sub_folders = next(os.walk(temporary_folder))[1]
        for folder in sub_folders:
            dst_folder = temporary_folder / folder
            if os.path.exists(config.working_folder / folder):
                shutil.rmtree(config.working_folder / folder)
            shutil.move(dst_folder, config.working_folder)

        if os.path.exists(temporary_folder):
            shutil.rmtree(temporary_folder)

        logger.info('Done')

    except (HTTPError, IOError, Exception, AssertionError) as e:
        logger.error("It is not possible to retrieve model data from Figshare. Try downloading them manually:"
                     "10.6084/m9.figshare.25329586")
        raise e


def check_and_setup_environment():
    working_folder = None
    config = None
    logger.info('Check environment...')
    try:
        config = PWNConfig()
        working_folder = config.working_folder
        assert config is not None
    except (Exception, AssertionError) as e:
        if working_folder is None:
            logger.error("The working folder specified in the pipeline_config.yaml "
                         "configuration file cannot be created. Please check your settings")
            logger.error(e)
            sys.exit(1)

    model_folder = config.models_folder
    is_need_download_models = False
    if not os.path.exists(model_folder):
        is_need_download_models = True
    elif not os.listdir(model_folder):
        is_need_download_models = True
    else:
        logger.info('Environment check completed successfully')

    if is_need_download_models:
        logger.info('Setup local environment. Downloading model weights and ALPS assembler from Figshare')
        try:
            retrieve_data_from_figshare()
        except Exception as e:
            sys.exit(1)


def process_folder(input_folder: str, output_folder: str):
    files = glob.glob(f'{input_folder}/*.mgf')

    if not files:
        logger.error(f"The mgf files in the specified folder were not found: {input_folder}")
        sys.exit(1)
    logger.info(f"Process all files in folder: {input_folder}")

    for file in files:
        file_size = os.path.getsize(file)
        logger.info(f"FILE: {os.path.basename(file)} ({round(file_size / (pow(1024, 2)), 2)} MB)")

    inference = PWNInference()
    inference.load_models()

    for file in files:
        inference.run(input_file=file, output_folder=output_folder)


def process_file(input_file: str, output_folder: str):
    inference = PWNInference()
    inference.load_models()
    inference.run(input_file=input_file, output_folder=output_folder)


def run_inference(inputs: str,
                  working_folder: str = 'pwn_work',
                  output_folder: str = '',
                  batch_size: int = 16,
                  use_assembler: bool = True,
                  protein_inference: bool = True,
                  fasta_path: str = '',
                  use_bert: bool = True,
                  num_contigs: int = 20,
                  contigs_kmers: list = [7, 8, 10]
                  ):
    """Setup config"""
    try:
        config = PWNConfig(working_folder=working_folder)
        config.inference_batch_size = batch_size
        config.use_bert_inference = use_bert
        config.peptides.use_alps = use_assembler
        config.peptides.n_contigs = num_contigs
        config.peptides.kmers = contigs_kmers
        config.proteins.inference = protein_inference

        if not os.path.exists(fasta_path):
            fasta_path = config.database_folder / 'UP000005640_9606.fasta'
            config.proteins.fasta_path = fasta_path
    except (Exception, AssertionError, KeyError, IOError) as e:
        logger.error('Error in incoming parameters. Please check your launch options')
        raise e
        sys.exit(1)

    check_and_setup_environment()

    if os.path.isfile(inputs):
        process_file(input_file=inputs, output_folder=output_folder)
    elif os.path.isdir(inputs):
        process_folder(input_folder=inputs, output_folder=output_folder)
    else:
        logger.error('Invalid inputs. Either an mgf file or a directory containing such files must be specified')


def main():
    parser = argparse.ArgumentParser(description="Start PowerNovo pipeline")
    parser.add_argument('inputs', type=str, help="Path to the mgf file or path to the folder containing the list of "
                                                 "*.mgf files")
    parser.add_argument('-w', '--working_folder', type=str, help='Working folder for download models, ALPS, etc.',
                        required=False, default='pwn_work')

    parser.add_argument('-o', '--output_folder', type=str, help='Output folder [optional]', required=False, default='')
    parser.add_argument('-batch_size', '--batch_size', type=int, help='Batch size', required=False, default=16)
    parser.add_argument('-alps', '--assembler', type=bool, help='Use ALPS assembler [optional]',
                        required=False, default=True)
    parser.add_argument('-num_contigs', '--num_contigs', type=int, help='Number of generated contigs',
                        required=False, default=20)
    parser.add_argument('-contigs_kmers', '--contigs_kmers', type=int, help='Contigs kmers size',
                        required=False, nargs='+', default=[7, 8, 10])
    parser.add_argument('-infer', '--protein_inference', type=bool, help='Use protein inference algorithm [optional]',
                        required=False, default=True)
    parser.add_argument('-fasta', '--fasta_path', type=str, help="Path to the fasta file that is used for "
                                                                 "peptide-protein mappings. "
                                                                 "If not specified will be used"
                                                                 "UP000005640_9606.fasta",
                        required=False, default='')
    parser.add_argument('-use-bert', '--use_bert', type=bool, help='Use BERT model',
                        required=False, default=True)
    args = parser.parse_args()

    inputs = args.inputs

    if not os.path.exists(inputs):
        logger.error(f"The specified file or folder was not found: {args.input}")
        sys.exit(1)

    run_inference(
        inputs=args.inputs,
        working_folder=args.working_folder,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        use_assembler=args.assembler,
        protein_inference=args.protein_inference,
        fasta_path=args.fasta_path,
        use_bert=args.use_bert,
        num_contigs=args.num_contigs,
        contigs_kmers=args.contigs_kmers
    )


if __name__ == '__main__':
    main()
