import argparse
import glob
import json
import logging
import os.path
import shutil
import sys
from pathlib import Path

import requests
from requests import HTTPError
import zipfile
from inference_runner.inference import PWNInference
from inference_runner.pipeline_config.config import PWNConfig

logger = logging.getLogger("powernova")


def retrieve_data_from_figshare():
    config = PWNConfig()
    figshare_id = config.figshare_id
    temporary_folder = Path(config.working_folder) / config.temporary_folder
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
        if not os.path.exists(working_folder):
            logger.error("The working folder specified in the pipeline_config.yaml "
                         "configuration file cannot be created. Please check your settings")
            logger.error(e)
            sys.exit(1)

    model_folder = Path(config.working_folder) / config.models_folder

    is_need_download_models = False
    if not os.path.exists(model_folder):
        is_need_download_models = True
    elif not os.listdir(model_folder):
        is_need_download_models = True
    else:
        logger.info('Environment check completed successfully')

    if is_need_download_models:
        logger.error('Setup local environment. Downloading model weights and ALPS assembler from Figshare')
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
    check_and_setup_environment()

    for file in files:
        file_size = os.path.getsize(file)
        logger.info(f"FILE: {os.path.basename(file)} ({round(file_size / (pow(1024, 2)), 2)} MB)")

    inference = PWNInference()
    inference.load_models()

    for file in files:
        inference.run(input_file=file, output_folder=output_folder)


def process_file(input_file: str, output_folder: str):
    check_and_setup_environment()
    inference = PWNInference()
    inference.load_models()
    inference.run(input_file=input_file, output_folder=output_folder)


def main():
    parser = argparse.ArgumentParser(description="Start PowerNovo pipeline")
    parser.add_argument('inputs', type=str, help="Path to the mgf file or path to the folder containing the list of "
                                                 "*.mgf files")
    parser.add_argument('-o', '--output', type=str, help='Output folder [optional]', required=False, default='')
    args = parser.parse_args()

    inputs = args.inputs

    if not os.path.exists(inputs):
        logger.error(f"The specified file or folder was not found: {args.input}")
        sys.exit(1)

    if os.path.isfile(inputs):
        process_file(input_file=inputs, output_folder=args.output)

    elif os.path.isdir(inputs):
        process_folder(input_folder=inputs, output_folder=args.output)
    else:
        logger.error("Invalid input args")


if __name__ == '__main__':
    main()
