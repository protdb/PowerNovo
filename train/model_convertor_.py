import os
from collections import OrderedDict

import torch
from transformers import BertConfig

from inference_runner.pipeline_config.config import PWNConfig
from inference_runner.models.peptide_bert.peptide_bert import PeptideBert
from inference_runner.models.spectrum.spectrum_inference import SpectrumTransformer


def convert_transformer_checkpoint_():
    config = PWNConfig()
    model_file = config.spectrum_transformer.trained_model
    model_path = config.models_folder / model_file
    checkpoint = str(model_path).replace('pt', 'ckpt')
    assert os.path.exists(checkpoint)

    model_data = torch.load(checkpoint, map_location='cpu')
    state_dict = model_data['state_dict']

    new_state = OrderedDict()

    for k, v in state_dict.items():
        new_k = k.replace('model.', '')

        if 'reverse' in new_k:
            continue

        if '_forward' in new_k:
            new_k = new_k.replace('_forward', '')

        new_state.update({new_k: v})

    # # test
    model = SpectrumTransformer()
    model.load_state_dict(new_state)

    model_data = {k: v for k, v in model_data.items() if k == 'state_dict'}
    model_data.update({'state_dict': new_state})
    torch.save(model_data, model_path)


bert_config = BertConfig.from_dict({
    "_name_or_path": "yarongef/DistilProtBert",
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.0,
    "classifier_dropout": None,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 40000,
    "model_type": "bert",
    "num_attention_heads": 8,
    "num_hidden_layers": 7,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "torch_dtype": "float32",
    "transformers_version": "4.10.0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30
})

tokenizer = {'name_or_path': 'yarongef/DistilProtBert',
             'vocab_size': 30,
             'model_max_len': 1000000000000000019884624838656,
             'is_fast': False,
             'padding_side': 'right',
             'special_tokens': {'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]',
                                'mask_token': '[MASK]'}
             }


def convert_bert_checkpoint_():
    config = PWNConfig()
    model_file = config.peptide_bert.trained_model
    model_path = config.models_folder / model_file
    checkpoint = str(model_path).replace('pt', 'ckpt')
    assert os.path.exists(checkpoint)

    model_data = torch.load(checkpoint, map_location='cpu')
    state_dict = model_data['state_dict']

    new_state = OrderedDict()

    for k, v in state_dict.items():
        new_k = k.replace('model.model.', 'model.')
        new_state.update({new_k: v})

    model = PeptideBert(config=bert_config, device=torch.device('cuda'))
    model.load_state_dict(new_state)

    torch_data = {'pipeline_config': bert_config, 'state_dict': new_state}
    torch.save(torch_data, model_path)


def convert_bert_checkpoint_with_cls():
    config = PWNConfig()
    model_file = config.peptide_bert.trained_model
    model_path = config.models_folder / model_file

    model_data = torch.load(model_path, map_location='cpu')
    state_dict = model_data['state_dict']

    new_state = OrderedDict()

    for k, v in state_dict.items():
        new_k = k.replace('classifier.model.', 'model.')
        if 'classifier.detectable_cls.' in new_k:
            new_k = new_k.replace('classifier.detectable_cls.', 'detectable_cls.')
        new_state.update({new_k: v})

    model = PeptideBert(config=bert_config, device=torch.device('cuda'))
    model.load_state_dict(new_state)

    torch_data = {'pipeline_config': bert_config, 'state_dict': new_state}
    torch.save(torch_data, model_path)


if __name__ == '__main__':
    convert_bert_checkpoint_with_cls()
