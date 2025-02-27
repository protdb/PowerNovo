import os.path
import torch
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.cli import ReduceLROnPlateau
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torchmetrics import Accuracy

from powernovo.pipeline_config.config import PWNConfig, DEFAULT_CONFIG_PARAMS
from powernovo.depthcharge_base.data.spectrum_datasets import AnnotatedSpectrumDataset
from powernovo.depthcharge_base.tokenizers.peptides import PeptideTokenizer
from powernovo.models.spectrum.spectrum_inference import SpectrumTransformer


class TrainingWrapper(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = SpectrumTransformer(device=self.device).float()
        state_dict = torch.load("/home/dp/Data/powernovo/models/pwn_spectrum.pt")
        self.model.load_state_dict(state_dict['state_dict'])
        self.tokenizer = PeptideTokenizer.from_massivekb(reverse=False)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=len(self.tokenizer))

    def forward(self, batch):
        spectrum = batch[0].float()
        precursors = batch[1]
        tokens_ids = batch[2]
        spectrum_embedding, memory_mask = self.model.spectrum_encoder(spectrum)
        decoder_logit = self.model.peptide_decoder(
            tokens=tokens_ids[:, :-1],
            precursors=precursors,
            memory=spectrum_embedding,
            memory_key_padding_mask=memory_mask
        )

        decoder_logit = decoder_logit.permute(0, 2, 1)
        loss = self.loss(decoder_logit, tokens_ids)
        tokens_pred = torch.argmax(decoder_logit, dim=1)
        accuracy = self.accuracy(tokens_pred, tokens_ids)
        results = {
            'loss': loss,
            'accuracy': accuracy
        }
        return results

    def step(self, batch, mode='train'):
        if batch[0] is None:
            return None

        results = self(batch=batch)
        loss = results['loss']
        self.log_metrics(mode, results)
        return loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        with torch.no_grad():
            loss = self.step(batch, mode='val')
        return loss

    def log_metrics(self, stage_name, metrics):
        for metric_name, metric_value in metrics.items():
            self.log(f'{stage_name}_{metric_name}',
                     metric_value.detach().cpu().item(),
                     batch_size=len(metric_name),
                     on_step=False,
                     on_epoch=True,
                     prog_bar=False)

    def configure_optimizers(self):
        config = PWNConfig()
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=float(config.learning_rate),
                                      betas=(0.9, 0.999),
                                      eps=1e-08,
                                      weight_decay=0.01,
                                      amsgrad=False,
                                      maximize=False,
                                      foreach=None,
                                      capturable=False)
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, monitor="val_loss", verbose=True),
            "monitor": "epoch",
            "frequency": 1
        }

        return [optimizer], lr_scheduler


def train_spectrum_model():
    torch.set_float32_matmul_precision('high')
    config = PWNConfig(working_folder="/home/dp/Data/powernovo/")
    config.annotated = True
    trainer_wrapper = TrainingWrapper()
    train_dataset = create_dataset(dataset_path="/home/dp/Data/ext_benchmark/dataset/train/",
                                   tokenizer=trainer_wrapper.tokenizer
                                   )

    val_dataset = create_dataset(dataset_path="/home/dp/Data/ext_benchmark/dataset/val/",
                                 tokenizer=trainer_wrapper.tokenizer
                                 )

    checkpoint_folder = config.checkpoint_folder
    if not os.path.exists(checkpoint_folder):
        checkpoint_folder = Path(config.working_folder) / checkpoint_folder
        checkpoint_folder.mkdir(exist_ok=True)

    model_logger = CSVLogger(save_dir=checkpoint_folder,
                             name='training_log')

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=3,
                                        verbose=True,
                                        mode="min")

    trainer = pl.Trainer(default_root_dir=checkpoint_folder,
                         accelerator='auto',
                         logger=model_logger,
                         max_epochs=config.max_epoch,
                         gradient_clip_val=0.5,
                         callbacks=[early_stop_callback],
                         )
    num_workers = int(config.max_workers)
    batch_size = int(config.train_batch_size)
    trainer.fit(
        model=trainer_wrapper,
        train_dataloaders=train_dataset.loader(batch_size=batch_size,
                                               num_workers=num_workers),
        val_dataloaders=val_dataset.loader(batch_size=batch_size,
                                           num_workers=num_workers),
    )


def create_dataset(dataset_path: str, tokenizer: PeptideTokenizer) -> AnnotatedSpectrumDataset:
    index_folder = Path(dataset_path) / 'index'
    index_folder.mkdir(exist_ok=True)
    dataset_files = glob.glob(f'{dataset_path}/*.mgf')
    index_path = index_folder / f'{Path(dataset_path).stem}.hdf5'
    dataset = AnnotatedSpectrumDataset(tokenizer=tokenizer,
                                       ms_data_files=dataset_files,
                                       overwrite=False,
                                       index_path=index_path,
                                       )
    return dataset


if __name__ == "__main__":
    train_spectrum_model()
