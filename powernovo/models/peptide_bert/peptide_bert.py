import os.path

import numpy as np
import torch
from torch import nn
from transformers import BertForMaskedLM, BertTokenizer
from powernovo.pipeline_config.config import PWNConfig
from powernovo.utils.utils import to_canonical


class PeptideBert(nn.Module):
    PAD_LEN = 100

    def __init__(self, *args, **kwargs):
        super().__init__()
        config = kwargs['config']
        self.device = kwargs['device']
        self.model = BertForMaskedLM(config).to(self.device)
        self.config = PWNConfig()
        vocab_filepath = self.config.models_folder / self.config.peptide_bert.vocab_file
        assert os.path.exists(vocab_filepath), f"Can't find peptide bert vocab file: {vocab_filepath}"
        self.tokenizer = BertTokenizer(
            vocab_file=vocab_filepath,
            vocab_size=30,
            do_lower_case=False,
            do_basic_tokenize=True,
            never_split=None,
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]',
        )
        self.mask_token_id = self.tokenizer.vocab['[MASK]']
        self.pad_token_id = self.tokenizer.vocab['[PAD]']
        self.aa_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()
                         if v not in self.tokenizer.all_special_ids}
        self.min_peptide_len = self.config.hypotheses.min_peptide_len
        self.special_tokens_ = torch.as_tensor([self.tokenizer.cls_token_id, self.tokenizer.pad_token_id],
                                               dtype=torch.long, device=self.device)

        self.special_tokens = torch.as_tensor([self.tokenizer.cls_token_id, self.tokenizer.pad_token_id,
                                               self.tokenizer.sep_token_id],
                                              dtype=torch.long, device=self.device)

        classifier_dim = self.config.peptide_bert.classifier_input_dim
        self.detectable_cls = nn.Sequential(nn.Linear(classifier_dim, classifier_dim // 4),
                                            nn.ELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(classifier_dim // 4, classifier_dim // 16),
                                            nn.ELU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(classifier_dim // 16, classifier_dim // 32),
                                            nn.ELU(),
                                            nn.Linear(classifier_dim // 32, 3)
                                            )

    def forward(self, tokens, attention_mask, classify=False):
        self.model.eval()

        with torch.no_grad():
            output = self.model(tokens, attention_mask, output_hidden_states=classify)
            detectability = None

            if classify:
                self.detectable_cls.eval()
                hx = output.hidden_states
                pooled_hx = torch.cat(tuple(hx[i] for i in (-4, -3, -2, -1)), dim=-1)
                pooled_emb = pooled_hx[:, 0, :]
                pooled_hx = torch.mean(pooled_hx, dim=1)
                pooled_output = torch.cat((pooled_emb, pooled_hx), dim=-1)
                detectability = torch.log_softmax(self.detectable_cls(pooled_output), dim=-1)
        return output, detectability

    def get_candidates(self, sequence=None):
        tokens_ids, attention_mask = self.mask_sequence(''.join(sequence))

        output, _ = self(tokens_ids, attention_mask)

        logits = output.logits[attention_mask.bool()]
        scores = torch.softmax(logits, dim=-1)
        pred = torch.argmax(scores, dim=-1)
        candidates = self.tokenizer.decode(pred, skip_special_tokens=False)
        candidate_tokens = np.array(candidates.split(' '), dtype=str)
        special_token_mask = candidate_tokens == self.tokenizer.cls_token
        special_token_mask |= candidate_tokens == self.tokenizer.sep_token
        candidates = candidates.split(self.tokenizer.sep_token)
        candidates = [s.replace(self.tokenizer.cls_token, '') for s in candidates]
        candidates = [s.replace(' ', '') for s in candidates]
        candidates = [s.replace('I', 'L') for s in candidates]  # replace_isoleucine_with_leucine
        candidates.remove('')
        return candidates

    def mask_sequence(self, sequence):
        canonical_seq = to_canonical(sequence)
        canonical_seq = ' '.join(list(canonical_seq))
        tokens_ids, attention_mask = self.tokenize(canonical_seq)
        masked_batch = [tokens_ids]
        batch_attention_mask = [attention_mask]
        tokens_ids = tokens_ids.squeeze()
        attention_mask = attention_mask.squeeze()

        for i, s in enumerate(tokens_ids):
            masked_tokens = tokens_ids.clone()
            if s in self.tokenizer.all_special_ids:
                if s == self.pad_token_id:
                    break
                continue
            masked_tokens[i] = self.mask_token_id
            masked_batch.append(masked_tokens)
            batch_attention_mask.append(attention_mask)

        masked_batch = torch.vstack(masked_batch).to(self.device)
        batch_attention_mask = torch.vstack(batch_attention_mask).to(self.device)
        return masked_batch, batch_attention_mask

    def tokenize(self, sequence):
        target_inputs = self.tokenizer.encode_plus(sequence,
                                                   padding="max_length",
                                                   max_length=self.PAD_LEN,
                                                   truncation=True,
                                                   return_tensors="pt")
        tokens_ids = target_inputs["input_ids"]
        attention_mask = target_inputs["attention_mask"]
        tokens_ids = tokens_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return tokens_ids, attention_mask

    def score_hypotheses(self, hypotheses: list):
        prefilter = [i for i, h in enumerate(hypotheses) if not h]
        sequence_batch = [to_canonical(''.join(h.sequence)) for h in hypotheses if h]
        scored_hypotheses = self.score(sequence_batch)
        updated_hypotheses = []
        hyp_idx = 0

        for i, h in enumerate(hypotheses):
            if i in prefilter:
                continue

            if scored_hypotheses[hyp_idx]['discarded']:
                hyp_idx += 1
                continue
            detectable_label = scored_hypotheses[hyp_idx]['detectable']
            detectable_score = scored_hypotheses[hyp_idx]['detectable_score']

            if detectable_label <= 0:
                hyp_idx += 1
                continue

            h.score *= detectable_label * detectable_score
            sequence = scored_hypotheses[hyp_idx]['sequence']
            aa_scores = scored_hypotheses[hyp_idx]['aa_scores']
            aa_scores *= min(1.0, h.score * 1.0 / np.exp(abs(h.mass_error) ** 2) / len(sequence))
            h.aa_scores = np.round(scored_hypotheses[hyp_idx]['aa_scores'], 2).tolist()
            updated_hypotheses.append(h)
            hyp_idx += 1
        return updated_hypotheses

    def score(self, sequences: list):
        discarded_mask = [len(s) < self.min_peptide_len for s in sequences]
        sequence_batch = [s for i, s in enumerate(sequences) if not discarded_mask[i]]
        sequence_batch = [' '.join(list(s)) for s in sequence_batch]

        batch_inputs = [self.tokenizer.encode_plus(s,
                                                   padding="max_length",
                                                   max_length=self.PAD_LEN,
                                                   truncation=True,
                                                   return_tensors="pt") for s in sequence_batch]

        batch_token_ids = [b["input_ids"].squeeze() for b in batch_inputs]
        batch_attention_mask = [b["attention_mask"].squeeze() for b in batch_inputs]
        batch_token_ids = torch.vstack(batch_token_ids)
        batch_attention_mask = torch.vstack(batch_attention_mask)
        batch_token_ids = batch_token_ids.to(self.device)
        batch_attention_mask = batch_attention_mask.to(self.device)
        output, detectability = self(batch_token_ids, batch_attention_mask, classify=True)
        p_detectable, detectability_labels = torch.max(detectability, dim=-1)
        detectability_labels = detectability_labels.cpu().numpy()
        detectability_scores = torch.exp(p_detectable).cpu().numpy()
        logits = output.logits[batch_attention_mask.bool()]
        scores = torch.softmax(logits, dim=-1)
        scores, token_ids = torch.max(scores, dim=-1)
        special_tokens_mask = torch.isin(token_ids, self.special_tokens_)
        all_special_token_mask = torch.isin(token_ids, self.special_tokens)
        token_ids = token_ids[~special_tokens_mask]
        scores = scores[~all_special_token_mask]

        pred_seq = self.tokenizer.decode(token_ids)
        pred_seq = pred_seq.split(self.tokenizer.sep_token)
        pred_seq.remove('')
        pred_seq = [s.replace(' ', '') for s in pred_seq]

        output = {}
        pred_idx = 0
        scores_idx = 0

        for i, seq in enumerate(sequences):
            try:
                if discarded_mask[i]:
                    raise ValueError('Discarded')

                predicted_ = pred_seq[pred_idx]
                aa_scores = scores[scores_idx: scores_idx + len(predicted_)].cpu().numpy()
                scores_idx += len(predicted_)
                aa_scores_norm = (aa_scores - np.min(aa_scores)) / (np.max(aa_scores) - np.min(aa_scores))
                aa_scores_norm_err = (1.0 - aa_scores_norm) / 2.0
                aa_scores = aa_scores - aa_scores_norm_err

                output.update({i: {'discarded': False,
                                   'sequence': predicted_,
                                   'detectable': detectability_labels[pred_idx],
                                   'detectable_score': detectability_scores[pred_idx],
                                   'aa_scores': aa_scores}})

                pred_idx += 1
                assert len(aa_scores) == len(predicted_)
                assert len(predicted_) >= self.min_peptide_len
            except (IndexError, AssertionError, ValueError):
                output.update({i: {'discarded': True,
                                   'sequence': seq,
                                   'detectable': -1,
                                   'aa_scores': []}})
                continue

        return output
