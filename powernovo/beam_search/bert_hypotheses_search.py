import logging
import os.path
from typing import Any

import numpy as np
import torch
from powernovo.depthcharge_base.tokenizers.peptides import PeptideTokenizer
from powernovo.depthcharge_base.utils.constants import MASS_SCALE, H2O
from powernovo.beam_search.adaptive_beam_search import AdaptiveBeamSearchDecoder
from powernovo.models.peptide_bert.peptide_bert import PeptideBert
from powernovo.models.spectrum.spectrum_inference import SpectrumTransformer
from Bio.PDB.Polypeptide import aa1
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from powernovo.utils.utils import all_equal

logger = logging.getLogger("powernovo")
logger.setLevel(logging.INFO)

creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)


class KnapsackBeamSearchDecoder(AdaptiveBeamSearchDecoder):
    def __init__(
            self,
            spectrum_model: SpectrumTransformer,
            tokenizer: PeptideTokenizer,
            peptide_bert_model: PeptideBert,
            device: torch.device,
            mass_scale: MASS_SCALE,
            lambda_: int = 100,
            cx_prob: float = 1.0,
            n_gen: int = 30):
        super().__init__(
            model=spectrum_model,
            tokenizer=tokenizer,
            device=device,
            mass_scale=mass_scale
        )
        self.peptide_bert_model = peptide_bert_model
        self.aa_n_term_str = np.array([k for k, v in self.tokenizer.index.items() if v in self.aa_n_term], dtype=str)
        self.lamda_ = lambda_
        self.cx_prob = cx_prob
        self.n_gen = n_gen

    @classmethod
    def from_bert_config(cls,
                         spectrum_model: SpectrumTransformer,
                         tokenizer: PeptideTokenizer,
                         bert_model_path: str,
                         device: torch.device,
                         mass_scale: MASS_SCALE,
                         lambda_: int = 100,
                         cx_prob: float = 1.0,
                         n_gen: int = 30
                         ):
        assert os.path.exists(bert_model_path), f"Can't find peptide bert model file: {bert_model_path}"

        logger.info(f"Load peptide bert model...")
        bert_model_data = torch.load(bert_model_path)
        bert_model_cfg = bert_model_data['config']
        bert_model = PeptideBert(config=bert_model_cfg, device=device)
        bert_model.load_state_dict(bert_model_data['state_dict'])
        bert_model = bert_model.to(device)

        logger.info('Peptide bert model loaded successfully')

        return cls(spectrum_model=spectrum_model,
                   tokenizer=tokenizer,
                   peptide_bert_model=bert_model,
                   device=device,
                   mass_scale=mass_scale,
                   lambda_=lambda_,
                   cx_prob=cx_prob,
                   n_gen=n_gen
                   )

    def detokenize(self, tokens: torch.Tensor) -> str:
        return super().detokenize(tokens, join=False)

    def decode(
            self,
            spectra: torch.FloatTensor,
            precursors: torch.FloatTensor,
            beam_size: int,
            max_length: int,
            mass_tolerance: float = 5e-5,
            max_isotope: int = 1,
            return_all_beams: bool = False
    ) -> list[Any]:
        hypotheses = super().decode(spectra=spectra,
                                    precursors=precursors,
                                    beam_size=beam_size,
                                    max_length=max_length,
                                    mass_tolerance=mass_tolerance,
                                    max_isotope=max_isotope,
                                    return_all_beams=return_all_beams
                                    )
        hypotheses = self.peptide_bert_model.score_hypotheses(hypotheses)

        for i, item in enumerate(hypotheses):
            if item:
                try:
                    if item.mass_error > mass_tolerance * self.mass_scale:
                        precursors_charge = precursors[:, 1].cpu().numpy()
                        precursors_mass = precursors[:, 0].cpu().numpy()
                        solved_record = self.bert_knapsack_search(item.sequence,
                                                                  precursors_mass[i],
                                                                  precursors_charge[i],
                                                                  source_mass_err=item.mass_error
                                                                  )
                        if not solved_record:
                            hypotheses[i] = []
                            continue

                        hypotheses[i].sequence = solved_record['sequence']
                        hypotheses[i].mass_err = solved_record['mass_err']
                        hypotheses[i].solved = solved_record['solved']

                except (Exception, AttributeError, AssertionError, KeyError):
                    continue
            try:
                hypotheses[i].sequence = ''.join(hypotheses[i].sequence)
            except (AttributeError, Exception):
                hypotheses[i] = []

        return hypotheses

    def bert_knapsack_search(self,
                             sequence: str,
                             precursor_mass: float,
                             precursor_charge: int,
                             source_mass_err: float
                             ):
        if sequence is None:
            return {}
        if len(sequence) < self.min_peptide_len:
            return {}
        candidates = self.peptide_bert_model.get_candidates(sequence)

        if all_equal(candidates):
            return {'sequence': sequence, 'mass_err': source_mass_err, 'solved': False}

        if not candidates:
            return {}
        knapsack = GAKnapsackSearch(tokenizer=self.tokenizer,
                                    candidates=[sequence] + candidates,
                                    precursor_mass=precursor_mass,
                                    precursor_charge=precursor_charge,
                                    mass_scale=self.mass_scale,
                                    lambda_=self.lamda_,
                                    cx_prob=self.cx_prob,
                                    n_gen=self.n_gen
                                    )
        solved_item, solved_mass_err = knapsack.solve()

        if abs(source_mass_err) > abs(solved_mass_err):
            return {'sequence': solved_item, 'mass_err': source_mass_err, 'solved': True}
        else:
            return {'sequence': sequence, 'mass_err': source_mass_err, 'solved': False}


class GAKnapsackSearch(object):
    def __init__(self,
                 tokenizer: PeptideTokenizer,
                 candidates: list[str],
                 precursor_mass: float,
                 precursor_charge: int,
                 mass_scale: int = MASS_SCALE,
                 lambda_: int = 100,
                 cx_prob: float = 1.0,
                 n_gen: int = 30
                 ):
        self.mass_scale = mass_scale
        self.precursor_mass = precursor_mass
        self.precursor_charge = precursor_charge
        self.tokenizer = tokenizer
        self.tokenizer_aa = {v: k for k, v in self.tokenizer.index.items() if k in aa1}
        self.candidates = candidates
        self.candidates = [candidates[0]] + [list(x) for x in candidates[1:]]
        self.population_counter = 0
        self.lambda_ = lambda_
        self.cx_prob = cx_prob
        self.n_gen = n_gen
        self.toolbox = base.Toolbox()
        self.toolbox.register('init_item', self.population_init)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.init_item, 1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evalKnapsack)
        self.toolbox.register("mate", self.cxSet)
        self.toolbox.register("mutate", self.mutSet)
        self.toolbox.register("select", tools.selNSGA2)

    def population_init(self):
        individual = self.candidates[self.population_counter]
        self.population_counter += 1
        return individual

    def evalKnapsack(self, individual):
        mass_err, mass = self.calc_mass_error(individual[0])

        return abs(mass_err),

    @staticmethod
    def cxSet(ind1, ind2):
        diff = []
        for i, x in enumerate(ind1[0]):
            if i >= len(ind2[0]):
                break
            if x != ind2[0][i]:
                diff.append(i)
        if diff:
            cx = np.random.choice(diff, 1)[0]
            tmp = ind1[0][cx]
            ind1[0][cx] = ind2[0][cx]
            ind2[0][cx] = tmp
        return ind1, ind2

    @staticmethod
    def mutSet(individual):
        return individual,

    def solve(self):
        mu = len(self.candidates)
        population = self.toolbox.population(n=len(self.candidates))
        hof = tools.ParetoFront()

        algorithms.eaMuPlusLambda(population,
                                  self.toolbox,
                                  mu,
                                  self.lambda_,
                                  self.cx_prob,
                                  1e-6,
                                  self.n_gen,
                                  halloffame=hof,
                                  verbose=False)
        best_hof = hof.items[0]
        mass_err, mass = self.calc_mass_error(best_hof[0])
        return best_hof[0], mass_err

    def calc_mass_error(self, sequence: list[str]):
        mass = np.sum([np.round(self.tokenizer.residues[s] * self.mass_scale) for s in sequence])
        mass_error = (round(self.precursor_mass * self.mass_scale - self.mass_scale * H2O) - mass) / self.mass_scale
        return mass_error, mass / self.mass_scale
