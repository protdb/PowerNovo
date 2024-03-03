import glob
import logging
import os.path
import subprocess
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import npysearch as npy
from powernovo.pipeline_config.config import PWNConfig
from powernovo.proteins.protein_inference import ProteinInference
from powernovo.utils.utils import from_proforma, to_canonical

logger = logging.getLogger("powernovo")
logger.setLevel(logging.INFO)

ASSEMBLY_COLUMNS = ['Spectrum Name', 'PowerNovo Peptides', 'PowerNovo aaScore', 'PowerNovo Score', 'Area']


class PeptideAggregator(object):
    def __init__(self,
                 output_folder: str,
                 output_filename: str
                 ):
        self.output_folder = Path(output_folder)
        self.output_filename = output_filename
        self.peptides = {}
        self.config = PWNConfig()
        self.use_alps = self.config.peptides.use_alps
        self.use_protein_inference = self.config.proteins.inference
        self.alps_executable = self.config.assembler_folder / self.config.peptides.running_file
        self.proteins_fasta_path = self.config.proteins.fasta_path
        self.n_contigs = self.config.peptides.n_contigs
        self.kmers = self.config.peptides.kmers
        self.proteins_map_minIdentity = self.config.proteins.minIdentity
        self.contigs = set()
        self.proteins = set()
        self.contigs_protein_map = {}
        self.map_modifications = self.config.proteins.map_modifications

        if self.use_alps and not os.path.exists(self.alps_executable):
            raise FileNotFoundError(f"You specified use_alps = True, "
                                    "but the ALPS.jar executable file was not found "
                                    f"in the assembler folder: {self.config.assembler_folder}")

        if self.use_protein_inference and not os.path.exists(self.proteins_fasta_path):
            raise FileNotFoundError(f"You specified protein inference = True, "
                                    f"but the FASTA file was not found: {self.proteins_fasta_path}")

    def add_record(self,
                   scan_id: str,
                   sequence: str,
                   mass_error: float,
                   score: float,
                   aa_scores: list[float]
                   ):
        if not sequence:
            return
        sequence, mod_dict = from_proforma(sequence)
        canonical_seq = to_canonical(sequence)
        mod_str = ''
        for k, v in mod_dict.items():
            mod_str += f"{k}:{','.join([str(m) for m in v])} "

        if not aa_scores:
            aa_scores = np.full(len(canonical_seq),
                                fill_value=min(1.0, np.round(score * 1.0 / np.exp(abs(mass_error) ** 2)
                                                             / len(sequence), 2)), dtype=np.float32)

        if len(aa_scores) != len(canonical_seq):
            return
        aa_scores = ' '.join(map(str, aa_scores))
        score = min(1.0, round(score / len(sequence), 2))

        self.peptides.update({
            scan_id: {'sequence': sequence,
                      'canonical_seq': canonical_seq,
                      'mass_error': mass_error,
                      'score': score,
                      'mod_dict': mod_dict,
                      'mod_str': mod_str,
                      'aa_scores': aa_scores
                      }
        })

    def solve(self):
        scored_filepath = self.assembly()

        if self.use_protein_inference and self.use_alps:
            self.map_proteins(scored_filepath)
            self.inference_proteins()

    def assembly(self) -> str:
        logger.info('Assembly peptides...')
        logger.info('Prepare predictions...')

        assembly_dataframe = {k: [] for k in ASSEMBLY_COLUMNS}
        modifications_dataframe = {'scan_id': [], 'sequence': [], 'mod': []}

        for scan_id, pep_record in self.peptides.items():
            assembly_dataframe['Spectrum Name'].append(f'{self.output_filename}:{scan_id}')
            assembly_dataframe['PowerNovo Peptides'].append(pep_record['sequence'])
            assembly_dataframe['PowerNovo aaScore'].append(pep_record['aa_scores'])
            assembly_dataframe['PowerNovo Score'].append(pep_record['score'])
            assembly_dataframe['Area'].append(1)

            if self.map_modifications:
                mod_str = pep_record['mod_str']
                if mod_str:
                    modifications_dataframe['scan_id'].append(scan_id)
                    modifications_dataframe['sequence'].append(pep_record['canonical_seq'])
                    modifications_dataframe['mod'].append(mod_str)

        df_local_score = pd.DataFrame.from_dict(assembly_dataframe)
        output_filename = self.output_folder / f'{self.output_filename}_pw_score.csv'
        df_local_score.to_csv(output_filename, index=False)

        if self.map_modifications:
            mod_filepath = self.output_folder / f'{self.output_filename}_mod.csv'
            df_local_mod = pd.DataFrame.from_dict(modifications_dataframe)
            df_local_mod.to_csv(mod_filepath, index=False)

        logger.info(f'Predictions saved: {output_filename}')

        logger.info(f'Run assembler: {self.alps_executable}')

        if self.use_alps:
            log_filepath = str(self.output_folder / f'{self.output_filename}_assembly.log')

            for kmer in self.kmers:
                subprocess.run(
                    ('java', '-jar', f'{self.alps_executable}', str(output_filename),
                     str(kmer), str(self.n_contigs), '>>', log_filepath), stdout=subprocess.DEVNULL)

        logger.info('Peptide assembly completed')

        return str(output_filename)

    def map_proteins(self, scored_filepath: str):
        assembled_fasta_pattern = f'{scored_filepath}*.fasta'
        fasta_files = glob.glob(assembled_fasta_pattern)

        logger.info(f'Start protein inference process with FASTA: {self.proteins_fasta_path}')

        if not fasta_files:
            raise FileNotFoundError(f"Peptide assembler Fasta output "
                                    f"files not found in folder {self.output_folder}")

        for query_path in fasta_files:
            results = npy.blast(query=str(query_path),
                                database=str(self.proteins_fasta_path),
                                minIdentity=float(self.proteins_map_minIdentity),
                                maxAccepts=5,
                                alphabet="protein")

            self._assign_proteins(results)

        if self.map_modifications:
            self.map_peptides_mod()

    def _assign_proteins(self, blast_results: dict):
        try:
            if len(blast_results['QueryId']) == 0:
                return
        except KeyError:
            return

        for i in range(len(blast_results['QueryId'])):
            target_match = blast_results['TargetMatchSeq'][i]
            target_matched_start = blast_results['TargetMatchStart'][i]
            target_matched_end = blast_results['TargetMatchEnd'][i]
            score = blast_results['Identity'][i]
            try:
                target_protein = blast_results['TargetId'][i].split('|')
                protein_id = target_protein[1]
                protein_name = target_protein[2]
            except (IndexError, KeyError, Exception):
                protein_id = protein_name = blast_results['TargetId'][i]

            if target_match in self.contigs:
                continue

            self.contigs.add(target_match)
            self.proteins.add(protein_id)

            self.contigs_protein_map.update({target_match: {
                'protein_id': protein_id,
                'protein_name': protein_name,
                'score': score,
                'start_idx': target_matched_start - 1,
                'end_idx': target_matched_end - 1,
                'mod': ''
            }})

    def map_peptides_mod(self):
        mod_filepath = self.output_folder / f'{self.output_filename}_mod.csv'
        if not os.path.exists(mod_filepath):
            return
        configs_fasta_file = self.output_folder / f'contigs_{uuid.uuid4()}.fasta'
        query_fasta_file = self.output_folder / f'query_{uuid.uuid4()}.fasta'

        try:
            dataframe = pd.read_csv(mod_filepath)
            configs_fasta_rec = {str(i): contig for i, contig in enumerate(self.contigs_protein_map)}

            npy.write_fasta(str(configs_fasta_file), configs_fasta_rec)
            query_fasta_rec = dataframe.set_index('scan_id')['sequence'].to_dict()

            npy.write_fasta(str(query_fasta_file), query_fasta_rec)

            map_results = npy.blast(query=str(query_fasta_file),
                                    database=str(configs_fasta_file),
                                    minIdentity=float(self.proteins_map_minIdentity),
                                    maxAccepts=5,
                                    alphabet="protein")

            if len(map_results['QueryId']) == 0:
                return

            for i in range(len(map_results['QueryId'])):
                try:
                    query_id = map_results['QueryId'][i]
                    contig_ = map_results['TargetMatchSeq'][i]
                    target_match = list(contig_)
                    target_matched_start = int(map_results['TargetMatchStart'][i])
                    query_match = list(map_results['QueryMatchSeq'][i])
                    mod_item = dataframe[dataframe['scan_id'] == query_id]
                    mod_dict = self.__parse_mod(mod_item['mod'].to_list())
                    contig_rec = self.contigs_protein_map[contig_]
                    mod_str = ''
                    for m, positions in mod_dict.items():
                        for pos in positions:
                            pos = int(pos)
                            if target_match[pos + target_matched_start - 1] == query_match[pos]:
                                protein_pos = contig_rec['start_idx'] + pos - int(m.startswith('('))
                                mod_str += f'{m}:{protein_pos} '

                    contig_rec['mod'] = mod_str.strip()

                except (AttributeError, KeyError, AssertionError, IndexError, Exception):
                    continue
        except (AttributeError, KeyError, Exception):
            return

        finally:
            if os.path.exists(configs_fasta_file):
                os.remove(configs_fasta_file)

            if os.path.exists(query_fasta_file):
                os.remove(query_fasta_file)

    @staticmethod
    def __parse_mod(mods: str):
        mods = mods[0].strip()
        mods = mods.split(' ')
        mod_dict = {}
        for mod in mods:
            key, pos = mod.split(':')
            pos = pos.strip()
            pos = pos.split(',')
            mod_dict.update({key: pos})
        return mod_dict

    def inference_proteins(self):
        if not self.contigs_protein_map:
            return

        logger.info('Solve protein problem network')

        inference = ProteinInference(protein_map_records=self.contigs_protein_map,
                                     uniques_contigs=self.contigs,
                                     uniques_proteins=self.proteins,
                                     output_folder=self.output_folder,
                                     output_filename=self.output_filename
                                     )
        inference.solve()
        logger.info('All pipeline task has been completed')
