from operator import itemgetter
from multiprocessing import cpu_count, Pool


class ProteinInferenceGreedySolver(object):
    def run(self, pn):

        for peptide in pn.get_peptides():
            pn.network.nodes[peptide]["allocated"] = 0

        for protein in pn.get_proteins():
            pn.network.nodes[protein]["major"] = 0

        n_allocated = 0
        circuit_breaker = 0
        max_iterations = 100

        while n_allocated < len(pn.get_peptides()):
            if len(self.get_unique_non_major_proteins(pn)) > 0:
                unique_only = 1
            else:
                unique_only = 0

            allocation = self.get_highest_scoring_protein(pn, unique_only)
            pn.network.nodes[allocation]["major"] = allocation
            pn.network.nodes[allocation]["score"] = self.score_protein(
                pn, allocation)

            pn = self.peptide_allocator(pn, allocation)  # dangerous line
            pn = self.protein_allocator(pn, allocation)
            n_allocated = sum(
                [i != 0 for i in pn.get_node_attribute_dict("allocated").values()])

            circuit_breaker = circuit_breaker + 1
            if circuit_breaker > max_iterations:
                break

        for protein in pn.get_proteins():
            if "score" not in pn.network.nodes[protein].keys():
                pn.network.nodes[protein]["score"] = 0

        pn = self._tag_razor(pn)

        return pn

    def running_wide_mode(self, pns):
        p = Pool(cpu_count())
        solved_pns = p.map(self.run, pns)

        return solved_pns

    def get_highest_scoring_protein(self, pn, unique_only=False):

        score_dict = self.score_all_proteins(pn, unique_only=unique_only)
        best_scoring_protein = max(score_dict.items(), key=itemgetter(1))[0]
        # deal with same score proteins! (id's should be unique even if scores aren't)
        best_scoring_proteins = [
            k for k, v in score_dict.items() if v == score_dict[best_scoring_protein]]

        return sorted(best_scoring_proteins)[0]

    def score_all_proteins(self, pn, unique_only=False):
        score_dict = {}
        if unique_only:
            proteins = pn.pick_nodes("unique_evidence", True)
        else:
            proteins = pn.get_proteins()

        for protein in proteins:
            score_dict[protein] = self.score_protein(pn, protein)

        return score_dict

    @staticmethod
    def score_protein(pn, protein):
        score = 0

        for peptide in pn.network.neighbors(protein):
            if not pn.network.nodes[peptide]["allocated"]:
                if pn.network.nodes[peptide]["unique"]:
                    score = score + pn.network.edges[peptide, protein]["score"]
                else:
                    score = score + pn.network.edges[peptide, protein]["score"]

        if protein in pn.get_node_attribute_dict("allocated").values():
            return -10

        return score

    @staticmethod
    def peptide_allocator(pn, protein):
        for peptide in pn.network.neighbors(protein):
            if not pn.network.nodes[peptide]["allocated"]:
                pn.network.nodes[peptide]["allocated"] = protein

        return pn

    @staticmethod
    def protein_allocator(pn, allocate):
        peptide_neighbours = set(pn.network.neighbors(allocate))
        protein_over_neighbours = []
        for peptide in peptide_neighbours:
            for protein_neighbour in pn.network.neighbors(peptide):
                protein_over_neighbours.append(protein_neighbour)

        protein_over_neighbours = set(protein_over_neighbours)
        unallocated_proteins = set(pn.pick_nodes("major", 0))

        unallocated_peptides = pn.pick_nodes("allocated", 0)

        for protein_neighbour in protein_over_neighbours:
            if protein_neighbour in unallocated_proteins:
                peptide_set = set(pn.network.neighbors(protein_neighbour))
                peptide_set = peptide_set.intersection(unallocated_peptides)
                if peptide_set.issubset(peptide_neighbours):
                    pn.network.nodes[protein_neighbour]["major"] = allocate

        return pn

    @staticmethod
    def get_unique_non_major_proteins(pn):
        set_of_unique_evidenced = set(pn.pick_nodes("unique_evidence", True))
        set_of_allocated = set(pn.get_node_attribute_dict("major").values())
        set_unique_non_major = [
            p for p in set_of_unique_evidenced if p not in set_of_allocated]
        return set_unique_non_major

    @staticmethod
    def _tag_razor(pn):
        for peptide in pn.get_peptides():
            neighbour_groups = []
            for protein in pn.network.neighbors(peptide):
                neighbour_groups.append(pn.network.nodes[protein]["major"])
            if len(set(neighbour_groups)) > 1:  # could be dense but clearer this way
                pn.network.nodes[peptide]["razor"] = False
            else:
                pn.network.nodes[peptide]["razor"] = True

        return pn
