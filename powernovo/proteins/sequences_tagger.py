from networkx import set_node_attributes
from powernovo.proteins.psm_network import PSMNetworkSolver


class SequencesTagger(object):
    def run(self, pn):

        network = self._tag_non_unique_peptides(pn.network)
        network = self._tag_unique_evidenced_protein(network)

        return PSMNetworkSolver(network)

    @staticmethod
    def _tag_non_unique_peptides(network):

        pn = PSMNetworkSolver(network)
        peptides = pn.get_peptides()
        list_uniqueness = list([1 == len(list(network.neighbors(peptide))) for
                                peptide in peptides])
        dict_uniqueness = dict(zip(peptides, list_uniqueness))

        set_node_attributes(network, dict_uniqueness, "unique")

        return network

    @staticmethod
    def _tag_unique_evidenced_protein(network):
        unique_evidence = {}

        pn = PSMNetworkSolver(network)

        for protein in pn.get_proteins():
            n_unique_pep_neighbours = 0
            for n, d in network.nodes(data=True):
                if n in network.neighbors(protein):
                    if d['unique'] == 1:
                        n_unique_pep_neighbours = n_unique_pep_neighbours + 1

            if n_unique_pep_neighbours > 0:
                unique_evidence[protein] = True
            else:
                unique_evidence[protein] = False

        set_node_attributes(network, unique_evidence, "unique_evidence")

        return network
