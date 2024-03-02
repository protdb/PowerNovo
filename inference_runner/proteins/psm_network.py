import pprint

import networkx as nx


class PSMNetworkSolver(object):
    def __init__(self, network):

        self.network = network

    def get_proteins(self):
        proteins = list([n for n, d in self.network.nodes(data=True) if d['is_protein'] == 1])
        return proteins

    def get_peptides(self):
        peptides = list([n for n, d in self.network.nodes(data=True) if d['is_protein'] == 0])
        return peptides

    def update_nodes(self, nodes, attribute, value):
        att_dict = dict(zip(nodes, [value] * len(nodes)))
        nx.set_node_attributes(self.network,
                               att_dict,
                               attribute)

        return

    def pick_nodes(self, attribute, value):
        picked = []
        for n, d in self.network.nodes(data=True):
            if attribute in d.keys():
                if d[attribute] == value:
                    picked.append(n)

        return picked

    def print_nodes(self):
        pp = pprint.PrettyPrinter(indent=4, )
        pp.pprint(dict(self.network.nodes(data=True)))

        return

    def print_edges(self):
        pp = pprint.PrettyPrinter(indent=4, )
        edge_rep = {(u[0], u[1]): u[2] for u in self.network.edges(data=True)}
        pp.pprint(edge_rep)

    def get_node_attribute_dict(self, attribute):
        out = {}

        for node in self.network.nodes():
            if attribute in self.network.nodes[node].keys():
                out[node] = self.network.nodes[node][attribute]

        return out

    def get_edge_attribute_dict(self, attribute):
        out = {}

        for edge in self.network.edges():
            if attribute in self.network.edges[edge].keys():
                out[edge] = self.network.edges[edge][attribute]

        return out


