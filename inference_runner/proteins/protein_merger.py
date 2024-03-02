from pandas import Series, DataFrame
from networkx import Graph
from operator import itemgetter


class ProteinMerger(object):
    def run(self, pn):

        mapping_df = self.get_named_proteins(pn)

        for _, row in mapping_df.iterrows():
            named_protein = row["named"]
            pn.network.nodes[named_protein]["indistinguishable"] = sorted(row["protein"])
            pn.network = Graph(pn.network)
            to_remove = row["protein"]
            pn.network.remove_nodes_from(to_remove)

        return pn

    @staticmethod
    def _list_to_string(l):
        return ";".join(l)

    @staticmethod
    def _string_to_list(s):
        return s.split(";")

    def get_mapping(self, pn):

        s1 = Series(pn.get_proteins())
        s2 = s1.apply(lambda x: list(pn.network.neighbors(x)))
        s2 = s2.apply(lambda x: self._list_to_string(sorted(x)))
        df = DataFrame([s1, s2]).T
        df.columns = ["protein", "peptides"]
        df = df.groupby("peptides").agg({"protein": list}).reset_index()
        df.peptides = df.peptides.apply(lambda x: self._string_to_list(x))
        df.protein = df.protein.apply(lambda x: sorted(x))
        return df

    def get_named_proteins(self, pn):

        mapping_df = self.get_mapping(pn)
        named_proteins = []
        for _, row in mapping_df.iterrows():
            score_dict = dict(zip(row["protein"], [pn.network.nodes[x]["score"] for x in row["protein"]]))
            best_scoring_protein = max(score_dict.items(), key=itemgetter(1))[0]
            best_scoring_proteins = [k for k, v in score_dict.items() if v == score_dict[best_scoring_protein]]
            named_proteins.append(sorted(best_scoring_proteins)[0])

            row["protein"].remove(best_scoring_protein)

        mapping_df["named"] = named_proteins
        return mapping_df
