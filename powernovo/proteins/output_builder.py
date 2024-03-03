from pandas import concat, DataFrame
from multiprocessing import cpu_count, Pool
import pandas as pd


class TableMaker(object):
    def get_protein_table(self, pn):
        # source the data as a list
        df = self._get_edge_list(pn)

        # perform protein level aggregation
        agg_dict = self._get_agg_dict(pn)
        protein_df = df.groupby("protein_id").agg(agg_dict)

        # calculate new columns
        protein_df["total_peptides"] = df.groupby("protein_id").size()

        # if we know unique
        if pn.get_node_attribute_dict("unique"):
            protein_df["non_unique"] = protein_df.total_peptides - protein_df.unique
            protein_df.non_unique = protein_df.non_unique.astype("int32")

        for col in ["razor", "non_unique", "unique"]:
            if pn.get_node_attribute_dict(col):
                protein_df[col] = protein_df[col].astype("int32")

        if pn.get_node_attribute_dict("non_unique"):
            protein_df.non_unique = protein_df.non_unique.astype("int32")

        if pn.get_node_attribute_dict("non_unique"):
            protein_df.non_unique = protein_df.non_unique.astype("int32")

        if pn.get_node_attribute_dict("modifications"):
            protein_df['modifications'] = protein_df.name.astype("str")

        if pn.get_node_attribute_dict("name"):
            protein_df['name'] = protein_df.name.astype("str")

        # sort sequence modified
        protein_df.sequence_modified = protein_df.sequence_modified.apply(lambda x: sorted(x))

        protein_df = protein_df.reset_index()  # otherwise protein id isn't a column

        # if solved, add new scores:
        dict_score = pn.get_node_attribute_dict("score")
        if dict_score:
            protein_df["score"] = protein_df.protein_id.apply(
                lambda x: dict_score[x])

        # if solved, add subset proteins:
        dict_subset = pn.get_node_attribute_dict("major")
        if dict_subset:
            protein_df["Group"] = protein_df.protein_id.apply(
                lambda x: dict_subset[x])

        if pn.get_node_attribute_dict("indistinguishable"):
            protein_df = self.add_indistinguishable_col(pn, protein_df)
            protein_df.indistinguishable = protein_df.indistinguishable.apply(lambda x: sorted(x))

        if pn.get_node_attribute_dict("major"):
            protein_df = self.add_subset_col(pn, protein_df)
            protein_df.subset = protein_df.subset.apply(lambda x: sorted(x))

        cols = ["protein_id",
                "name",
                "unique",
                "non_unique",
                "razor",
                "total_peptides",
                "score",
                "modifications",
                "Group",
                "indistinguishable",
                "subset",
                "sequence_modified"]

        new_cols = []
        for col in cols:
            if col in protein_df.columns:
                new_cols.append(col)

        protein_df = protein_df.loc[:, new_cols]

        if "score" in protein_df.columns:
            return protein_df.sort_values("score", ascending=False)
        else:
            return protein_df

    def get_protein_tables(self, pns):

        p = Pool(cpu_count())
        protein_tables = p.map(self.get_protein_table, pns)

        return protein_tables

    def get_system_protein_table(self, pns):
        protein_table = concat(self.get_protein_tables(pns))
        protein_table = self.emulate_percolator_formatting(protein_table)
        return protein_table

    @staticmethod
    def get_peptide_table(pn):

        def label_score(protein, score_dict):
            return score_dict[protein]

        score_dict = pn.get_node_attribute_dict("score")

        df = TableMaker()._get_edge_list(pn)
        df["protein_score"] = df.apply(lambda row: label_score(row["protein_id"],
                                                               score_dict), axis=1)

        df = df.sort_values(["sequence_modified", "protein_score"],
                            ascending=[True, False])

        # collapse rows to peptide result
        agg_dict = {"unique": 'min',
                    "razor": 'min',
                    "unique_evidence": 'max',
                    "major": 'min',
                    "score": 'min',
                    "modifications": list,
                    'protein_name': 'min',
                    "protein_score": 'max',
                    "protein_id": list
                    }
        df = df.groupby("sequence_modified").aggregate(agg_dict).reset_index()

        df = df.rename(columns={"sequence_modified": "sequence",
                                "protein_name": "major_name",
                                "protein_id": "all_proteins"})

        # note that the q-value gets overwritten by FDR calculator in runner
        cols = ['sequence',
                'modifications',
                'unique',
                'razor',
                'unique_evidence',
                'score',
                'major',
                'major_name',
                'protein_score',
                "all_proteins"]

        new_cols = []
        for col in cols:
            if col in df.columns:
                new_cols.append(col)

        df = df.loc[:, new_cols]
        return df

    def get_peptide_tables(self, pns):

        p = Pool(cpu_count())
        peptide_tables = p.map(self.get_peptide_table, pns)

        return peptide_tables

    def get_system_peptide_table(self, pns):
        peptide_table = concat(self.get_peptide_tables(pns))
        return peptide_table

    @staticmethod
    def _get_edge_list(pn):

        rows = []
        for u, v, d in pn.network.edges(data=True):
            node_1_data = pn.network.nodes[u]
            node_2_data = pn.network.nodes[v]
            row = dict()
            if node_1_data["is_protein"]:
                row["protein_id"] = u
                row["sequence_modified"] = v
            else:
                row["protein_id"] = v
                row["sequence_modified"] = u
            row.update(node_1_data)
            row.update(node_2_data)
            row.update(d)
            rows.append(row)

        df = DataFrame(rows)

        return df.drop("is_protein", axis=1)

    def _flip_dict(self, old_dict):
        new_dict = {}
        for key, value in old_dict.items():
            if value in new_dict:
                new_dict[value].append(key)
            else:
                new_dict[value] = [key]
        return new_dict

    @staticmethod
    def add_indistinguishable_col(pn, table):

        indistinguishable_dict = pn.get_node_attribute_dict("indistinguishable")
        new_col = []
        for _, row in table.iterrows():
            new_col.append(indistinguishable_dict[row["protein_id"]])

        table["indistinguishable"] = new_col

        return table

    def add_subset_col(self, pn, table):

        subset_dict = pn.get_node_attribute_dict("major")
        subset_dict = self._flip_dict(subset_dict)
        new_col = []
        for _, row in table.iterrows():
            if row["protein_id"] == row["Group"]:  # and row["protein_id"] in subset_dict.keys():
                new_col.append(subset_dict[row["protein_id"]])
            else:
                new_col.append([])

        table["subset"] = new_col

        return table

    @staticmethod
    def _get_agg_dict(pn):
        agg_dict = {}
        if pn.get_node_attribute_dict("razor"):
            agg_dict.update({"razor": 'sum'})
        if pn.get_node_attribute_dict("unique"):
            agg_dict.update({"unique": 'sum'})
        if pn.get_node_attribute_dict("score"):
            agg_dict.update({"score": 'sum'})

        # always add sequence_modified
        agg_dict.update({"sequence_modified": list})
        agg_dict.update({"modifications": list})
        agg_dict.update({"name": "first"})

        return agg_dict

    @staticmethod
    def emulate_percolator_formatting(protein_table):
        col_dict = {"protein_id": "ProteinId", "sequence_modified": "peptideIds"}
        protein_table = protein_table.rename(columns=col_dict)
        protein_table = protein_table.sort_values("ProteinId")
        protein_table["peptideIds"] = protein_table.peptideIds.apply(lambda x: " ".join(x))
        labels, _ = pd.factorize(protein_table.Group)
        protein_table["ProteinGroupId"] = labels

        return protein_table
