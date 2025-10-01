# %% 
import pandas as pd
import json
import sys
sys.path.append("/root/evo2-mech-interp")

import datasets
import re
from pprint import pprint
from tqdm import tqdm

from utils.phylogenetics import gtdb_pipeline, get_gtdb_tree
from utils.data import load_data_from_hf

# %% - LOAD DATA
data_files = [
    "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/gtdb_v220_stitched/data_gtdb_train_chunk1.jsonl.gz"
]
# # TODO: there don't seem to be any record IDs in the midtraining OpenGenome2 dataset for GTDB, but the species IDs are included in the text

# # df = load_data_from_hf(data_files)
# # load data manually since the species IDs are embedded in the text
# dataset = datasets.load_dataset("json", data_files=data_files)
# df = dataset["train"].to_pandas()
# df.head()

# data_files = [
#     "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/pretraining_or_both_phases/gtdb_v220_imgpr/data_gtdb_train_chunk1.jsonl.gz"
# ]

df = load_data_from_hf(data_files)
df.head()

# %%
def extract_tags(text):
    # Extract the text between the first and second '|' characters
    first = text.find('|')
    if first == -1:
        return set()
    second = text.find('|', first + 1)
    if second == -1:
        return set()
    return text[first + 1:second]


df["species"] = df["tag"].str.split(";").str[-1]
df.head()

# %%
df["record"].nunique()
metadata = pd.read_csv("../data/gtdb/bac120_metadata_r220.tsv", sep="\t")
metadata.head()

# %%
species_to_gtdb_accession = dict(zip(metadata["gtdb_taxonomy"].str.upper(), metadata["accession"]))

# %%

df["gtdb_record"] = df["tag"].map(species_to_gtdb_accession)
df.head()

# %%
from ete3 import Tree
tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree"
tree = Tree(tree_file, format=1, quoted_node_names=True)
# %%

accession1, accession2 = list(df.loc[:1, "gtdb_record"].values)
known_nodes = []
leaf_names = set(tree.get_leaf_names())
for accession in tqdm(df["gtdb_record"].unique()[:100]):
    if accession in leaf_names:
        known_nodes.append(tree&accession)
print(f"Found {len(known_nodes)} known nodes out of {df['gtdb_record'].nunique()} unique accessions.")
# %%
# def extract_tags(text):
#     # Extract the text between the first and second '|' characters
#     first = text.find('|')
#     if first == -1:
#         return set()
#     second = text.find('|', first + 1)
#     if second == -1:
#         return set()
#     return text[first + 1:second]

# df["tag"] = df["text"].apply(extract_tags)
# df.head()

# %%
print(df.describe())

# %%
min_length = df["text"].apply(len).min()
median_length = df["text"].apply(len).median()
max_length = df["text"].apply(len).max()
print(f"Min sequence/text length: {min_length}")
print(f"Median sequence/text length: {median_length}")
print(f"Max sequence/text length: {max_length}")

# %%
pprint(df["text"][0])

# %%
# num_hash = df["text"].str.contains("#").sum()
# num_at = df["text"].str.contains("@").sum()
# num_both = df["text"].str.contains("#") & df["text"].str.contains("@")
# num_both = num_both.sum()

# print(f"Rows with '#': {num_hash}")
# print(f"Rows with '@': {num_at}")
# print(f"Rows with both '#'' and '@': {num_both}")

# %%
# %%
print([col for col in metadata.columns if 'ncbi' in col])

# %%
metadata[[col for col in metadata.columns if 'accession' in col]].head()
# %%
metadata["gtdb_taxonomy"][0]
# %%
print([col for col in metadata.columns if 'seq' in col])
# %%
print(df["tag"][20])
print(df["tag"][21])
print(df["tag"][22])
print(df["tag"][23])
print(df["tag"][24])
print(df["tag"][25])
print(df["tag"][26])
print(df["tag"][27])
# %%
get_gtdb_tree()
# %%
from ete3 import Tree

def _norm(tag: str) -> str:
    """Normalize a tag for comparison (lowercase, trim)."""
    return tag.strip().lower()

def find_node_by_phylotag(tree: Tree, phylotag: str, sep: str = ";"):
    """
    Traverse a GTDB tree using a phylotag like
    'D__BACTERIA;P__PSEUDOMONADOTA;...;S__HYPHOCOCCUS'
    and return the final ete3 Node.

    This version is robust to internal node names that may include branch lengths or node numbers,
    e.g. '100:d__Bacteria', by searching for the label as a substring after splitting on ':' or by matching the label as a substring (ignoring any numeric or other prefix).
    """
    labels = [_norm(x) for x in phylotag.split(sep) if x.strip()]
    node = tree.get_tree_root()

    for label in labels:
        # Find the *closest* descendant (fewest edges away) whose name matches (after splitting on ':'),
        # or whose name ends with the label (to handle weird prefixes like '100:d__Bacteria').
        best = None
        best_depth = None
        node_anc_depth = len(node.get_ancestors())
        for desc in node.iter_descendants():
            name = getattr(desc, "name", None)
            if name:
                # If name contains ':', take the part after the last ':'
                if ':' in name:
                    name_part = name.split(':')[-1]
                else:
                    name_part = name
                # Try exact match first
                print(name_part, label)
                if _norm(name_part) == label:
                    rel_depth = len(desc.get_ancestors()) - node_anc_depth
                    if best is None or rel_depth < best_depth:
                        best, best_depth = desc, rel_depth
                # If not, try matching the label as a substring at the end (to handle weird prefixes)
                elif _norm(name_part).endswith(label):
                    rel_depth = len(desc.get_ancestors()) - node_anc_depth
                    if best is None or rel_depth < best_depth:
                        best, best_depth = desc, rel_depth
                # As a fallback, try matching the label as a substring anywhere (less strict)
                elif label in _norm(name_part):
                    rel_depth = len(desc.get_ancestors()) - node_anc_depth
                    if best is None or rel_depth < best_depth:
                        best, best_depth = desc, rel_depth
        if best is None:
            # Instead of raising, print a helpful error and return None
            print(f"[find_node_by_phylotag] Tag '{label}' not found beneath node '{getattr(node,'name',None)}'.")
            return None
        node = best

    return node

# %%
t = Tree("/root/evo2-mech-interp/data/gtdb/bac120_r220.tree", format=1, quoted_node_names=True)  # or your GTDB Newick file
phylotag = "D__BACTERIA;P__PSEUDOMONADOTA;C__ALPHAPROTEOBACTERIA;O__CAULOBACTERALES;F__PARVULARCULACEAE;G__HYPHOCOCCUS;S__HYPHOCOCCUS"
node = find_node_by_phylotag(t, phylotag)
print(node.name)       # should be 's__Hyphococcus' (or whatever the exact case is)
print(node.get_leaf_names()[:5])  # inspect some leaves under that species

# %%
t.show('GB_GCA_018399855.1')
# %%
t.search_nodes(name='GB_GCA_018399855.1')[0].show()
# %%
t.search_nodes(name='s__Hyphococcus')
# %%