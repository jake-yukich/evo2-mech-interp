import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from ete3 import Tree
from Bio import Entrez
Entrez.email = "user.name@gmail.com"

import time
from evo2.utils import make_phylotag_from_gbif

def get_tag_to_gtdb_accession_map(metadata_path: str = "../data/gtdb/bac120_metadata_r220.tsv") -> dict:
    """
    Returns a mapping of species names to GTDB accessions.
    """
    metadata = pd.read_csv(metadata_path, sep="\t")
    return dict(zip(metadata["gtdb_taxonomy"].str.upper(), metadata["accession"]))


def get_tree(tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree") -> Tree:
    """
    Loads and returns the GTDB tree.
    """
    assert os.path.exists(tree_file)
    tree = Tree(tree_file, format=1, quoted_node_names=True)
    return tree


def get_tree_leaf_names(tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree") -> set[str]:
    """
    Returns a set of all leaf names in the GTDB tree.
    """
    tree = get_tree(tree_file)
    return set(tree.get_leaf_names())


def filter_genomes_in_tree(df: pd.DataFrame, tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree", accession_column: str = "gtdb_accession") -> pd.DataFrame:
    """Filter dataframe to only keep genomes whose accession IDs are in the phylogenetic tree."""
    print(f"Loading tree to validate accession IDs...")
    tree_leaves = get_tree_leaf_names(tree_file)
    print(f"Tree contains {len(tree_leaves)} leaves")
    
    # Filter to genomes that are in the tree
    initial_count = len(df)
    df_filtered = df[df[accession_column].isin(tree_leaves)].reset_index(drop=True)
    final_count = len(df_filtered)
    
    dropped_count = initial_count - final_count
    if dropped_count > 0:
        print(f"Filtered out {dropped_count} genomes not found in tree ({dropped_count/initial_count*100:.1f}%)")
        print(f"Kept {final_count} genomes that are in the tree")
    else:
        print(f"All {final_count} genomes are in the tree")
    
    return df_filtered
