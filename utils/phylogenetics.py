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

def record_to_phylotag(record: str) -> str | None: # make_phylotag_from_gbif Evo 2 util claims to return dict but doesn't?
    """
    Single record -> phylotag, using Evo 2's GBIF function.

    For example
        input:  "PAHO01000009.1"
        output: "|D__BACTERIA;P__PLANCTOMYCETOTA;C__PHYCISPHAERAE;O__NONE;F__NONE;G__NONE;S__NONE|"
    """
    try:
        handle = Entrez.efetch(db="nucleotide", id=record, rettype="gb", retmode="xml")
        record = Entrez.read(handle)[0]
        handle.close()

        organism = record.get('GBSeq_organism', '')

        if organism:
            time.sleep(0.1) # might be able to get away with less than 0.5
            return make_phylotag_from_gbif(organism)

    except Exception as e:
        print(f"Error getting phylotag for {record}: {e}")
        return None


def record_to_organism(record: str) -> str | None:
    """Record to organism name, for accessing GTDB metadata."""
    try:
        handle = Entrez.efetch(db="nucleotide", id=record, rettype="gb", retmode="xml")
        record = Entrez.read(handle)[0]
        handle.close()
        # time.sleep(0.1)
        return record.get('GBSeq_organism', '')
    except Exception as e:
        print(f"Error getting organism for {record}: {e}")
        return None


def organisms_to_gtdb(records: list[str], metadata: pd.DataFrame) -> dict:
    """
    Returns a dictionary mapping of records to their GTDB accession, taxonomy, and organism:
    {
        "record": {
            "gtdb_accession": "accession",
            "gtdb_taxonomy": "taxonomy",
            "organism": "organism",
        }
    }

    If no match is found, the value is None.
    """
    organisms = [record_to_organism(record) for record in records]
    mappings = {}

    for record, organism in zip(records, organisms):
        exact_matches = metadata[metadata["ncbi_organism_name"] == organism]
        m = exact_matches.iloc[0]

        if m:
            mappings[record] = {
                "gtdb_accession": m["accession"],
                "gtdb_taxonomy": m["gtdb_taxonomy"],
                "organism": organism,
            }
        else:
            mappings[record] = {
                "gtdb_accession": None,
                "gtdb_taxonomy": None,
                "organism": organism,
            }

    return mappings


def get_gtdb_tree(tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree"):
    """
    Returns a tuple of (distance_matrix, mapped_records), where
        distance_matrix: (n, n) array of phylogenetic distances between GTDB accessions
        mapped_records: list of records that were successfully mapped
    """

    print("Loading GTDB tree...")
    assert os.path.exists(tree_file)
    
    # Try different format options to handle the GTDB tree format
    try:
        # First try with format=1 and quoted_node_names=True
        tree = Tree(tree_file, format=1, quoted_node_names=True)
        print("Successfully loaded tree with format=1, quoted_node_names=True")
    except Exception as e1:
        print(f"Failed with format=1, quoted_node_names=True: {e1}")
        # try:
        #     # Try with format=0 (flexible with support values)
        #     tree = Tree(tree_file, format=0)
        #     print("Successfully loaded tree with format=0")
        # except Exception as e2:
        #     print(f"Failed with format=0: {e2}")
        #     try:
        #         # Try with format=2 (all branches, leaf names, and internal supports)
        #         tree = Tree(tree_file, format=2)
        #         print("Successfully loaded tree with format=2")
        #     except Exception as e3:
        #         print(f"Failed with format=2: {e3}")
        #         try:
        #             # Try with format=3 (all branches and all names)
        #             tree = Tree(tree_file, format=3)
        #             print("Successfully loaded tree with format=3")
        #         except Exception as e4:
        #             print(f"Failed with format=3: {e4}")
        #             # Last resort: try without any format specification
        #             tree = Tree(tree_file)
        #             print("Successfully loaded tree with default format")

    # tree.search_nodes()
    # Visualize the ete3 tree so we can see node names and structure
    try:
        from ete3 import TreeStyle, NodeStyle, faces, AttrFace
        import matplotlib.pyplot as plt
        
        def show_tree(tree, max_nodes=100):
            """
            Visualize the ete3 tree, showing node names. 
            Only shows up to max_nodes leaves for performance.
            """
            ts = TreeStyle()
            ts.show_leaf_name = True
            ts.show_scale = True
            ts.scale = 120
            ts.branch_vertical_margin = 10
            ts.title.add_face(faces.TextFace("GTDB Tree (showing up to %d leaves)" % max_nodes, fsize=14), column=0)
            ts.layout_fn = None

            # Optionally collapse the tree if too large
            leaves = tree.get_leaves()
            if len(leaves) > max_nodes:
                # Prune to a random subset for visualization
                import random
                keep = set(random.sample(leaves, max_nodes))
                tree = tree.copy()
                for leaf in tree.get_leaves():
                    if leaf not in keep:
                        leaf.delete(prevent_nondicotomic=False)

            # Set node styles to show names
            for n in tree.traverse():
                nstyle = NodeStyle()
                nstyle["size"] = 0
                n.set_style(nstyle)
                if n.is_leaf():
                    n.add_face(AttrFace("name", fsize=10), column=0, position="branch-right")

            tree.show(tree_style=ts)
        
        show_tree(tree, max_nodes=10)
    except ImportError:
        print("ete3 visualization components not available, skipping tree visualization")

    # Basic tree information
    print(f"Tree loaded successfully!")
    print(f"Number of leaves: {len(tree.get_leaves())}")
    print(f"Tree height: {tree.get_farthest_leaf()[1]}")
    
    # Show a few sample leaf names
    sample_leaves = tree.get_leaves()[:5]
    print(f"Sample leaf names: {[leaf.name for leaf in sample_leaves]}")

    # given a tag like "D__BACTERIA;P__PSEUDOMONADOTA;C__ALPHAPROTEOBACTERIA;O__CAULOBACTERALES;F__PARVULARCULACEAE;G__HYPHOCOCCUS;S__HYPHOCOCCUS" let's check out if we can retrieve the node from the tree and print some info about it
    # Example tag
    example_tag = "D__BACTERIA;P__PSEUDOMONADOTA;C__ALPHAPROTEOBACTERIA;O__CAULOBACTERALES;F__PARVULARCULACEAE;G__HYPHOCOCCUS;S__HYPHOCOCCUS"

    def find_node_by_tag(tree, tag):
        """
        Given a GTDB-style tag string, traverse the tree to find the corresponding node.
        The tag is a semicolon-separated list of taxonomic ranks, e.g.:
        "D__Bacteria;P__Pseudomonadota;C__Alphaproteobacteria;O__Caulobacterales;F__Parvularculaceae;G__Hyphococcus;S__Hyphococcus"
        """
        # Split the tag into levels
        levels = tag.split(";")
        # Clean up whitespace and ensure consistent case
        levels = [level.strip() for level in levels if level.strip()]
        # The tree is usually named by the last non-empty level
        if not levels:
            return None
        # Try to find the node by the most specific name (species, genus, etc.)
        for i in range(len(levels), 0, -1):
            name = levels[i-1]
            matches = tree.search_nodes(name=name)
            if matches:
                return matches[0]
        return None

    node = find_node_by_tag(tree, example_tag)
    if node is not None:
        print(f"Node found for tag: {example_tag}")
        print(f"Node name: {node.name}")
        print(f"Is leaf: {node.is_leaf()}")
        print(f"Number of children: {len(node.children)}")
        print(f"Distance to root: {node.get_distance(tree)}")
        # Print lineage up to root
        lineage = []
        n = node
        while n is not None:
            lineage.append(n.name)
            n = n.up
        print("Lineage (from node to root):")
        print(" -> ".join(lineage))
    else:
        print(f"No node found for tag: {example_tag}")

    # Show the tree for inspection (comment out if running headless)
    # show_tree(tree)

    # mapped_records = [record for record, mapping in mappings.items() if mapping["gtdb_accession"] is not None]
    # mapped_gtdb_accessions = [mapping["gtdb_accession"] for mapping in mappings.values() if mapping["gtdb_accession"] is not None]

    # # create a distance matrix
    # n = len(mapped_records)
    # distance_matrix = np.zeros((n, n))
    
    # print(f"Calculating distance matrix for {n} records...")
    # for i in tqdm(range(n)):
    #     for j in range(i+1, n):
    #         gtdb_accession_i = mapped_gtdb_accessions[i]
    #         gtdb_accession_j = mapped_gtdb_accessions[j]

    #         node_i = tree.search_nodes(name=gtdb_accession_i)[0]
    #         node_j = tree.search_nodes(name=gtdb_accession_j)[0]

    #         distance_matrix[i, j] = distance_matrix[j, i] = node_i.get_distance(node_j)
    
    # return distance_matrix, mapped_records


def gtdb_pipeline(records: list[str]) -> dict:
    # TODO: at some point implement getting the tree and metadata programmatically
    # (currently had been downloaded manually)
    
    metadata = pd.read_csv("data/gtdb/bac120_metadata_r220.tsv.gz", sep="\t")
    mappings = organisms_to_gtdb(records, metadata)

    success_rate = sum(1 for mapping in mappings.values() if mapping["gtdb_accession"] is not None) / len(mappings)
    print(f"Mapping success rate: {success_rate:.2f}")

    distance_matrix, mapped_records = get_gtdb_tree(mappings)

    # TODO: tree building (e.g. using scipy.cluster.hierarchy)

    return {
        "distance_matrix": distance_matrix,
        "mapped_records": mapped_records,
    }


# use the phylo tags from midtraining OpenGenome2 data to traverse the GTDB tree
# return a node, which downstream will be used to get the distance from another node