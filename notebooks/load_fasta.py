from Bio import SeqIO
# from huggingface_hub import hf_hub_download
from datasets import load_dataset
import gzip

def load_fasta_from_hf():
    # Download the fasta file from Hugging Face Hub
    import tarfile
    import tempfile
    import os
    from huggingface_hub import hf_hub_download

    # Download the tar file from Hugging Face Hub
    tar_path = hf_hub_download(
        repo_id="arcinstitute/opengenome2",
        filename="batch1.tar",
        subfolder="fasta/gtdb_v220/v214"
    )

    # Extract FASTA files from the tar archive to a temporary directory
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=temp_dir)

    # Find all FASTA files in the extracted directory
    fasta_files = [
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if f.endswith(".fasta") or f.endswith(".fa") or f.endswith(".fna") or f.endswith(".gz")
    ]
    # Open the gzipped FASTA file
    with gzip.open(file_path, "rt") as file:
        for record in SeqIO.parse(file, "fasta"):
            print(record.id)
            print(record.seq)
            # Try to extract taxonomic info from the description/header
            description = record.description
            tax_info = None
            # Example: >id tax=Kingdom;Phylum;Class;Order;Family;Genus;Species
            if "tax=" in description:
                tax_info = description.split("tax=")[-1].split()[0]
                print("Taxonomy:", tax_info)
            else:
                print("Description:", description)

# Example usage:
# The file is in: 
# https://huggingface.co/datasets/arcinstitute/opengenome2/tree/main/fasta/gtdb_v220/v214/gtdb_v220_genomes_v214_20240925.fasta.gz
repo_id = "arcinstitute/opengenome2"
filename = "gtdb_v220_genomes_v214_20240925.fasta.gz"
subfolder = "fasta/gtdb_v220/v214"
load_fasta_from_hf()