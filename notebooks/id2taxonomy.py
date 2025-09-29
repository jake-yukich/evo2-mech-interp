import time

import pandas as pd
from Bio import Entrez

# Set your email for NCBI
Entrez.email = ""


def get_taxonomy_from_accession(accession_list, batch_size=200):
    """
    Retrieve taxonomic information for a list of accessions
    """
    taxonomy_data = []

    # Process in batches to respect NCBI limits
    for i in range(0, len(accession_list), batch_size):
        batch = accession_list[i : i + batch_size]

        try:
            # Search for the records
            search_handle = Entrez.esearch(
                db="nucleotide", term=" OR ".join(batch), retmax=batch_size
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            if search_results["IdList"]:
                # Fetch the records
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    id=search_results["IdList"],
                    rettype="gb",
                    retmode="xml",
                )
                records = Entrez.read(fetch_handle)
                fetch_handle.close()

                # Extract taxonomy for each record
                for record in records:
                    accession = record["GBSeq_primary-accession"]
                    organism = record.get("GBSeq_organism", "Unknown")
                    taxonomy = record.get("GBSeq_taxonomy", "").split("; ")

                    taxonomy_dict = {
                        "accession": accession,
                        "organism": organism,
                        "kingdom": taxonomy[0] if len(taxonomy) > 0 else "Unknown",
                        "phylum": taxonomy[1] if len(taxonomy) > 1 else "Unknown",
                        "class": taxonomy[2] if len(taxonomy) > 2 else "Unknown",
                        "order": taxonomy[3] if len(taxonomy) > 3 else "Unknown",
                        "family": taxonomy[4] if len(taxonomy) > 4 else "Unknown",
                        "genus": taxonomy[5] if len(taxonomy) > 5 else "Unknown",
                        "full_taxonomy": "; ".join(taxonomy),
                    }
                    taxonomy_data.append(taxonomy_dict)
            else:
                print(f"No records found for batch starting with {batch[0]}")

            # Be nice to NCBI servers
            time.sleep(0.5)

        except Exception as e:
            raise e

    return pd.DataFrame(taxonomy_data)


if __name__ == "__main__":
    # Example usage
    record_numbers = ["NC_000913.3", "NC_000964.3"]  # Your record numbers
    taxonomy_df = get_taxonomy_from_accession(record_numbers)
    taxonomy_df.to_csv("taxonomy_data.csv", index=False)
