import datasets
import json
import pandas as pd

def load_data_from_hf(data_files: list[str]) -> pd.DataFrame:
    """
    Load dataset from Hugging Face Hub or local files.
    """
    dataset = datasets.load_dataset("json", data_files=data_files)
    return pd.DataFrame(dataset["train"])

def extract_tag(text: str) -> str:
    """Extract the text between the first and second '|' characters"""
    first = text.find('|')
    if first == -1:
        return set()
    second = text.find('|', first + 1)
    if second == -1:
        return set()
    return text[first + 1:second]

def remove_tags(sequences: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Remove tags from sequences and return cleaned sequences and tags."""
    tags = sequences.apply(extract_tag)
    cleaned_sequences = pd.Series([s1.replace(f"|{s2}|", '') for s1, s2 in zip(sequences, tags)])
    assert not any(cleaned_sequences.str.contains('|', regex=False)), "Some sequences still contain '|' characters"
    return cleaned_sequences, tags

def load_phylotags(file_path: str) -> pd.DataFrame:
    """
    Load a phylotags JSON file into a pandas DataFrame.
    """
    with open(file_path) as f:
        phylotags = json.load(f)
        return pd.DataFrame.from_dict(phylotags, orient="index", columns=["phylotag"]).reset_index().rename(columns={"index": "record"})
    
def preprocess(df: pd.DataFrame, min_length: int, subset: int, random_seed: int) -> pd.DataFrame:
    """Filter sequences by minimum length, shuffle and return a subset."""
    df["sequence_length"] = df["text"].apply(len)
    df = df[df["sequence_length"] > min_length]
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df = df.rename(columns={"text": "sequence"})
    return df.head(subset)

def filter_by_phylotags(df: pd.DataFrame, phylotags_df: pd.DataFrame) -> pd.DataFrame:
    """Filter sequences to keep only those with specific phylotags."""
    all_records = set(df["record"])
    phylotag_records = set(accession_id for accession_id, tag in phylotags_df.itertuples() if not ("C__NONE" in tag and "O__NONE" in tag and "F__NONE" in tag))
    keep_records = all_records & phylotag_records
    return df[df["record"].isin(keep_records)].reset_index(drop=True).rename(columns={"text": "sequence"})