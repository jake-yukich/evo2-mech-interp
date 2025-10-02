import datasets
import pandas as pd


def load_data_from_hf(data_files: list[str]) -> pd.DataFrame:
    """
    Load dataset from Hugging Face Hub or local files.
    """
    dataset = datasets.load_dataset("json", data_files=data_files)
    return pd.DataFrame(dataset["train"])


def extract_tag(text: str) -> str:
    """Extract the text between the first and second '|' characters"""
    first = text.find("|")
    if first == -1:
        return ""
    second = text.find("|", first + 1)
    if second == -1:
        return ""
    return text[first + 1 : second]


def extract_tags(sequences: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Remove tags from sequences and return cleaned sequences and tags."""
    tags = sequences.apply(extract_tag)
    cleaned_sequences = pd.Series(
        [s1.replace(f"|{s2}|", "") for s1, s2 in zip(sequences, tags)]
    )
    assert not any(cleaned_sequences.str.contains("|", regex=False)), (
        "Some sequences still contain '|' characters"
    )
    return cleaned_sequences, tags


def preprocess(
    df: pd.DataFrame, min_length: int, subset: int, random_seed: int
) -> pd.DataFrame:
    """Filter sequences by minimum length, shuffle and return a subset."""
    df["sequence_length"] = df["text"].apply(len)
    df = df[df["sequence_length"] > min_length]
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df = df.rename(columns={"text": "sequence"})
    return df.head(subset)


def preprocess_gtdb_sequences(
    df: pd.DataFrame, min_length: int, subset: int, random_seed: int, remove_tags: bool
) -> pd.DataFrame:
    """Filter sequences by minimum length, shuffle, remove tags, and extract taxonomic columns."""
    print("Preprocessing data...")
    df["sequence_length"] = df["text"].apply(len)
    filtered_df = df[df["sequence_length"] > min_length]
    shuffled_df = filtered_df.sample(frac=1, random_state=random_seed).reset_index(
        drop=True
    )
    shuffled_df = shuffled_df.rename(columns={"text": "sequence"})

    # Remove tags and extract them
    sequences_with_tags_removed, tags = extract_tags(shuffled_df["sequence"])
    if remove_tags:
        shuffled_df["sequence"] = sequences_with_tags_removed
    shuffled_df["tags"] = tags

    # Extract taxonomic columns from tags
    shuffled_df["class"] = shuffled_df["tags"].str.split(";").str[2]
    shuffled_df["order"] = shuffled_df["tags"].str.split(";").str[3]
    shuffled_df["family"] = shuffled_df["tags"].str.split(";").str[4]

    return shuffled_df.head(subset)


def add_gtdb_accession(df: pd.DataFrame, tag_to_accession_map: dict) -> pd.DataFrame:
    """Add GTDB accession IDs to the dataframe."""
    df["gtdb_accession"] = df["tags"].map(tag_to_accession_map)
    return df
