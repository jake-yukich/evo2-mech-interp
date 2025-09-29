import os

from datasets import load_dataset


def main():
    data_dir = "data/gtdb_v220_imgpr"
    os.makedirs(data_dir, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/pretraining_or_both_phases/gtdb_v220_imgpr/*.jsonl.gz",
        split="train",
        cache_dir=data_dir,
    )

    print(f"Downloaded {len(dataset)} records to {data_dir}")


if __name__ == "__main__":
    main()
