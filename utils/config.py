from dataclasses import dataclass, asdict
from typing import Literal, Optional
import json
import hashlib
from pathlib import Path


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment run.
    
    All parameters that affect the output should be included here to ensure
    proper cache validation.
    """
    # Model configuration
    model: Literal["1b", "7b"]
    layer_name: str
    batch_size: int
    
    # Data configuration
    num_species: int
    random_seed: int
    data_sources: list[str]  # URL or identifier for the dataset
    min_sequence_length: int
    remove_tags: bool
    
    # Sampling configuration
    region_length: int
    num_samples: Optional[int] = None
    coverage_fraction: Optional[float] = None
    
    # Embedding aggregation
    average_over_last_bp: int = 2000
    
    # GTDB configuration
    gtdb_tree_path: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree"
    gtdb_metadata_path: str = "/root/evo2-mech-interp/data/gtdb/bac120_metadata_r220.tsv"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_samples is None and self.coverage_fraction is None:
            raise ValueError("Must specify either num_samples or coverage_fraction")
        if self.num_samples is not None and self.coverage_fraction is not None:
            raise ValueError("Cannot specify both num_samples and coverage_fraction")
        if self.model not in ["1b", "7b"]:
            raise ValueError(f"Model must be '1b' or '7b', got {self.model}")
    
    @property
    def d_model(self) -> int:
        return 1920 if self.model == "1b" else 4096
    
    @property
    def model_name(self) -> str:
        return "evo2_1b_base" if self.model == "1b" else "evo2_7b"
    
    @property
    def sampling_str(self) -> str:
        if self.num_samples is not None:
            return f"num_samples_{self.num_samples}"
        else:
            return f"coverage_frac_{self.coverage_fraction}"
    
    @property
    def config_hash(self) -> str:
        config_dict = asdict(self)
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]
    
    def get_cache_path(self, base_dir: str = "data/experiments") -> Path:
        """
        Get the cache directory path for this configuration.
        
        Structure:
            data/experiments/
                <config_hash>/
                    experiment_config.json
                    genomes_metadata.csv
                    sampled_regions.csv
                    embeddings/
                        <genome_hash_1>.pt
                        <genome_hash_2>.pt
                        ...
                    all_genomes_embeddings.pt
                    phylogenetic_distance_matrix.pt
        
        Args:
            base_dir: Base directory for all experiments
            
        Returns:
            Path object for the experiment directory
        """
        # Use config hash as the primary directory name for clean organization
        experiment_dir = Path(base_dir) / self.config_hash
        return experiment_dir
    
    def get_embeddings_dir(self, base_dir: str = "data/experiments") -> Path:
        return self.get_cache_path(base_dir) / "embeddings"
    
    def save(self, cache_path: Path):
        cache_path.mkdir(parents=True, exist_ok=True)
        config_file = cache_path / "experiment_config.json"
        with open(config_file, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, cache_path: Path) -> "ExperimentConfig":
        config_file = cache_path / "experiment_config.json"
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def validate_cache(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        
        config_file = cache_path / "experiment_config.json"
        if not config_file.exists():
            print(f"Warning: No config file found in {cache_path}")
            return False
        
        try:
            cached_config = self.load(cache_path)
            # Compare all relevant fields
            return asdict(self) == asdict(cached_config)
        except Exception as e:
            print(f"Warning: Error validating cache: {e}")
            return False
    
    def print_summary(self):
        """Print a human-readable summary of the configuration."""
        print("=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"Config Hash: {self.config_hash}")
        print()
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer_name}")
        print(f"D_model: {self.d_model}")
        print(f"Batch size: {self.batch_size}")
        print()
        print(f"Num species: {self.num_species}")
        print(f"Random seed: {self.random_seed}")
        print(f"Min sequence length: {self.min_sequence_length}")
        print()
        print(f"Region length: {self.region_length} bp")
        if self.num_samples is not None:
            print(f"Samples per genome: {self.num_samples}")
        else:
            print(f"Coverage fraction: {self.coverage_fraction}")
        print(f"Average over last: {self.average_over_last_bp} bp")
        print()
        print(f"Experiment dir: {self.get_cache_path()}")
        print(f"Embeddings dir: {self.get_embeddings_dir()}")
        print("=" * 60)


def create_default_config(**overrides) -> ExperimentConfig:
    defaults = {
        "model": "7b",
        "layer_name": "blocks.24.mlp.l3",
        "batch_size": 8,  # Will be adjusted based on model
        "num_species": 64,
        "random_seed": 42,
        "data_sources": [
            "https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/main/json/midtraining_specific/gtdb_v220_stitched/data_gtdb_train_chunk1.jsonl.gz",
        ],
        "min_sequence_length": 40000,
        "remove_tags": True,
        "region_length": 4000,
        "num_samples": 5,
        "coverage_fraction": None,
        "average_over_last_bp": 2000,
    }
    
    # Apply overrides
    defaults.update(overrides)
    
    # Auto-adjust batch size if not explicitly overridden
    if "batch_size" not in overrides:
        defaults["batch_size"] = 48 if defaults["model"] == "1b" else 8
    
    return ExperimentConfig(**defaults)
