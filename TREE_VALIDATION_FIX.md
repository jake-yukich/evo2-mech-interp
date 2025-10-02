# Tree Validation Fix: Prevent Wasted Resources

## Problem

The original pipeline was dropping genomes late in the process after expensive computation had already been performed:

1. **Data Loading** → Load GTDB sequences
2. **Preprocessing** → Filter by length, extract tags, map to GTDB accessions
3. **Sampling** → Extract multiple genomic regions from each genome ⚠️
4. **Tokenization** → Tokenize all sampled regions ⚠️
5. **Inference** → Run expensive GPU inference to extract embeddings ⚠️ ⚠️ ⚠️
6. **Phylogenetic Distances** → Compute tree distances
   - **HERE**: Some accession IDs not found in tree → NaN values
   - These genomes would need to be dropped from analysis
7. **Analysis** → Correlation plots fail or include NaN values

The issue is that steps 3-5 (marked with ⚠️) are expensive operations, especially the GPU inference. If a genome doesn't have a valid tree entry, all that work was wasted.

## Root Cause

The problem occurred because:

1. **Tag → Accession mapping** is done via GTDB metadata file (`bac120_metadata_r220.tsv`)
2. **Tree validation** uses the actual phylogenetic tree file (`bac120_r220.tree`)
3. **Mismatch**: Not all accession IDs in the metadata are present in the tree as leaf nodes

Example:
```python
# Genome has tag that maps to accession ID
tag = "D__BACTERIA;P__PSEUDOMONADOTA;C__ALPHAPROTEOBACTERIA;..."
accession = "GCA_123456789.1"  # Found in metadata

# But this accession might not be in the tree!
tree = Tree("bac120_r220.tree")
if accession not in tree.get_leaf_names():
    # Distance computation returns NaN
    # All previous work (sampling, inference) was wasted!
```

## Solution

**Validate tree membership EARLY**, before any expensive operations:

### New Workflow

1. **Data Loading** → Load GTDB sequences
2. **Preprocessing** → Filter by length, extract tags, map to GTDB accessions
3. **Tree Validation** → **NEW STEP**: Filter to only genomes in tree ✅
4. **Sampling** → Extract regions (only from valid genomes)
5. **Tokenization** → Tokenize (only valid genomes)
6. **Inference** → GPU inference (only valid genomes)
7. **Phylogenetic Distances** → Compute distances (no NaN values!)
8. **Analysis** → Clean correlation plots

### Implementation

#### New Function: `filter_genomes_in_tree()`

Added to `utils/phylogenetics.py`:

```python
def filter_genomes_in_tree(
    df: pd.DataFrame, 
    tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree",
    accession_column: str = "gtdb_accession"
) -> pd.DataFrame:
    """
    Filter dataframe to only keep genomes whose accession IDs are in the phylogenetic tree.
    This prevents wasting resources on genomes that will be dropped later.
    
    Args:
        df: DataFrame with accession IDs
        tree_file: Path to the GTDB tree file
        accession_column: Name of the column containing accession IDs
    
    Returns:
        Filtered DataFrame with only genomes that are in the tree
    """
    print("Loading tree to validate accession IDs...")
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
```

#### Helper Function: `get_tree_leaf_names()`

```python
def get_tree_leaf_names(tree_file: str = "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree") -> set[str]:
    """
    Returns a set of all leaf names in the GTDB tree.
    This is useful for filtering genomes before processing.
    """
    tree = get_tree(tree_file)
    return set(tree.get_leaf_names())
```

### Updated Pipeline Code

In `notebooks/phylo_tree_prod_refactor.py`:

```python
# PREPROCESSING
df = preprocess_gtdb_sequences(
    df, 
    min_length=config.min_sequence_length, 
    subset=config.num_species, 
    random_seed=config.random_seed,
    remove_tags=config.remove_tags
)
df = add_gtdb_accession(df, get_tag_to_gtdb_accession_map())
df = df.dropna(subset=["gtdb_accession"]).reset_index(drop=True)

# Filter to only keep genomes that are in the phylogenetic tree
# This prevents wasting resources on genomes that will be dropped later
df = filter_genomes_in_tree(df, config.gtdb_tree_path, accession_column="gtdb_accession")

# Save metadata to experiment directory
df.to_csv(experiment_dir / "genomes_metadata.csv", index=False)
```

## Impact

### Before Fix
```
Initial genomes: 64
After preprocessing: 64
After sampling: 64 genomes × 10 samples = 640 regions
After tokenization: 640 tokenized sequences
After inference: 64 genome embeddings (expensive GPU work!)
After distance computation: 58 valid genomes (6 dropped due to missing tree entries)
→ Wasted: 6 genomes × 10 samples × GPU inference time
```

### After Fix
```
Initial genomes: 64
After preprocessing: 64
After tree validation: 58 genomes (6 filtered early)
After sampling: 58 genomes × 10 samples = 580 regions
After tokenization: 580 tokenized sequences
After inference: 58 genome embeddings
After distance computation: 58 valid genomes (0 dropped!)
→ Saved: 6 genomes × 10 samples × GPU inference time
```

### Resource Savings Example

For a typical experiment with:
- 64 genomes
- 10 samples per genome
- 4000 bp regions
- 7B model on GPU

**Wasted computation per invalid genome:**
- Sampling: ~1 second
- Tokenization: ~5 seconds
- **GPU Inference: ~30 seconds** (most expensive!)
- Total: ~36 seconds per invalid genome

**If 10% of genomes are invalid (6 out of 64):**
- Old method: Wastes ~3.6 minutes of GPU time
- New method: Adds ~2 seconds upfront (loading tree once)
- **Net savings: ~3.5 minutes per experiment run**

For larger experiments (e.g., 1000 genomes):
- Could waste **hours** of GPU time on invalid genomes
- Tree validation takes only a few seconds

## Testing

To verify the fix works:

```python
# Check how many genomes would be filtered
from utils.phylogenetics import get_tree_leaf_names, filter_genomes_in_tree

# Load your dataframe after adding accessions
tree_leaves = get_tree_leaf_names()
print(f"Tree has {len(tree_leaves)} leaves")

# See what would be filtered
invalid_accessions = df[~df["gtdb_accession"].isin(tree_leaves)]
print(f"Would filter out {len(invalid_accessions)} genomes")
print(f"Invalid accessions: {invalid_accessions['gtdb_accession'].tolist()}")

# Apply filter
df_filtered = filter_genomes_in_tree(df)
```

## Migration Guide

### If using `phylo_tree_prod.py` (old version)

Add this line after `add_gtdb_accession()` and `dropna()`:

```python
df = add_gtdb_accession(df)
df = df.dropna(subset=["gtdb_accession"]).reset_index(drop=True)

# ADD THIS LINE
from utils.phylogenetics import filter_genomes_in_tree
df = filter_genomes_in_tree(df, "/root/evo2-mech-interp/data/gtdb/bac120_r220.tree")
```

### If using `phylo_tree_prod_refactor.py` (refactored version)

Already implemented! Just update your imports:

```python
from utils.phylogenetics import get_tag_to_gtdb_accession_map, filter_genomes_in_tree
```

And the filter is automatically applied in the preprocessing section.

## Related Documentation

- See `.github/copilot-instructions.md` under "Tree Validation" for best practices
- Function documentation in `utils/phylogenetics.py`
- Configuration includes `gtdb_tree_path` in `utils/config.py`

## Future Improvements

1. **Cache tree leaves**: Load once and reuse across experiments
2. **Parallel tree loading**: If dealing with very large trees
3. **Provide diagnostics**: Report which specific accessions are missing and why
4. **Upstream fix**: Investigate why metadata contains accessions not in tree
