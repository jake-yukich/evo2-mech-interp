# (Re)Finding A Phylogenetic Manifold In Evo2

As part of [ARENA 6.0](https://www.arena.education/)'s Capstone Week, we aimed to reproduce recent work by [Goodfire](https://www.goodfire.ai/) on [Finding the Tree of Life in Evo 2](https://www.goodfire.ai/research/phylogeny-manifold).

[Evo 2](https://arcinstitute.org/tools/evo) is an autoregressive genomic foundation model built by [Arc Institute](https://arcinstitute.org/), trained using over 9 trillion nucleotides across both eukaryotic and prokaryotic organisms to predict next tokens for DNA (and RNA) nucleotide sequences. 

Goodfire's experiments with Evo 2 consisted of:
- Finding a phylogenetic manifold within the embeddings of Evo 2
- Finding a flat representation of this manifold (a subspace where the phylogenetic distance is fully described by the dimensions of the subspace)
- Understanding how genomic "styles" are captured within the embeddings

The motivation for why doing interpretability work on scientific models is well motivated in the original blogpost, and the aim of our work was to:

1. Reproduce Goodfire's results in locating the phylogenetic manifold
2. Find another manifold within the embedding space with some semantic meaning

Part 1 of this blog posts presents our results in reconstructing the phylogenetic manifold and compare them to those in the original blog post, as well as our extensions to their work. 
Part 2 consists of "Field Notes" with a more detailed explanation of our journey of recreating this work went, where we detail the observations, obstacles and decisions we learned along the way in the hope that it both stimulates further discussion around interpretability investigations for scientific models and that it helps others that may wish to use Goodfire's work get going quickly. We highlight where we may have made different decisions given different time and compute constraints (5 days and 1 L40S GPU). Our code with detailed instructions on how to run it are available on GitHub.

## Part 1: Results

## Part 2: Field Notes

### Data Curation
Generally, we found the [Pareto Principle](https://en.wikipedia.org/wiki/Pareto_principle) for data science to hold here, in that we spent approximately 80% of our time wrangling data and 20% running experiments and analysing results. Goodfire state that they *"obtain the phylogenetic trees for 2400+ bacterial species from the Genome Taxonomy Database (GTDB)"* but didn't provide any additional information about how they selected species to look at from GTDB or constructed the phylogenetic tree (we realize this may be obvious to others who are familiar with GTDB as a data source!).
#### Dataset(s)
We suspected that attempting to evenly sample phylogenetic space may be a good idea, but in an attempt to get some small results quickly, we decided to use preprocessed data from the OpenGenome2 dataset used to train the model, from [Arc Institute on Huggingface](https://huggingface.co/datasets/arcinstitute/opengenome2). Evo 2 was trained in two phases: pretraining, where tokenized sequences 8192 in length were fed to the model, and midtraining where context length was extended to 1M input tokens. In midtraining, phylogenetic tags were included in the nucleotide sequences, every 131 kb for the GTDB data.

After exploring both the pretraining and midtraining datasets, we found that pretraining data contained accession IDs alongside the sequences, while the midtraining data only contained sequences, however it did include the phylogenetic tags embedded within the sequences. Given this, and that the pretraining sequences were shorted than the midtraining ones, we elected to use the midtraining dataset.

We explored using subsets of the dataset, ranging from ~60 species to ~1000 from one chunk of the dataset, before running the full analysis on a subset of 2400 across several chunks.

[MORE TIME] 
- It might have been nice to curate a separate dataset outside of the training data, however checking for contamination with the entire training dataset would be computationally intensive, and for this kind of analysis we're not testing for generalization ability, just for semantic structure within the embeddings.

[RED HERRINGS]
- Accession IDs stored with the midtraining data were NCBI, so would need translating to GTDB finding distances within the GTDB tree
### Data Cleaning
We loaded the full genomes from hugging face and performed the following data cleaning steps to extract all the information we needed to understand the phylogenetic relationships between different sequences:

1. Apply a minimum sequence length threshold to ensure that we can sample appropriately
2. Randomly shuffle the data
3. Extract the full phylogenetic tag, removing any repetitions of the tag within the sequence
4. Extract class, order and family fields

We then used the GTDB metadata file to map the phylogenetic tags to accession IDs for finding distances between species within the GTDB tree.

[RED HERRINGS]
- Querying the GTDB over the API (slow)
### Sampling
The original blog post explains how phylogenetic relationships are traditionally computed by comparing similar sequences, and that passing highly similar sequences through the model will inevitably result in high similar embeddings. To avoid this confounding factor, they implement a sampling process across the genome of different species, extracting some number (N) samples across the genome of 4000 bp, pass each sequence through the model and average the embeddings. They take the average over the the last 2000 bp of each genomic region, so that the model has enough context to build up rich representations (due to the autoregressive nature of the model, it only has prior context).
### Calculating Embeddings
#### 1B vs 7B model

### Visualizing the Embedding Space
### Comparing Phylogenetic Distance to Manifold Distance

### Open Questions





Glossary
- Phylogeny
- Manifold
- Codon