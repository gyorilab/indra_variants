# Automated variant network assembly and prediction framework

This repository implements:
- An approach for deriving variant-specific networks from the INDRA Database,
  connecting genetic variants to their downstream biological and disease processes
  potentially via molecular intermediaries. (`indra_variant.generate` module)
- A web application that allows browsing and interacting with these networks,
  deployed at https://variants.indra.bio. (`indra_variant.app` module)
- An integrative, transformer-based prediction framework that makes use of the variant-effect
  networks as well as protein language model embeddings and other features
  to predict variant-effect mechanisms. (`indra_variant.predict` module)

## Data Resource:

- **training_feature_input.tsv** - input features for training
- **genes_to_pmids.tsv** - mapping of proteins to relevant publications
- **label_classified.tsv** - mapping of 1085 bp/disease labels to 30 categories
- **human_domains.tsv** - domain info from uniprot_sprot.dat.gz (download: May 13, 2025)
- **clinvar_patho_subset.tsv.gz** - extract the fields from the ClinVar: clinvar_variant_summary.txt (.gz)

## Start Guide:
### 1. Generate Protein Embeddings
```bash
cd predict/seq_embedding

# input: data/training_feature_input.tsv
# output: training_feature_esm2.tsv
python embedding.py 

```

### 2. Prepare Knowledge Graph Paths
```bash
cd predict/gnn_pretraining

# Extract triples (subject, rel, object) from assembly causal paths
# input: data/training_feature_esm2.tsv
# output: triples.csv; triples_unique.csv; variant_paths.tsv
python extract_triples.py 


# Generate node embeddings using GNN
# input: triples_unique.csv
# output: node_embeds.pt
python rgcn_pretrain.py 

```

### 3. Train
```bash
cd predict/train

# Split dataset randomly
# input: data/label_clssified.tsv; variant_paths.tsv
# output: splits/(splits_index)
python dataset_split.py

# Build training dataset
# input: training_feature_esm2.tsv; variant_paths.tsv; data/label_classified.tsv; node_embeds.pt
# output: path_dataset_bag_full.pt; W_var.pt
python build_train_dataset.py

# Train the model
# input: path_dataset_bag_full.pt
python training.py

# Validate model
python prediction.py
```
