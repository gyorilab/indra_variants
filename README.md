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

## Requirements
- python >= 3.9

## Data availability:
Place the following files under `data/`:
- `training_feature_input.tsv` - input features for training
- `genes_to_pmids.tsv` - mapping of proteins to relevant publications
- `label_classified.tsv` - mapping of 1085 bp/disease labels to 30 categories
- `human_domains.tsv` - domain info from UniProt `uniprot_sprot.dat.gz` (downloaded May 13, 2025)
- `clinvar_patho_subset.tsv.gz` - extract the fields from the ClinVar `clinvar_variant_summary.txt (.gz)`

## Usage
### Generate protein embeddings
Use `predict/seq_embedding/embedding.py` to convert protein features into ESM-2 embeddings for downstream tasks.
- input: `data/training_feature_input.tsv`
- output: `training_feature_esm2.tsv`


### Extract knowledge-graph paths
Use `predict/gnn_pretraining/extract_triples.py` to extract triples (subject, rel, object) from literature-derived causal paths.
- input: `data/training_feature_esm2.tsv`
- output: `triples.csv`; `triples_unique.csv`; `variant_paths.tsv`

### Generate node embeddings using GNN
Use `predict/gnn_pretraining/rgcn_pretrain.py` to learn structural representations of nodes in the knowledge graph.
- input: `triples_unique.csv`
- output: `node_embeds.pt` (graph node embedding weights)


### Split dataset
Use `predict/train/dataset_split.py` to split the dataset randomly for model training and evaluation.
- input: `data/label_clssified.tsv`; `variant_paths.tsv`
- output: `splits/(splits_index)`

### Build training dataset
Use `predict/train/build_train_dataset.py` to assemble complete training dataset by combining embeddings, paths, and label mappings.
- input: `training_feature_esm2.tsv`; `variant_paths.tsv`; `data/label_classified.tsv`; `node_embeds.pt`
- output: `path_dataset_bag_full.pt`; `W_var.pt`

### Train model
Use `predict/train/training.py`.
- input: `path_dataset_bag_full.pt`

Run the script with the -h argument to see additional arguments.

### Predict
Use `predict/train/prediction.py`.

Run the script with the -h argument to see additional arguments.

