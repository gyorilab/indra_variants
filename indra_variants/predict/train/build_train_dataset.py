# -*- coding: utf-8 -*-
"""
build_train_dataset.py

Build complete dataset with features and labels (without splitting)
Splitting will be handled externally by CSV files
"""

import json, os, random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ─────────────────────────── file paths ────────────────────────────────
X_FILE = "../seq_embedding/training_feature_esm2.tsv"
PATH_FILE = "../gnn_pretraining/variant_paths.tsv"
EMB_FILE = "../gnn_pretraining/node_embeds.pt"
LABEL_CLASSIFIED_FILE = "../seq_embedding/label_classified.tsv"
OUT_FILE = "path_dataset_bag_full.pt"
W_VAR_PT = "W_var.pt"

# ───────────────────── hyper-parameters / dims ─────────────────────────
D_MISC, D_AA, D_POS = 9, 40, 1
D_GNN, D_TOK = 256, 512

# ───────────────────────── reproducibility ─────────────────────────────
random.seed(42);  np.random.seed(42);  torch.manual_seed(42)

# ══════════════════ load tables ═══════════════════════════════════
print("Loading tables …")
x_df = pd.read_csv(X_FILE, sep="\t")
x_df["variant_id"] = x_df["variant_protein"] + "_" + x_df["variant_info"]
x_df["seq_len"]    = x_df["sequence"].str.len()
ESM_COLS = [c for c in x_df.columns if c.startswith("esm2_")]
D_VAR    = len(ESM_COLS) + D_MISC + D_AA + D_POS
x_df = (x_df.drop_duplicates("variant_id", keep="first")
             .set_index("variant_id"))

paths_df = pd.read_csv(PATH_FILE, sep="\t")
BP_COL   = "biological_process/disease"
if "path_score" not in paths_df.columns:
    paths_df["path_score"] = 1.0

# load label classification file
label_class_df = pd.read_csv(LABEL_CLASSIFIED_FILE, sep="\t")
label_to_category = dict(zip(label_class_df["term"], label_class_df["primary_category"]))

# Create category mapping
all_categories = sorted(label_class_df["primary_category"].unique())
category_to_id = {cat: i for i, cat in enumerate(all_categories)}
n_categories = len(all_categories)

print(f"variants in sequence table : {len(x_df):,}")
print(f"rows in path table        : {len(paths_df):,}")
print(f"labels in classification  : {len(label_to_category):,}")
print(f"number of categories      : {n_categories}")

# ══════════════════ GNN embeddings ════════════════════════════════
print("Loading GNN embeddings...")
gnn = torch.load(EMB_FILE)
node_emb = gnn.get("node_emb", gnn.get("emb"))
rel_emb  = gnn.get("rel_emb", node_emb)
node2id  = gnn["node2id"];  rel2id = gnn["rel2id"]

# ══════════════════ projection matrices ═══════════════════════════
if os.path.exists(W_VAR_PT):
    W_var = torch.load(W_VAR_PT);  print("Loaded W_var")
else:
    W_var = torch.randn(D_VAR, D_TOK)/np.sqrt(D_VAR)
    torch.save(W_var, W_VAR_PT);   print("Initialised W_var randomly")
W_gnn = torch.randn(D_GNN, D_TOK)/np.sqrt(D_GNN)
W_rel = torch.randn(D_GNN, D_TOK)/np.sqrt(D_GNN)
project = lambda v, W: torch.as_tensor(v, dtype=torch.float32) @ W

# ══════════════════ helpers ═══════════════════════════════════════
AA2IDX = {a:i for i,a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
def aa_onehot(a:str):
    vec = torch.zeros(20)
    if a and a[0] in AA2IDX: vec[AA2IDX[a[0]]] = 1.
    return vec
MISC = ["patho_score","star_score",
        "dom_DOMAIN","dom_REPEAT","dom_ZN_FING",
        "dom_COMPBIAS","dom_REGION","dom_COILED","dom_MOTIF"]

def build_variant_vec(row):
    esm  = torch.tensor(row[ESM_COLS].to_numpy(np.float32))
    misc = torch.tensor(row[MISC].astype("float32").fillna(0).to_numpy())
    ref  = aa_onehot(str(row.get("ref_aa","")))
    alt  = aa_onehot(str(row.get("alt_aa","")))
    pos  = torch.tensor([row["mutation_pos"]/max(row["seq_len"],1)], dtype=torch.float32)
    return project(torch.cat([esm,misc,ref,alt,pos]), W_var)

def node_tok(n): return (project(node_emb[node2id[n]], W_gnn)
                         if n in node2id else torch.zeros(D_TOK))
def rel_tok(r):  return (project(rel_emb[rel2id[r]],  W_rel)
                         if r in rel2id  else torch.zeros(D_TOK))

# ══════════════════ build dataset with category info ══════════════
label2id = {bp:i for i,bp in enumerate(sorted(paths_df[BP_COL].unique()))}
n_cls = len(label2id)

variant_vecs, paths_tok, paths_mask = [], [], []
label_vecs, category_vecs, n_paths, n_bps = [], [], [], []
variant_ids_list = []  # 重要：保存variant_id的顺序
miss_node, miss_rel, miss_var = set(), set(), set()

print("Building variant-level bags with category info …")
for vid, g in tqdm(paths_df.groupby("variant_id")):
    if vid not in x_df.index:
        miss_var.add(vid);  continue

    # 重要：记录variant_id的顺序
    variant_ids_list.append(vid)

    # 1) variant vector
    v_vec = build_variant_vec(x_df.loc[vid])
    variant_vecs.append(v_vec)

    # 2) full BP multi-hot
    y = torch.zeros(n_cls)
    y_cat = torch.zeros(n_categories)  # Category-level multi-hot
    
    for bp in g[BP_COL].unique():
        y[label2id[bp]] = 1.
        if bp in label_to_category:
            cat = label_to_category[bp]
            y_cat[category_to_id[cat]] = 1.
    
    label_vecs.append(y)
    category_vecs.append(y_cat)
    n_bps.append(int(y.sum()))

    # 3) all paths
    pv_tok, pv_msk = [], []
    for _, row in g.sort_values("path_score", ascending=False).iterrows():
        try:
            nodes = json.loads(row["nodes_json"])
            rels  = json.loads(row["rels_json"])
            
            # Track missing nodes/relations
            for n in nodes[1:]:
                if n not in node2id:
                    miss_node.add(n)
            for r in rels:
                if r not in rel2id:
                    miss_rel.add(r)
            
            toks = [v_vec] + \
                   sum([[rel_tok(r), node_tok(n)] for r, n in zip(rels, nodes[1:])], [])
            seq = torch.stack(toks)
            msk = torch.ones(len(toks))
            pv_tok.append(seq);  pv_msk.append(msk)
        except (json.JSONDecodeError, KeyError) as e:
            # Skip problematic paths
            continue
            
    paths_tok.append(pv_tok)
    paths_mask.append(pv_msk)
    n_paths.append(len(pv_tok))

# ═════════════════ Dataset statistics ════════════════════════
print("\n=== Dataset statistics ===")
print(f"Total variants kept : {len(variant_vecs):,}")
print(f"Missing variants    : {len(miss_var)}")
print(f"Number of categories: {n_categories}")
print(f"Number of labels    : {n_cls}")
print(f"Missing nodes       : {len(miss_node)}")
print(f"Missing relations   : {len(miss_rel)}")

# Calculate basic statistics
avg_labels_per_variant = torch.stack(label_vecs).sum(1).float().mean()
avg_categories_per_variant = torch.stack(category_vecs).sum(1).float().mean()
avg_paths_per_variant = torch.tensor(n_paths).float().mean()

print(f"Avg labels per variant    : {avg_labels_per_variant:.2f}")
print(f"Avg categories per variant: {avg_categories_per_variant:.2f}")
print(f"Avg paths per variant     : {avg_paths_per_variant:.1f}")

# ═════════════════ save complete dataset ════════════════════════
save_obj = {
    # Core data
    "variant_vec": torch.stack(variant_vecs),
    "paths_tok": paths_tok,
    "paths_mask": paths_mask,
    "n_paths": torch.tensor(n_paths),
    "n_bps": torch.tensor(n_bps),
    "y": torch.stack(label_vecs),
    "y_category": torch.stack(category_vecs),
    
    # Critical: variant IDs for split matching
    "variant_ids": variant_ids_list,
    
    # Mapping dictionaries
    "label2id": label2id,
    "category2id": category_to_id,
    "id2category": {v: k for k, v in category_to_id.items()},
    "label_to_category": label_to_category,
    "rel2id": rel2id,
    
    # Metadata
    "n_categories": n_categories,
    "n_cls": n_cls,
    
    # Statistics
    "missing_info": {
        "missing_nodes": list(miss_node)[:100],
        "missing_rels": list(miss_rel)[:100],
        "n_missing_nodes": len(miss_node),
        "n_missing_rels": len(miss_rel),
        "n_missing_variants": len(miss_var)
    },
    
    # Basic statistics
    "statistics": {
        "avg_labels_per_variant": float(avg_labels_per_variant),
        "avg_categories_per_variant": float(avg_categories_per_variant),
        "avg_paths_per_variant": float(avg_paths_per_variant),
        "total_variants": len(variant_vecs)
    }
}

torch.save(save_obj, OUT_FILE)
print(f"\nSaved ➜ {OUT_FILE}   variants = {len(variant_vecs):,}")
print("Done. Dataset ready for external splitting.")