# -*- coding: utf-8 -*-
"""
rgcn_pretrain.py
---------------------------------
• Read triples_unique.csv → Train 2-layer R-GCN (DistMult)
• Simultaneously learn node embedding node_emb and relation embedding rel_emb
• Output node_embeds.pt:
{'node2id','rel2id','node_emb','rel_emb'}
"""

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import json
import re

# ---------- hyperparams ----------
TRIPLE_CSV = "triples_unique.csv"
OUT_FILE   = "node_embeds.pt"
D_GNN  = 256
EPOCHS = 300
LR     = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"Using device: {DEVICE}")

# ---------- load triples ----------
print(f"Loading triples from {TRIPLE_CSV}...")
df = pd.read_csv(TRIPLE_CSV)
# Handle both 3-column and 4-column CSV files
if 'path_id' in df.columns:
    print("Found path_id column, ignoring it for GNN training")
    df = df[['head', 'rel', 'tail']]

heads, rels, tails = df["head"].tolist(), df["rel"].tolist(), df["tail"].tolist()

nodes = sorted(set(heads) | set(tails))
node2id = {n: i for i, n in enumerate(nodes)}

# Analyze node types based on actual data patterns
# Nodes with underscore are mostly variants (4511) plus some special cases
nodes_with_underscore = [n for n in nodes if '_' in n]
nodes_with_colon = [n for n in nodes if ':' in n]  # Special BP nodes
protein_nodes = [n for n in nodes if n.isupper() and '_' not in n and ':' not in n]

# Separate variants from other underscore nodes
# Variants should be protein_mutation format where mutation contains letter+digits+letter
variant_nodes = []
other_underscore_nodes = []

# Known non-variant patterns with underscore
non_variant_patterns = ['_complex', '_family', '_C', '_A', '_protein', '_GTPase']

for n in nodes_with_underscore:
    parts = n.split('_')
    is_variant = False
    
    if len(parts) == 2:
        protein_part, mutation_part = parts
        # Check if it's a known non-variant pattern
        is_non_variant = any(n.endswith(pattern) for pattern in non_variant_patterns)
        
        if not is_non_variant:
            # Check mutation pattern: should have letter, digits, and typically ends with letter
            if (len(mutation_part) >= 3 and 
                mutation_part[0].isalpha() and 
                any(c.isdigit() for c in mutation_part) and
                # Most mutations have position number in the middle
                any(c.isalpha() for c in mutation_part[1:])):
                is_variant = True
    
    if is_variant:
        variant_nodes.append(n)
    else:
        other_underscore_nodes.append(n)

# BP/Disease nodes: everything else (lowercase names, GO/HP/DOID terms, etc.)
bp_nodes = []
for n in nodes:
    if (n not in variant_nodes and 
        n not in protein_nodes and 
        n not in other_underscore_nodes):
        bp_nodes.append(n)

# Add special categories to bp_nodes
bp_nodes.extend(nodes_with_colon)
bp_nodes.extend(other_underscore_nodes)

print(f"\nNode statistics:")
print(f"Total nodes: {len(nodes):,}")
print(f"- Variant nodes: {len(variant_nodes):,}")
print(f"- Protein nodes: {len(protein_nodes):,}")
print(f"- BP/Disease nodes: {len(bp_nodes):,}")

# Verify our classification
if len(variant_nodes) != 4511:
    print(f"Note: Found {len(variant_nodes)} variants (expected 4511)")
    if other_underscore_nodes:
        print(f"Underscore nodes classified as non-variants: {other_underscore_nodes[:10]}")


print(f"\nNode statistics:")
print(f"Total nodes: {len(nodes):,}")
print(f"- Variant nodes: {len(variant_nodes):,}")
print(f"- Protein nodes: {len(protein_nodes):,}")
print(f"- BP/Disease nodes: {len(bp_nodes):,}")

# ensure "Identity" relation is always at index 0
unique_rels = sorted(set(rels))
if "Identity" not in unique_rels:
    unique_rels.insert(0, "Identity")
else:
    unique_rels.remove("Identity")
    unique_rels.insert(0, "Identity")
rel2id = {r: idx for idx, r in enumerate(unique_rels)}

num_nodes, num_rels = len(nodes), len(rel2id)
print(f"nodes={num_nodes:,}  relations={num_rels}  (Identity id=0)")

# ---------- PyG graph ----------
edge_index = torch.tensor([[node2id[h] for h in heads],
                           [node2id[t] for t in tails]], dtype=torch.long)
edge_type  = torch.tensor([rel2id[r] for r in rels], dtype=torch.long)
x_init     = torch.randn(num_nodes, D_GNN)

data = Data(x=x_init, edge_index=edge_index, edge_type=edge_type)
data.num_nodes = num_nodes
data = data.to(DEVICE)

print(f"Total edges: {len(heads):,}")

# ---------- model ----------
class RGCN(torch.nn.Module):
    def __init__(self, dim, num_rels):
        super().__init__()
        self.conv1 = RGCNConv(dim, dim, num_rels)
        self.conv2 = RGCNConv(dim, dim, num_rels)
        self.rel_emb = torch.nn.Parameter(torch.randn(num_rels, dim))

    def forward(self, x, ei, et):
        x = torch.relu(self.conv1(x, ei, et))
        x = torch.relu(self.conv2(x, ei, et))
        return x, self.rel_emb

model = RGCN(D_GNN, num_rels).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------- DistMult loss ----------
def link_loss(node_emb, rel_emb, ei, et, num_samples=2):
    h = node_emb[ei[0]]                # (E,d)
    t = node_emb[ei[1]]
    r = rel_emb[et]                    # (E,d)
    pos = (h * r * t).sum(dim=1)       # DistMult score

    # negative sampling: corrupt tail
    neg_t_idx = torch.randint(0, node_emb.size(0), (ei.size(1)*num_samples,),
                              device=node_emb.device)
    neg_t  = node_emb[neg_t_idx]
    neg_r  = rel_emb[et.repeat_interleave(num_samples)]
    neg_h  = h.repeat_interleave(num_samples, dim=0)
    neg = (neg_h * neg_r * neg_t).sum(dim=1)

    return (torch.nn.functional.softplus(-pos).mean() +
            torch.nn.functional.softplus( neg).mean())

# ---------- training ----------
print(f"\nStarting training for {EPOCHS} epochs...")
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    opt.zero_grad()
    node_emb, rel_emb = model(data.x, data.edge_index, data.edge_type)
    loss = link_loss(node_emb, rel_emb, data.edge_index, data.edge_type)
    loss.backward()
    opt.step()

    if epoch==1 or epoch%10==0:
        print(f"epoch {epoch:03d}/{EPOCHS}  link-loss={loss.item():.4f}")
    
    # Save checkpoint every 50 epochs
    if epoch % 50 == 0:
        checkpoint_path = f"checkpoints/rgcn_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss.item(),
            'node2id': node2id,
            'rel2id': rel2id,
        }, checkpoint_path)

print(f"\nTraining completed!")

# ---------- save ----------
with torch.no_grad():
    final_node_emb = node_emb.detach().cpu()
    final_rel_emb = rel_emb.detach().cpu()

# Prepare comprehensive output
output_data = {
    # Mappings
    "node2id": node2id,
    "rel2id": rel2id,
    "id2node": {v: k for k, v in node2id.items()},
    "id2rel": {v: k for k, v in rel2id.items()},
    
    # Embeddings
    "node_emb": final_node_emb,
    "rel_emb": final_rel_emb,
    
    # Node type information
    "variant_nodes": variant_nodes,
    "protein_nodes": protein_nodes,
    "bp_nodes": bp_nodes,
    
    # Training information
    "training_config": {
        "embedding_dim": D_GNN,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "num_nodes": num_nodes,
        "num_relations": num_rels,
        "num_edges": len(heads),
        "device": str(DEVICE),
    },
    
    # Metadata
    "timestamp": datetime.now().isoformat(),
}

torch.save(output_data, OUT_FILE)

print(f"\nSaved embeddings → {OUT_FILE}")
print("node_emb shape:", node_emb.shape, "  rel_emb shape:", rel_emb.shape)
print("Identity rel id =", rel2id["Identity"])

# Save a human-readable summary
summary_file = OUT_FILE.replace('.pt', '_summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"R-GCN Training Summary\n")
    f.write(f"====================\n\n")
    f.write(f"Training completed at: {datetime.now()}\n")
    f.write(f"Total nodes: {num_nodes:,}\n")
    f.write(f"- Variants: {len(variant_nodes):,}\n")
    f.write(f"- Proteins: {len(protein_nodes):,}\n")
    f.write(f"- BP/Diseases: {len(bp_nodes):,}\n")
    f.write(f"Total edges: {len(heads):,}\n")
    f.write(f"Relations: {num_rels}\n")
    f.write(f"Final loss: {loss.item():.4f}\n")

print(f"Summary saved to {summary_file}")

# Save node statistics JSON
node_stats_file = OUT_FILE.replace('.pt', '_node_stats.json')
node_stats = {
    "total_nodes": num_nodes,
    "variant_nodes": len(variant_nodes),
    "protein_nodes": len(protein_nodes),
    "bp_nodes": len(bp_nodes),
    "total_edges": len(heads),
    "num_relations": num_rels,
    "example_variants": variant_nodes[:10],
    "example_proteins": protein_nodes[:10],
    "example_bps": bp_nodes[:10],
    "relations": list(rel2id.keys())
}
with open(node_stats_file, 'w') as f:
    json.dump(node_stats, f, indent=2)
print(f"Node statistics saved to {node_stats_file}")

print("\n" + "="*50)
print("OUTPUT FILES SUMMARY")
print("="*50)
print(f"Main embedding file: {OUT_FILE}")
print(f" - Contains node/relation embeddings and all mappings")
print(f"Summary file: {summary_file}")
print(f" - Human-readable training summary")
print(f"Node statistics: {node_stats_file}")
print(f" - Detailed graph statistics in JSON")
print(f"Checkpoints: checkpoints/rgcn_epoch_*.pt")
print(f" - Saved every 50 epochs")
print("="*50)

print("\nDone!")