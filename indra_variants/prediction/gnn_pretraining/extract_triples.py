
# -*- coding: utf-8 -*-
"""
extract_triples.py

Extract triples from variant pathway data and prepare for GNN pretraining.
"""

import pandas as pd
import re
import json
import csv
from collections import defaultdict

IN_FILE    = "../seq_embedding/variant_with_esm2.tsv"
TRIPLE_OUT = "triples.csv"
TRIPLE_UNIQUE_OUT = "triples_unique.csv"
PATH_OUT   = "variant_paths.tsv"

PAT = re.compile(r"(.*?)\s-\[([^\]]+?)]->\s(.*)")

def split_chain(chain):
    """return triples[], nodes[], rels[]"""
    triples, nodes, rels = [], [], []
    cur = chain.strip()
    while True:
        m = PAT.match(cur)
        if not m: break
        h, r, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        nxt = PAT.match(rest)
        t = nxt.group(1).strip() if nxt else rest
        triples.append((h, r, t))
        rels.append(r)
        if not nodes: nodes.append(h)
        nodes.append(t)
        if not nxt: break
        cur = rest
    return triples, nodes, rels

def clean_bp_name(bp):
    """clean biological process/disease name for path_id"""
    if pd.isna(bp):
        return "unknown"
    return bp.replace(" ", "_").replace("/", "-").replace(":", "").replace(",", "")[:50]

df = pd.read_csv(IN_FILE, sep="\t",
                 usecols=["variant_protein", "variant_info", "chain","biological_process/disease"],)

all_triples = []
path_records = []
path_counters = defaultdict(int)
seen_identity = set()

for _, row in df.iterrows():
    if pd.isna(row["chain"]): continue

    triples, nodes, rels = split_chain(row["chain"])
    
    # get variant_id
    variant_id = f'{row["variant_protein"]}_{row["variant_info"]}'
    protein_name = row["variant_protein"]
    bp_name = clean_bp_name(row["biological_process/disease"])
    
    # get path_id
    path_key = f"{variant_id}::{bp_name}"
    path_counters[path_key] += 1
    path_id = f"{path_key}::{path_counters[path_key]:03d}"
    
    h0 = nodes[0]
    
    for h, r, t in triples:
        triple_with_id = (h, r, t, path_id)
        all_triples.append(triple_with_id)
    
    identity_key = (variant_id, protein_name)
    identity_path_id = f"{variant_id}::var_identity"
    
    # identity relation
    identity_triple = (variant_id, "Identity", h0, identity_path_id)
    all_triples.append(identity_triple)

    if nodes[0] != variant_id:
        nodes.insert(0, variant_id)
        rels.insert(0, "Identity")
    
    path_records.append({
        "variant_protein": row.get("variant_protein", ""),
        "variant_id": variant_id,
        "nodes_json": json.dumps(nodes, ensure_ascii=False),
        "rels_json" : json.dumps(rels , ensure_ascii=False),
        "biological_process/disease": row.get("biological_process/disease",""),
        "path_id": path_id
    })

# output1: all triples with path_id
with open(TRIPLE_OUT, "w", newline="") as fw:
    cw = csv.writer(fw)
    cw.writerow(["head", "rel", "tail", "path_id"])
    all_triples_sorted = sorted(all_triples, key=lambda x: (x[0], x[1], x[2], x[3]))
    cw.writerows(all_triples_sorted)
print(f"Save to {TRIPLE_OUT}, rows: {len(all_triples)}")

# output2: unique triples for GNN
# only keep unique triples for GNN, deduplicate Identity edges
unique_triples_for_gnn = []
seen_identity_for_gnn = set()

for h, r, t, path_id in all_triples:
    if r == "Identity":
        # Identity edges are special, deduplicate them
        identity_key = (h, r, t)
        if identity_key not in seen_identity_for_gnn:
            # for identity edges, we use a special path_id
            unique_triples_for_gnn.append((h, r, t, f"{h}::var_identity"))
            seen_identity_for_gnn.add(identity_key)
    else:
        unique_triples_for_gnn.append((h, r, t, path_id))

with open(TRIPLE_UNIQUE_OUT, "w", newline="") as fw:
    cw = csv.writer(fw)
    cw.writerow(["head", "rel", "tail", "path_id"])
    cw.writerows(sorted(unique_triples_for_gnn, key=lambda x: (x[0], x[1], x[2], x[3])))
print(f"Save to {TRIPLE_UNIQUE_OUT}, triples for GNN: {len(unique_triples_for_gnn)} (Identity deduped, others kept)")

# output3: paths with nodes_json
pd.DataFrame(path_records).to_csv(PATH_OUT, sep="\t", index=False)
print(f"Save to {PATH_OUT}, paths: {len(path_records)}")

print("\n=== Statistics ===")
print(f"Total variants: {len(set(pr['variant_id'] for pr in path_records))}")
print(f"Total biological processes: {len(set(pr['biological_process/disease'] for pr in path_records if not pd.isna(pr['biological_process/disease'])))}")
print(f"Total paths: {len(path_records)}")
print(f"Total triples (with duplicates): {len(all_triples)}")
print(f"Triples for GNN: {len(unique_triples_for_gnn)}")
print(f"Identity relations (unique): {len(seen_identity_for_gnn)}")


print("\n=== Sample path IDs ===")
for i, (k, v) in enumerate(path_counters.items()):
    if i >= 5: break
    print(f"{k}: {v} paths")