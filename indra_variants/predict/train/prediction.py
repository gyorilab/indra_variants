#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction.py

Hierarchical prediction of biological processes and diseases from variant features and paths.
"""

from __future__ import annotations
import argparse, json, re, warnings, math, os
from collections import defaultdict
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# ---------- Constants ----------
HERE = os.path.basename(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.pardir, os.pardir, 'data')
AA3 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
RE_ONE = re.compile(r"([A-Z])(\d+)([A-Z\*])")
RE_THREE = re.compile(r"[p\.]?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*)")
AA2IDX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}


# ---------- Utility Functions ----------
def parse_mut(mut: str) -> Tuple[str, int, str]:
    """Return (ref, pos, alt) from protein HGVS (1- or 3-letter)."""
    mut = mut[2:] if mut.lower().startswith("p.") else mut
    if (m := RE_ONE.fullmatch(mut)):
        return m.group(1), int(m.group(2)), m.group(3)
    if (m := RE_THREE.fullmatch(mut)):
        ref = AA3.get(m.group(1).upper(), "")
        alt = AA3.get(m.group(3).upper(), "")
        if ref and alt:
            return ref, int(m.group(2)), alt
    raise ValueError(f"Unparsable mutation string: {mut}")


def aa_onehot(a: str) -> torch.Tensor:
    """Convert amino acid to one-hot encoding."""
    v = torch.zeros(20)
    if a and a[0] in AA2IDX:
        v[AA2IDX[a[0]]] = 1.
    return v


# ---------- Model Components ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PathEncoder(nn.Module):
    def __init__(self, d=512, heads=8, layers=2, drop=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d, dropout=drop)
        
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4*d, dropout=drop, 
            batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, tok, mask):
        h = self.pos_enc(tok)
        padding_mask = ~mask
        h = self.enc(h, src_key_padding_mask=padding_mask)
        return self.norm(h[:, 0, :])


class HierarchicalBPModel(nn.Module):
    """Predict 30 categories and 1085 labels with hierarchical structure"""
    def __init__(self, n_cls, n_categories, category_to_labels, d=512, heads=8, layers=2, drop=0.1):
        super().__init__()
        self.d = d
        self.n_categories = n_categories
        self.n_cls = n_cls
        self.category_to_labels = category_to_labels
        
        # Path encoder
        self.path_enc = PathEncoder(d, heads, layers, drop)
        
        self.variant_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        
        # Path aggregator
        self.aggregator = nn.MultiheadAttention(
            d, heads, batch_first=True, dropout=drop
        )
        self.agg_norm = nn.LayerNorm(d)
        
        # Category prediction head
        self.category_head = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d, n_categories)
        )
        
        # Category embeddings for feature enhancement
        self.category_embeddings = nn.Parameter(torch.randn(n_categories, d))
        nn.init.xavier_uniform_(self.category_embeddings)
        
        # Label prediction head
        self.label_head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(d // 2, n_cls)
        )
        
        # Category gate for feature enhancement
        self.category_gate = nn.Sequential(
            nn.Linear(n_categories, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.Sigmoid()
        )

    def forward(self, batch_data):
        variants = batch_data['variants']
        paths = batch_data['paths']
        masks = batch_data['masks']
        path_to_sample = batch_data['path_to_sample']
        
        B = variants.shape[0]
        device = variants.device
        
        # Path embeddings
        if paths.shape[0] > 0:
            path_embeddings = self.path_enc(paths, masks)
        else:
            path_embeddings = torch.zeros(0, self.d, device=device)

        v_enhanced = self.variant_proj(variants)
        
        # Aggregate paths for each sample
        aggregated = []
        for b in range(B):
            sample_mask = (path_to_sample == b)
            sample_paths = path_embeddings[sample_mask]
            
            if sample_paths.shape[0] > 0:
                query = v_enhanced[b:b+1].unsqueeze(1)
                keys = values = sample_paths.unsqueeze(0)
                agg, _ = self.aggregator(query, keys, values)
                agg = self.agg_norm(agg.squeeze(0) + v_enhanced[b:b+1])
            else:
                agg = v_enhanced[b:b+1]
            
            aggregated.append(agg)
        
        features = torch.cat(aggregated, dim=0)
        
        # Category prediction
        category_logits = self.category_head(features)
        category_probs = torch.sigmoid(category_logits)
        
        # Category-enhanced features
        weighted_cat_emb = torch.matmul(category_probs, self.category_embeddings)
        gate = self.category_gate(category_probs)
        category_enhanced_features = features + gate * weighted_cat_emb
        
        # Label prediction
        combined_features = torch.cat([features, category_enhanced_features], dim=-1)
        label_logits = self.label_head(combined_features)
        
        return {
            'label_logits': label_logits,
            'category_logits': category_logits,
            'features': features
        }


# ---------- Predictor Class ----------
class HierarchicalPredictor:
    def __init__(
        self,
        ckpt_path: str,
        esm_tsv_path: str,
        device: str = "cpu",
        *,
        domains: str | None = None,
        clinvar: str | None = None,
        w_var_path: str | None = None,
        label_classified_path: str = "label_classified.tsv"
    ):
        self.device = device
        
        # Load model
        self._load_model(ckpt_path)
        
        # Load data resources
        self._load_esm_embeddings(esm_tsv_path)
        self._load_projection_matrix(w_var_path)
        self.domains = self._load_domains(domains)
        self.clinvar = self._load_clinvar(clinvar)
        self.label_to_category = self._load_label_category_mapping(label_classified_path)
        
        self._print_model_info()

    def _load_model(self, ckpt_path: str):
        """Load model from checkpoint."""
        print(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # Extract model config
        if "model_config" in ckpt:
            cfg = ckpt["model_config"]
        else:
            cfg = {
                "n_cls": ckpt.get("n_cls", 1085),
                "n_categories": ckpt.get("n_categories", 30),
                "category_to_labels": ckpt.get("category_to_labels", {}),
                "heads": ckpt.get("args", {}).get("heads", 8),
                "layers": ckpt.get("args", {}).get("layers", 2),
                "dropout": ckpt.get("args", {}).get("dropout", 0.2)
            }
        
        # Initialize model
        self.has_categories = "n_categories" in cfg and cfg.get("n_categories", 0) > 0
        
        if self.has_categories:
            print("Loading hierarchical model with categories...")
            self.model = HierarchicalBPModel(
                n_cls=cfg["n_cls"],
                n_categories=cfg.get("n_categories", 30),
                category_to_labels=cfg.get("category_to_labels", {}),
                heads=cfg.get("heads", 8),
                layers=cfg.get("layers", 2),
                drop=cfg.get("dropout", 0.2)
            ).to(self.device)
        else:
            raise NotImplementedError("Old flat model not implemented in this version")
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        
        if missing_keys:
            print(f"WARNING: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING: Unexpected keys: {unexpected_keys}")
        
        self.model.eval()
        
        # Store model parameters
        self.best_threshold = ckpt.get("test_results", {}).get("threshold", 0.5)
        self.n_categories = cfg.get("n_categories", 30)
        self.category_to_labels = cfg.get("category_to_labels", {})
        self.label2id = cfg.get("label2id", {})
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.category2id = cfg.get("category2id", {})
        self.id2category = {v: k for k, v in self.category2id.items()}

    def _load_esm_embeddings(self, esm_tsv_path: str):
        """Load ESM embeddings from TSV file."""
        print(f"Loading ESM embeddings from {esm_tsv_path}")
        self.esm_df = pd.read_csv(esm_tsv_path, sep="\t")
        self.esm_cols = [c for c in self.esm_df.columns if c.startswith("esm2_")]
        print(f"Found {len(self.esm_cols)} ESM columns")
        
        # Create protein -> ESM mapping
        self.protein2esm = {}
        for protein_name, group in self.esm_df.groupby("variant_protein"):
            esm_features = group.iloc[0][self.esm_cols].values.astype(np.float32)
            seq_len = len(group.iloc[0]["sequence"]) if "sequence" in group.columns else 1000
            self.protein2esm[protein_name] = (esm_features, int(seq_len))
        print(f"Loaded ESM embeddings for {len(self.protein2esm)} proteins")

    def _load_projection_matrix(self, w_var_path: str | None):
        """Load W_var projection matrix."""
        self.W_var = None
        
        # Try different paths
        paths_to_try = [
            w_var_path,
            "W_var.pt",
            "path_dataset_bag_full.pt"
        ]
        
        for path in paths_to_try:
            if path and os.path.exists(path):
                try:
                    print(f"Loading W_var from {path}")
                    w_var_data = torch.load(path, map_location=self.device)
                    
                    if path == "path_dataset_bag_full.pt":
                        self.W_var = w_var_data.get("W_var")
                        if self.W_var is not None:
                            torch.save(self.W_var, "W_var.pt")
                            print("Saved W_var to W_var.pt for future use")
                    else:
                        self.W_var = w_var_data if isinstance(w_var_data, torch.Tensor) else w_var_data.get("W_var")
                    
                    if self.W_var is not None:
                        break
                except Exception as e:
                    print(f"Failed to load W_var from {path}: {e}")
                    continue
        
        if self.W_var is None:
            print("ERROR: Could not find W_var matrix. Using random initialization (results will be poor)")
            self.W_var = torch.randn(2611, 512) / np.sqrt(2611)
        
        self.W_var = self.W_var.to(self.device)
        print(f"W_var shape: {self.W_var.shape}")

    def _load_domains(self, path: str | None) -> dict:
        """Load domain annotations."""
        if not path or not os.path.exists(path):
            return {}
        
        print(f"Loading domains from {path}")
        df = pd.read_csv(path, sep="\t", dtype={"GeneName": str})
        out = defaultdict(list)
        for _, row in df.iterrows():
            gene = row["GeneName"]
            feature = row["FeatureType"]
            start = int(row["Start"])
            end = int(row["End"])
            out[gene].append((feature, start, end))
        return dict(out)

    def _load_clinvar(self, path: str | None) -> dict:
        """Load ClinVar annotations."""
        if not path or not os.path.exists(path):
            return {}
        
        print(f"Loading ClinVar from {path}")
        comp = "gzip" if str(path).endswith(".gz") else None
        df = pd.read_csv(path, sep="\t", compression=comp, low_memory=False)
        
        # Find protein change column
        pc_col = None
        for col in ["ProteinChange", "Protein Change", "Name"]:
            if col in df.columns:
                pc_col = col
                break
        
        if pc_col is None:
            print("Warning: ClinVar file missing protein change column")
            return {}
        
        cmap = {}
        for _, row in df.iterrows():
            gene = row.get("GeneSymbol", row.get("Gene", ""))
            pc = row.get(pc_col, "")
            cs = row.get("ClinicalSignificance", "")
            rs = row.get("ReviewStatus", "")
            
            if pd.isna(pc) or pd.isna(gene):
                continue
            
            # Extract protein HGVS
            prot_hgvs = None
            if isinstance(pc, str):
                if pc.startswith("p."):
                    prot_hgvs = pc[2:]
                else:
                    m = re.search(r"\(p\.([A-Za-z0-9*]+)\)", pc)
                    if m:
                        prot_hgvs = m.group(1)
            
            if not prot_hgvs:
                continue
            
            try:
                ref, pos, alt = parse_mut(prot_hgvs)
                patho = self._map_patho(cs)
                star = self._map_star(rs)
                cmap[(gene, pos, ref, alt)] = (patho, star)
            except:
                continue
        
        return cmap

    def _load_label_category_mapping(self, path: str) -> dict:
        """Load label to category mapping from TSV file."""
        label_to_category = {}
        
        if not os.path.exists(path):
            print(f"Warning: Label classification file not found at {path}")
            return label_to_category
        
        try:
            print(f"Loading label-category mapping from {path}")
            df = pd.read_csv(path, sep="\t")
            
            for _, row in df.iterrows():
                term = row['term']
                primary_category = row['primary_category']
                
                if pd.notna(term) and pd.notna(primary_category):
                    label_to_category[term] = primary_category
            
            print(f"Loaded {len(label_to_category)} label-category mappings")
            return label_to_category
            
        except Exception as e:
            print(f"Error loading label classification file: {e}")
            return label_to_category

    @staticmethod
    def _map_patho(txt: str) -> float:
        """Map ClinVar pathogenicity to score."""
        if pd.isna(txt): return 0.5
        t = txt.lower()
        if "pathogenic" in t and "likely" not in t: return 0.99
        if "likely pathogenic" in t: return 0.9
        if "uncertain" in t or "conflicting" in t: return 0.5
        if "likely benign" in t: return 0.1
        if "benign" in t and "likely" not in t: return 0.01
        return 0.5

    @staticmethod
    def _map_star(txt: str) -> int:
        """Map ClinVar review status to star rating."""
        if pd.isna(txt): return 0
        t = txt.lower()
        if "practice guideline" in t: return 0.99
        if "reviewed by expert panel" in t: return 0.9
        if "criteria provided, multiple submitters, no conflicts" in t: return 0.5
        if "criteria provided" in t: return 0.01
        return 0

    def _print_model_info(self):
        """Print model information."""
        print(f"\nModel ready:")
        print(f"- Labels: {len(self.label2id)}")
        print(f"- Categories: {self.n_categories}")
        print(f"- Threshold: {self.best_threshold:.2f}")
        print(f"- ESM embeddings: {len(self.protein2esm)} proteins")
        print(f"- Domain annotations: {len(self.domains)} proteins")
        print(f"- ClinVar annotations: {len(self.clinvar)} variants")
        print(f"- Label-category mappings: {len(self.label_to_category)}")

    def _get_variant_features(self, variant_id: str) -> torch.Tensor:
        """Build variant features matching the training data format."""
        gene, mut = variant_id.split("_", 1)
        ref, pos, alt = parse_mut(mut)
        
        # ESM features
        if gene in self.protein2esm:
            esm_vec, seq_len = self.protein2esm[gene]
            esm_vec = torch.tensor(esm_vec, dtype=torch.float32)
        else:
            print(f"Warning: No ESM features for protein {gene}")
            esm_vec = torch.zeros(2560)
            seq_len = 1000
        
        # Domain features
        dom_features = torch.zeros(7)
        dom_mapping = {
            "DOMAIN": 0, "REPEAT": 1, "ZN_FING": 2,
            "COMPBIAS": 3, "REGION": 4, "COILED": 5, "MOTIF": 6
        }
        
        for feat, start, end in self.domains.get(gene, []):
            if start <= pos <= end and feat in dom_mapping:
                dom_features[dom_mapping[feat]] = 1.0
        
        # ClinVar features
        patho, star = self.clinvar.get((gene, pos, ref, alt), (0.5, 0))
        
        # Combine features
        misc_features = torch.cat([
            torch.tensor([patho, float(star)]),
            dom_features
        ])
        
        ref_oh = aa_onehot(ref)
        alt_oh = aa_onehot(alt)
        pos_norm = torch.tensor([pos / max(seq_len, 1)])
        
        features = torch.cat([
            esm_vec.flatten(),
            misc_features,
            ref_oh,
            alt_oh,
            pos_norm
        ])
        
        return features @ self.W_var

    @torch.inference_mode()
    def predict_one(self, variant_id: str, top_k: int = 30, return_categories: bool = True):
        """Predict biological processes and categories for a single variant."""
        # Get variant vector
        variant_vec = self._get_variant_features(variant_id).to(self.device)
        
        # Create batch_data structure
        batch_data = {
            'variants': variant_vec.unsqueeze(0),
            'paths': torch.zeros(0, 1, 512).to(self.device),
            'masks': torch.zeros(0, 1, dtype=torch.bool).to(self.device),
            'path_to_sample': torch.tensor([], dtype=torch.long).to(self.device),
            'paths_per_sample': torch.tensor([0]).to(self.device)
        }
        
        # Get predictions
        outputs = self.model(batch_data)
        label_probs = torch.sigmoid(outputs['label_logits']).squeeze(0)
        category_probs = torch.sigmoid(outputs['category_logits']).squeeze(0)
        
        # Get label predictions with categories
        label_preds = []
        for i, prob in enumerate(label_probs):
            if i in self.id2label:
                label_name = self.id2label[i]
                category_name = self.label_to_category.get(label_name, "Unknown")
                label_preds.append((label_name, prob.item(), category_name))
        
        label_preds.sort(key=lambda x: x[1], reverse=True)
        
        # Get category predictions
        category_preds = []
        if return_categories and self.has_categories:
            for i, prob in enumerate(category_probs):
                if i in self.id2category:
                    category_preds.append((self.id2category[i], prob.item()))
            category_preds.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'labels': label_preds[:top_k],
            'categories': category_preds[:top_k] if return_categories else []
        }

    def explain_variant(self, variant_id: str):
        """Get domain and ClinVar annotations for a variant."""
        gene, mut = variant_id.split("_", 1)
        ref, pos, alt = parse_mut(mut)
        doms = [feat for feat, start, end in self.domains.get(gene, []) 
                if start <= pos <= end]
        patho, star = self.clinvar.get((gene, pos, ref, alt), (0.5, 0))
        return doms, patho, star


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        "Hierarchical Attention BP Predictor",
        description="Predict biological processes and disease associations for protein variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model and data paths
    ap.add_argument("--model", default=os.path.join(HERE, "models", "best_model.pt"),
                    help="Path to trained model checkpoint")
    ap.add_argument("--esm_tsv", default=os.path.join(DATA_PATH, "variant_with_esm2.tsv"),
                    help="TSV file with ESM embeddings")
    ap.add_argument("--w_var", help="Path to W_var matrix (optional)")
    ap.add_argument("--label_classified", default=os.path.join(DATA_PATH, "label_classified.tsv"),
                    help="TSV file with label-category mappings")
    
    # Input options
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--variant", metavar="GENE_MUTATION",
                     help="Single variant to predict (e.g., 'BRCA1_p.Arg1699Trp')")
    grp.add_argument("--batch", metavar="FILE", nargs='?', const="variants.txt",
                     help="File with variants (one per line). Default: variants.txt")
    
    # Resource files
    ap.add_argument("--domains", default=os.path.join(DATA_PATH, "human_domains.tsv"),
                    help="UniProt domain annotations file")
    ap.add_argument("--clinvar", default=os.path.join(DATA_PATH, "clinvar_patho_subset.tsv.gz"),
                    help="ClinVar pathogenicity annotations file")
    
    # Output options
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of top predictions to show (default: 10)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device to use for prediction (default: cpu)")
    ap.add_argument("--output", "-o", metavar="FILE",
                    help="Output file for batch predictions (JSON format)")
    ap.add_argument("--no_categories", action="store_true", 
                    help="Disable category prediction output")
    
    args = ap.parse_args()
    
    # Show help if no input specified
    if not args.variant and not args.batch:
        print("\n" + "="*70)
        print("Welcome to Hierarchical BP Predictor!")
        print("="*70)
        print("\nThis tool predicts biological processes and disease associations")
        print("for protein variants using a hierarchical attention model.")
        print("\nPlease specify either a single variant or a batch file.\n")
        print("Quick examples:")
        print("  python prediction_category2label.py --variant BRCA1_p.Arg1699Trp")
        print("  python prediction_category2label.py --batch variants.txt")
        print("\nFor more help: python prediction_category2label.py -h")
        print("="*70 + "\n")
        return

    # Initialize predictor
    predictor = HierarchicalPredictor(
        args.model,
        args.esm_tsv,
        device=args.device,
        domains=args.domains,
        clinvar=args.clinvar,
        w_var_path=args.w_var,
        label_classified_path=args.label_classified
    )

    if args.variant:
        # Single variant prediction
        print(f"\n{'='*70}")
        print(f"Variant Analysis Report")
        print(f"{'='*70}")
        
        doms, patho, star = predictor.explain_variant(args.variant)
        print(f"\nVariant : {args.variant}")
        print("Domain  : " + (", ".join(doms) if doms else "None"))
        print(f"ClinVar : patho_score={patho:.2f}  star={star}")
        print("\nRunning predictions...")
        
        res = predictor.predict_one(args.variant, top_k=args.top_k, return_categories=not args.no_categories)
        
        # Print label predictions with categories
        print(f"\n{'='*80}")
        print(f"Top {len(res['labels'])} Biological Process/Disease Predictions")
        print(f"{'='*80}")
        print(f"{'#':>3}  {'Label':<35} {'Category':<25} {'Prob':>7}  {'Status':>10}")
        print("-" * 80)
        
        for r, (lbl, prob, cat) in enumerate(res['labels'], 1):
            status = "PASS" if prob > predictor.best_threshold else ""
            print(f"{r:>3}  {lbl:<35} {cat:<25} {prob*100:>6.1f}%  {status:>10}")
        
        print(f"\n{'='*70}")
        print(f"Summary (Threshold: {predictor.best_threshold:.2f})")
        print(f"{'='*70}")
        label_above = sum(1 for _, p, _ in res['labels'] if p > predictor.best_threshold)
        print(f"Labels above threshold: {label_above}")
        print(f"{'='*70}\n")
        
    else:
        # Batch prediction
        batch_file = args.batch
        
        if not os.path.exists(batch_file):
            print(f"\nError: Batch file '{batch_file}' not found.")
            print("\nPlease create a file with one variant per line, for example:")
            print("  BRCA1_R1699Q")
            return
        
        out = {}
        with open(batch_file) as f:
            variants = [line.strip() for line in f if line.strip()]
        
        if not variants:
            print(f"\nError: No variants found in '{batch_file}'")
            print("Please ensure the file contains variants in GENE_MUTATION format.")
            return
        
        print(f"\n{'='*70}")
        print(f"Batch Prediction Mode")
        print(f"{'='*70}")
        print(f"Input file: {batch_file}")
        print(f"Variants to process: {len(variants)}")
        print(f"Output: {args.output if args.output else 'Console (use -o to save)'}")
        print(f"{'='*70}\n")
        
        print("Processing variants...")
        successful = 0
        failed = 0
        
        for v in tqdm(variants, desc="Progress"):
            try:
                preds = predictor.predict_one(v, top_k=args.top_k, return_categories=not args.no_categories)
                out[v] = {
                    'labels': [(lbl, round(prob, 4), category) for lbl, prob, category in preds['labels']],
                    'categories': [(cat, round(prob, 4)) for cat, prob in preds['categories']]
                }
                successful += 1
            except Exception as e:
                out[v] = f"ERROR: {str(e)}"
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"Batch Processing Complete")
        print(f"{'='*70}")
        print(f"Successfully processed: {successful}")
        if failed > 0:
            print(f"Failed: {failed}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            print("View with: cat " + args.output + " | jq '.' | less")
        else:
            print("\nShowing first 3 results (use -o to save all):")
            print("-" * 70)
            shown = 0
            for v, pred in out.items():
                if shown >= 3:
                    print(f"\n... and {len(out) - 3} more variants")
                    print("Use -o/--output to save complete results")
                    break
                if isinstance(pred, str) and pred.startswith("ERROR"):
                    print(f"\n{v}: {pred}")
                else:
                    print(f"\n{v}:")
                    top_labels_with_categories = [
                        f"{label} ({prob*100:.1f}%) [{category}]" 
                        for label, prob, category in pred['labels'][:3]
                    ]
                    print("  Top labels:", ", ".join(top_labels_with_categories))
                shown += 1
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()