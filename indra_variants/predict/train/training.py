# -*- coding: utf-8 -*-
"""
training.py

Compare three BP prediction strategies in this framework:
1. Fine-grained label prediction (no category info)
2. Category prediction only  
3. Hierarchical prediction (category guides label)
"""

import argparse, copy, random, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, precision_score, accuracy_score,
                             roc_auc_score, average_precision_score,
                             coverage_error, recall_score)
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import warnings
import os


warnings.filterwarnings('ignore')


# Dataset with Categories and Split Loading
class HierarchicalBagDataset(Dataset):
    """Load path_dataset_bag_full.pt with dynamic split loading"""
    def __init__(self, pt_file, split_id, seed=0):
        d = torch.load(pt_file, map_location="cpu", weights_only=False)
        
        # Load split from CSV file
        split_csv = os.path.join(DATA_PATH, 'splits', f'split_seed_{seed:02d}.csv')
        if os.path.exists(split_csv):
            split_df = pd.read_csv(split_csv)
            split_dict = dict(zip(split_df['variant_id'], split_df['split']))
            
            # Match with variant_ids in dataset
            variant_ids = d["variant_ids"]
            split_array = [split_dict.get(vid, 0) for vid in variant_ids]
            sel = torch.tensor(split_array) == split_id
        else:
            # Fallback to original split if CSV not found
            sel = d["split"] == split_id
            
        idx = sel.nonzero(as_tuple=True)[0].tolist()

        # 1085 labels
        self.variant = d["variant_vec"][idx]
        self.paths_tok = [d["paths_tok"][i] for i in idx]
        self.paths_mask = [d["paths_mask"][i] for i in idx]
        self.y = d["y"][idx].float()  # (N, 1085) save all labels
        self.n_paths = d["n_paths"][idx]
        self.n_bps = d["n_bps"][idx]
        
        # 30 categories
        self.y_category = d["y_category"][idx].float()  # (N, 30)
        self.label_to_category = d["label_to_category"]
        self.category2id = d["category2id"]
        self.id2category = d["id2category"]
        self.n_categories = d["n_categories"]
        
        # mapping category to labels
        self.category_to_labels = {cat: [] for cat in range(self.n_categories)}
        for label, label_id in d["label2id"].items():
            if label in self.label_to_category:
                cat = self.label_to_category[label]
                cat_id = self.category2id[cat]
                self.category_to_labels[cat_id].append(label_id)
        
        self.n_cls = self.y.shape[1]  # 1085
        self.label2id = d["label2id"]
        self.id2label = {v: k for k, v in d["label2id"].items()}
        
        # weights for loss functions
        self.class_weights = (self.y.shape[0] - self.y.sum(0)) / self.y.sum(0).clamp(min=1)
        self.category_weights = (self.y_category.shape[0] - self.y_category.sum(0)) / self.y_category.sum(0).clamp(min=1)
        
        # statistics
        print(f"  Split {split_id} (seed={seed}): {len(self)} samples")
        print(f"    Categories: {self.n_categories}")
        print(f"    Labels: {self.n_cls}")
        print(f"    Avg paths: {self.n_paths.float().mean():.1f}")
        print(f"    Avg labels/sample: {self.y.sum(1).mean():.1f}")
        print(f"    Avg categories/sample: {self.y_category.sum(1).mean():.1f}")

    def __len__(self): 
        return len(self.y)
    
    def __getitem__(self, i):
        return (self.variant[i], self.paths_tok[i], self.paths_mask[i], 
                self.y[i], self.y_category[i])


def collate_fn_hierarchical(batch):
    """Collate function with category labels"""
    variants, paths_list, masks_list, ys, y_cats = zip(*batch)
    
    B = len(batch)
    
    # Stack fixed-size tensors
    variant_batch = torch.stack(variants)
    y_batch = torch.stack(ys)
    y_cat_batch = torch.stack(y_cats)
    
    # Handle variable-length paths
    all_paths = []
    all_masks = []
    path_to_sample = []
    paths_per_sample = []
    
    for b, (paths, masks) in enumerate(zip(paths_list, masks_list)):
        paths_per_sample.append(len(paths))
        for seq, mask in zip(paths, masks):
            all_paths.append(seq)
            all_masks.append(mask)
            path_to_sample.append(b)
    
    # Pad paths
    if all_paths:
        max_len = max(seq.shape[0] for seq in all_paths)
        padded_paths = torch.zeros(len(all_paths), max_len, 512)
        padded_masks = torch.zeros(len(all_paths), max_len, dtype=torch.bool)
        
        for i, (seq, mask) in enumerate(zip(all_paths, all_masks)):
            seq_len = seq.shape[0]
            padded_paths[i, :seq_len] = seq
            padded_masks[i, :seq_len] = mask.bool()
    else:
        padded_paths = torch.zeros(0, 1, 512)
        padded_masks = torch.zeros(0, 1, dtype=torch.bool)
    
    return {
        'variants': variant_batch,
        'paths': padded_paths,
        'masks': padded_masks,
        'path_to_sample': torch.tensor(path_to_sample if path_to_sample else []),
        'paths_per_sample': torch.tensor(paths_per_sample),
        'labels': y_batch,
        'categories': y_cat_batch
    }


# Model Architecture Components
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


# Unified Model with Three Prediction Modes
class UnifiedBPModel(nn.Module):
    """
    Unified model supporting three prediction modes:
    1. 'direct': Direct label prediction (no category info)
    2. 'category': Category prediction only
    3. 'hierarchical': Category guides label prediction
    """
    def __init__(self, n_cls, n_categories, category_to_labels, 
                 prediction_mode='hierarchical', d=512, heads=8, layers=2, drop=0.1):
        super().__init__()
        self.d = d
        self.n_categories = n_categories
        self.n_cls = n_cls
        self.category_to_labels = category_to_labels
        self.prediction_mode = prediction_mode
        
        # Shared components
        self.path_enc = PathEncoder(d, heads, layers, drop)
        
        self.variant_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        
        # path aggregator
        self.aggregator = nn.MultiheadAttention(
            d, heads, batch_first=True, dropout=drop
        )
        self.agg_norm = nn.LayerNorm(d)
        
        # Prediction heads based on mode
        if prediction_mode in ['category', 'hierarchical']:
            # Category prediction head
            self.category_head = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(d, n_categories)
            )
        
        if prediction_mode in ['direct', 'hierarchical']:
            # Label prediction head
            if prediction_mode == 'direct':
                # Direct prediction - same architecture as original
                self.label_head = nn.Sequential(
                    nn.Linear(d, d),
                    nn.LayerNorm(d),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(d, d // 2),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(d // 2, n_cls)
                )
            else:  # hierarchical
                # Hierarchical prediction - uses enhanced features
                self.label_head = nn.Sequential(
                    nn.Linear(d * 2, d),  # features + category enhanced features
                    nn.LayerNorm(d),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(d, d // 2),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(d // 2, n_cls)
                )
                
                # Category embeddings and gate for hierarchical mode
                self.category_embeddings = nn.Parameter(torch.randn(n_categories, d))
                nn.init.xavier_uniform_(self.category_embeddings)
                
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
        paths_per_sample = batch_data['paths_per_sample']
        
        B = variants.shape[0]
        device = variants.device
        
        # Shared feature extraction
        if paths.shape[0] > 0:
            path_embeddings = self.path_enc(paths, masks)
        else:
            path_embeddings = torch.zeros(0, self.d, device=device)

        v_enhanced = self.variant_proj(variants)
        
        # aggregate paths for each sample
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
        
        features = torch.cat(aggregated, dim=0)  # (B, d)
        
        # Mode-specific predictions
        result = {'features': features}
        
        if self.prediction_mode in ['category', 'hierarchical']:
            # Category prediction
            category_logits = self.category_head(features)
            result['category_logits'] = category_logits
        
        if self.prediction_mode in ['direct', 'hierarchical']:
            if self.prediction_mode == 'direct':
                # Direct label prediction
                label_logits = self.label_head(features)
            else:  # hierarchical
                # Hierarchical label prediction with category guidance
                category_probs = torch.sigmoid(result['category_logits'])
                weighted_cat_emb = torch.matmul(category_probs, self.category_embeddings)
                gate = self.category_gate(category_probs)
                category_enhanced_features = features + gate * weighted_cat_emb
                combined_features = torch.cat([features, category_enhanced_features], dim=-1)
                label_logits = self.label_head(combined_features)
            
            result['label_logits'] = label_logits
        
        return result


# Loss Functions
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        los_pos = los_pos * torch.pow(1 - xs_pos, self.gamma_pos)
        los_neg = los_neg * torch.pow(xs_pos, self.gamma_neg)

        loss = -los_pos - los_neg
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


# Unified Loss Function
class UnifiedLoss(nn.Module):
    def __init__(self, prediction_mode, category_to_labels=None, 
                 alpha=0.3, beta=0.7, gamma=0.1, 
                 label_criterion='asl', category_criterion='bce'):
        super().__init__()
        self.prediction_mode = prediction_mode
        self.category_to_labels = category_to_labels or {}
        self.alpha = alpha  # category loss weight
        self.beta = beta   # label loss weight
        self.gamma = gamma # consistency loss weight
        
        # Initialize loss functions
        if label_criterion == 'asl':
            self.label_loss = AsymmetricLoss()
        elif label_criterion == 'focal':
            self.label_loss = FocalLoss()
        else:
            self.label_loss = nn.BCEWithLogitsLoss()
        
        if category_criterion == 'focal':
            self.category_loss = FocalLoss()
        else:
            self.category_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        losses = {'total': torch.tensor(0.0, device=next(iter(predictions.values())).device)}
        
        if self.prediction_mode in ['category', 'hierarchical']:
            # Category loss
            category_logits = predictions['category_logits']
            category_targets = targets['categories']
            l_category = self.category_loss(category_logits, category_targets)
            losses['category'] = l_category
            
            if self.prediction_mode == 'category':
                losses['total'] = l_category
            else:  # hierarchical
                losses['total'] += self.alpha * l_category
        
        if self.prediction_mode in ['direct', 'hierarchical']:
            # Label loss
            label_logits = predictions['label_logits']
            label_targets = targets['labels']
            l_label = self.label_loss(label_logits, label_targets)
            losses['label'] = l_label
            
            if self.prediction_mode == 'direct':
                losses['total'] = l_label
            else:  # hierarchical
                losses['total'] += self.beta * l_label
        
        if self.prediction_mode == 'hierarchical':
            # Consistency loss
            l_consistency = self.compute_consistency_loss(
                predictions['label_logits'], predictions['category_logits'],
                targets['labels'], targets['categories']
            )
            losses['consistency'] = l_consistency
            losses['total'] += self.gamma * l_consistency
        
        return losses
    
    def compute_consistency_loss(self, label_logits, category_logits, 
                                label_targets, category_targets):
        device = label_logits.device
        consistency_loss = torch.tensor(0.0, device=device)
        
        for cat_id, label_ids in self.category_to_labels.items():
            if not label_ids:
                continue
                
            cat_prob = torch.sigmoid(category_logits[:, cat_id])
            
            if label_ids:
                valid_label_ids = [lid for lid in label_ids if lid < label_logits.shape[1]]
                if valid_label_ids:
                    label_probs = torch.sigmoid(label_logits[:, valid_label_ids])
                    max_label_prob, _ = label_probs.max(dim=1)
                    
                    pos_consistency = torch.where(
                        cat_prob > 0.5,
                        torch.relu(0.5 - max_label_prob),
                        torch.zeros_like(max_label_prob)
                    )
                    
                    neg_consistency = torch.where(
                        max_label_prob < 0.3,
                        torch.relu(cat_prob - 0.5),
                        torch.zeros_like(cat_prob)
                    )
                    
                    consistency_loss += (pos_consistency.mean() + neg_consistency.mean())
        
        return consistency_loss / len(self.category_to_labels) if self.category_to_labels else consistency_loss


# Evaluation Functions
@torch.inference_mode()
def evaluate_with_threshold_search(model, loader, dev, thresholds=[0.5]):
    """Evaluate model with multiple thresholds"""
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
        
    model.eval()
    
    # Collect predictions based on model mode
    results_data = {
        'labels': {'true': [], 'scores': []},
        'categories': {'true': [], 'scores': []}
    }
    
    with tqdm(loader, desc="Evaluating", leave=False) as pbar:
        for batch_data in pbar:
            for key in ['variants', 'paths', 'masks', 'path_to_sample', 'paths_per_sample']:
                batch_data[key] = batch_data[key].to(dev)
            
            predictions = model(batch_data)
            
            # Collect label predictions if available
            if 'label_logits' in predictions:
                label_scores = predictions['label_logits'].sigmoid()
                results_data['labels']['true'].append(batch_data['labels'].cpu())
                results_data['labels']['scores'].append(label_scores.cpu())
            
            # Collect category predictions if available
            if 'category_logits' in predictions:
                cat_scores = predictions['category_logits'].sigmoid()
                results_data['categories']['true'].append(batch_data['categories'].cpu())
                results_data['categories']['scores'].append(cat_scores.cpu())
    
    # Compute results for available predictions
    results = []
    
    for thresh in thresholds:
        result = {'threshold': thresh}
        
        # Label metrics
        if results_data['labels']['true']:
            y_true = torch.cat(results_data['labels']['true']).numpy()
            y_scores = torch.cat(results_data['labels']['scores']).numpy()
            y_pred = (y_scores > thresh).astype(float)
            label_metrics = compute_simple_metrics(y_true, y_pred, y_scores)
            result.update({f'label_{k}': v for k, v in label_metrics.items()})
        
        # Category metrics
        if results_data['categories']['true']:
            cat_true = torch.cat(results_data['categories']['true']).numpy()
            cat_scores = torch.cat(results_data['categories']['scores']).numpy()
            cat_pred = (cat_scores > thresh).astype(float)
            cat_metrics = compute_simple_metrics(cat_true, cat_pred, cat_scores)
            result.update({f'category_{k}': v for k, v in cat_metrics.items()})
        
        results.append(result)
    
    return results


def compute_simple_metrics(y_true, y_pred, y_scores):
    """Compute evaluation metrics"""
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    micro_p  = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r  = recall_score   (y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score       (y_true, y_pred, average="micro", zero_division=0)

    acc = accuracy_score(y_true, y_pred)
    
    # AUC scores
    auc_scores = []
    pr_auc_scores = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < y_true.shape[0]:
            try:
                auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                auc_scores.append(auc)
                pr_auc = average_precision_score(y_true[:, i], y_scores[:, i])
                pr_auc_scores.append(pr_auc)
            except:
                pass
    
    # Coverage
    try:
        coverage = coverage_error(y_true, y_scores)
    except:
        coverage = 0.0
    
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'micro_precision': micro_p, 'micro_recall': micro_r, 'micro_f1': micro_f1,
        'acc': acc,
        'auroc': np.mean(auc_scores) if auc_scores else 0.0,
        'auprc': np.mean(pr_auc_scores) if pr_auc_scores else 0.0,
        'coverage': coverage
    }


# Training and Evaluation Functions
def train_single_mode_seed(args, mode, seed):
    """Train model for a single mode and seed"""
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()} | SEED: {seed}")
    print(f"{'='*60}")
    
    # Set seed for this run
    torch.manual_seed(args.base_seed + seed)
    random.seed(args.base_seed + seed)
    np.random.seed(args.base_seed + seed)
    
    # GPU setup
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load datasets
    train_dataset = HierarchicalBagDataset(args.data, 0, seed)
    val_dataset = HierarchicalBagDataset(args.data, 1, seed)
    test_dataset = HierarchicalBagDataset(args.data, 2, seed)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, 
        collate_fn=collate_fn_hierarchical, num_workers=args.workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs*2, shuffle=False,
        collate_fn=collate_fn_hierarchical, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.bs*2, shuffle=False,
        collate_fn=collate_fn_hierarchical, num_workers=args.workers
    )
    
    # Model
    model = UnifiedBPModel(
        n_cls=train_dataset.n_cls,
        n_categories=train_dataset.n_categories,
        category_to_labels=train_dataset.category_to_labels,
        prediction_mode=mode,
        heads=args.heads,
        layers=args.layers,
        drop=args.dropout
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    if args.scheduler == 'onecycle':
        total_steps = len(train_loader) * args.epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function
    criterion = UnifiedLoss(
        prediction_mode=mode,
        category_to_labels=train_dataset.category_to_labels,
        alpha=args.cat_loss_weight,
        beta=args.label_loss_weight,
        gamma=args.consistency_weight,
        label_criterion=args.loss
    )
    
    # Training loop
    seed_results = []
    best_val_score = 0
    best_model_state = None
    best_epoch = 0
    
    # Define primary metric based on mode
    if mode == 'category':
        primary_metric = 'category_micro_f1'
    else:  # direct or hierarchical
        primary_metric = 'label_micro_f1'
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"{mode.capitalize()} S{seed:02d} E{epoch}/{args.epochs}", leave=False)
        for batch_data in pbar:
            for key in ['variants', 'paths', 'masks', 'path_to_sample', 'paths_per_sample']:
                batch_data[key] = batch_data[key].to(device)
            
            predictions = model(batch_data)
            targets = {
                'labels': batch_data['labels'].to(device),
                'categories': batch_data['categories'].to(device)
            }
            losses = criterion(predictions, targets)
            
            optimizer.zero_grad()
            losses['total'].backward()
            
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            if args.scheduler == 'onecycle':
                scheduler.step()
            
            train_loss += losses['total'].item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'best': f"{best_val_score:.4f}"
            })
        
        if args.scheduler == 'cosine':
            scheduler.step()
        
        avg_train_loss = train_loss / n_batches
        
        # Validation
        val_results = evaluate_with_threshold_search(model, val_loader, device, [0.5])
        
        # Add metadata
        for result in val_results:
            result.update({
                'mode': mode,
                'seed': seed,
                'epoch': epoch,
                'train_loss': avg_train_loss
            })
            seed_results.append(result)
        
        # Track best model
        current_score = val_results[0].get(primary_metric, 0)
        if current_score > best_val_score:
            best_val_score = current_score
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        # Print progress
        if epoch % 20 == 0:
            print(f"{mode.capitalize()} S{seed:02d} E{epoch}: {primary_metric}={current_score:.4f} (Best: {best_val_score:.4f} @ E{best_epoch})")
    
    # Test evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_results = evaluate_with_threshold_search(model, test_loader, device, [0.5])
    
    # Add test results
    for result in test_results:
        result.update({
            'mode': mode,
            'seed': seed,
            'epoch': -1,  # Test results
            'train_loss': 0.0
        })
        seed_results.append(result)
    
    best_test_result = test_results[0]
    test_score = best_test_result.get(primary_metric, 0)
    
    print(f"{mode.capitalize()} S{seed:02d} completed: Best Val={best_val_score:.4f} @ E{best_epoch}, Test={test_score:.4f}")
    
    return seed_results, best_test_result, best_epoch


def save_results_to_tsv(results_list, output_path):
    """Save results to TSV file"""
    if not results_list:
        print("No results to save")
        return
        
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    base_cols = ['mode', 'seed', 'epoch', 'threshold']
    metric_types = ['label', 'category']
    metric_names = ['tp', 'tn', 'fp', 'fn','micro_precision','micro_recall','micro_f1','acc', 'auroc', 'auprc', 'coverage']
    
    ordered_cols = base_cols
    for mtype in metric_types:
        for mname in metric_names:
            col = f'{mtype}_{mname}'
            if col in df.columns:
                ordered_cols.append(col)
    
    # Add remaining columns
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    df[ordered_cols].to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"Results saved to: {output_path}")


def save_comparison_summary(all_best_results, output_path):
    """Save comparison summary across all modes"""
    df = pd.DataFrame(all_best_results)
    
    # Group by mode and calculate statistics
    summary_stats = []
    
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        
        # Get metrics for this mode
        if mode == 'category':
            primary_metrics = [col for col in mode_df.columns if col.startswith('category_')]
        else:
            primary_metrics = [col for col in mode_df.columns if col.startswith('label_')]
        
        for metric in primary_metrics:
            if metric in mode_df.columns and mode_df[metric].dtype in ['float64', 'int64']:
                summary_stats.append({
                    'mode': mode,
                    'metric': metric,
                    'mean': mode_df[metric].mean(),
                    'std': mode_df[metric].std(),
                    'min': mode_df[metric].min(),
                    'max': mode_df[metric].max(),
                    'median': mode_df[metric].median()
                })
    
    # Save detailed results
    df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats)
    summary_path = output_path.replace('.tsv', '_statistics.tsv')
    summary_df.to_csv(summary_path, sep='\t', index=False, float_format='%.4f')
    
    print(f"Comparison results saved to: {output_path}")
    print(f"Summary statistics saved to: {summary_path}")


def main_unified_comparison(args):
    """Main function to run unified comparison"""
    print(f"{'='*80}")
    print("UNIFIED BP PREDICTION COMPARISON")
    print(f"{'='*80}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Seeds: {args.num_seeds}")
    print(f"Device: {'GPU' if torch.cuda.is_available() and not args.cpu else 'CPU'}")
    print(f"{'='*80}")
    
    all_results = []
    all_best_results = []
    
    for mode in args.modes:
        print(f"\nStarting mode: {mode.upper()}")
        mode_best_results = []
        
        for seed in range(args.seed_start, args.seed_end):
            seed_results, best_test_result, best_epoch = train_single_mode_seed(args, mode, seed)
            all_results.extend(seed_results)
            
            # Store best result
            best_result = {
                'mode': mode,
                'seed': seed,
                'best_epoch': best_epoch,
                **best_test_result
            }
            mode_best_results.append(best_result)
            all_best_results.append(best_result)
        
        # Print mode summary
        mode_df = pd.DataFrame(mode_best_results)
        if mode == 'category':
            primary_metric = 'category_micro_f1'
        else:
            primary_metric = 'label_micro_f1'
        
        if primary_metric in mode_df.columns:
            mean_score = mode_df[primary_metric].mean()
            std_score = mode_df[primary_metric].std()
            print(f"\n {mode.upper()} completed: {primary_metric} = {mean_score:.4f} ± {std_score:.4f}")
    
    # Save results
    save_results_to_tsv(all_results, args.output_tsv)
    save_comparison_summary(all_best_results, args.comparison_summary)
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    df_best = pd.DataFrame(all_best_results)
    
    for mode in args.modes:
        mode_df = df_best[df_best['mode'] == mode]
        print(f"\n📊 {mode.upper()} Results:")
        
        if mode == 'category':
            if 'category_micro_f1' in mode_df.columns:
                print(f"  Category F1: {mode_df['category_micro_f1'].mean():.4f} ± {mode_df['category_micro_f1'].std():.4f}")
            if 'category_auroc' in mode_df.columns:
                print(f"  Category AUROC: {mode_df['category_auroc'].mean():.4f} ± {mode_df['category_auroc'].std():.4f}")
        else:
            if 'label_micro_f1' in mode_df.columns:
                print(f"  Label F1: {mode_df['label_micro_f1'].mean():.4f} ± {mode_df['label_micro_f1'].std():.4f}")
            if 'label_auroc' in mode_df.columns:
                print(f"  Label AUROC: {mode_df['label_auroc'].mean():.4f} ± {mode_df['label_auroc'].std():.4f}")
            if mode == 'hierarchical' and 'category_micro_f1' in mode_df.columns:
                print(f"  Category F1: {mode_df['category_micro_f1'].mean():.4f} ± {mode_df['category_micro_f1'].std():.4f}")
        
        if 'best_epoch' in mode_df.columns:
            print(f"  Best Epochs: {mode_df['best_epoch'].mean():.1f} ± {mode_df['best_epoch'].std():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unified BP Prediction Comparison")
    
    # Data
    parser.add_argument("--data", default=os.path.join(DATA_PATH, "path_dataset_bag_full.pt"))
    parser.add_argument("--seed_start", type=int, default=0, help="Starting seed (inclusive)")
    parser.add_argument("--seed_end", type=int, default=30, help="Ending seed (exclusive)")
    
    # Backwards compatibility
    parser.add_argument("--num_seeds", type=int, help="Number of seeds (deprecated, use seed_start/seed_end)")
    
    # Modes to compare
    parser.add_argument("--modes", nargs='+', 
                       choices=['direct', 'category', 'hierarchical'],
                       default=['direct', 'category', 'hierarchical'],
                       help="Prediction modes to compare")
    
    # Model
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Loss
    parser.add_argument("--loss", choices=['bce', 'focal', 'asl'], default='asl')
    parser.add_argument("--cat_loss_weight", type=float, default=0.3)
    parser.add_argument("--label_loss_weight", type=float, default=0.7)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    
    parser.add_argument("--scheduler", choices=['cosine', 'onecycle'], default='onecycle')
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    
    # Output
    parser.add_argument("--output_tsv", default="unified_bp_comparison_all_results.tsv")
    parser.add_argument("--comparison_summary", default="unified_bp_comparison_summary.tsv")
    
    args = parser.parse_args()
    
    # Handle backwards compatibility
    if args.num_seeds is not None and args.seed_start == 0 and args.seed_end == 30:
        args.seed_end = args.num_seeds
        
    main_unified_comparison(args)