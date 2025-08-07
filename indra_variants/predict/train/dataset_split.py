# -*- coding: utf-8 -*-
"""
dataset_split.py

Generate 30 random splits following two key principles:
1. Category coverage: ensure main categories are represented in each split
2. Protein-level splitting: avoid data leakage by keeping same protein variants together

"""

import pandas as pd
import numpy as np
import random
import os
from collections import defaultdict, Counter
from scipy.spatial.distance import jensenshannon
import warnings
import tqdm
warnings.filterwarnings('ignore')

# ─────────────────────────── file paths ────────────────────────────────
HERE = os.path.basename(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.pardir, os.pardir, 'data')
PATH_FILE = os.path.join(DATA_PATH, "variant_paths.tsv")
LABEL_CLASSIFIED_FILE = os.path.join(DATA_PATH, "label_classified.tsv")
SPLIT_OUTPUT = os.path.join(DATA_PATH, "splits")
STATS_OUTPUT = os.path.join(DATA_PATH, "split_statistics.csv")


def get_variant_protein_info(paths_df, label_to_category):
    """Extract variant-protein-category relationships"""
    variant_info = {}
    protein_to_variants = defaultdict(list)

    for variant_id in paths_df['variant_id'].unique():
        # Extract protein name
        protein = variant_id.split("_")[0]
        protein_to_variants[protein].append(variant_id)

        # Get categories for this variant
        variant_rows = paths_df[paths_df['variant_id'] == variant_id]
        categories = set()

        for bp in variant_rows[BP_COL].unique():
            if bp in label_to_category:
                categories.add(label_to_category[bp])

        variant_info[variant_id] = {
            'protein': protein,
            'categories': categories
        }

    return variant_info, dict(protein_to_variants)


def protein_aware_random_split(variant_info, protein_to_variants, all_categories,
                               train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Generate random split with two key constraints:
    1. Same protein variants stay together
    2. Major categories are represented in each split
    """
    random.seed(seed)
    np.random.seed(seed)

    # Randomly shuffle proteins
    proteins = list(protein_to_variants.keys())
    random.shuffle(proteins)

    # Initial protein-level split
    n_proteins = len(proteins)
    n_train = int(n_proteins * train_ratio)
    n_val = int(n_proteins * val_ratio)

    train_proteins = set(proteins[:n_train])
    val_proteins = set(proteins[n_train:n_train + n_val])
    test_proteins = set(proteins[n_train + n_val:])

    # Create initial variant splits
    variant_splits = {}
    for protein in train_proteins:
        for variant in protein_to_variants[protein]:
            variant_splits[variant] = 0
    for protein in val_proteins:
        for variant in protein_to_variants[protein]:
            variant_splits[variant] = 1
    for protein in test_proteins:
        for variant in protein_to_variants[protein]:
            variant_splits[variant] = 2

    # Check category coverage and make adjustments
    def update_split_categories():
        split_categories = {0: set(), 1: set(), 2: set()}
        for variant, split_id in variant_splits.items():
            for cat in variant_info[variant]['categories']:
                split_categories[split_id].add(cat)
        return split_categories

    # Find categories with sufficient samples (at least 3 variants)
    major_categories = [cat for cat in all_categories
                        if sum(1 for v in variant_info.values() if cat in v['categories']) >= 3]

    # Enhanced adjustments for better category coverage
    adjustment_count = 0
    max_adjustments = 20

    for attempt in range(3):  # Multiple passes to improve coverage
        split_categories = update_split_categories()

        missing_coverage = []
        for split_id in [0, 1, 2]:
            missing = [cat for cat in major_categories if cat not in split_categories[split_id]]
            if missing:
                missing_coverage.append((split_id, missing))

        if not missing_coverage or adjustment_count >= max_adjustments:
            break

        for split_id, missing_cats in missing_coverage:
            if adjustment_count >= max_adjustments:
                break

            for cat in missing_cats[:3]:  # Try to fix top 3 missing categories
                # Find a protein with this category that can be moved
                candidate_proteins = []
                for other_split in [0, 1, 2]:
                    if other_split == split_id:
                        continue
                    for protein in (train_proteins if other_split == 0 else
                    val_proteins if other_split == 1 else test_proteins):
                        variants = protein_to_variants[protein]
                        if any(cat in variant_info[v]['categories'] for v in variants):
                            candidate_proteins.append((protein, other_split))

                # Increased probability to make adjustments
                if candidate_proteins and random.random() < 0.6:  # Increased from 0.3 to 0.6
                    protein_to_move, from_split = random.choice(candidate_proteins)

                    # Move protein between splits
                    if from_split == 0:
                        train_proteins.remove(protein_to_move)
                    elif from_split == 1:
                        val_proteins.remove(protein_to_move)
                    else:
                        test_proteins.remove(protein_to_move)

                    if split_id == 0:
                        train_proteins.add(protein_to_move)
                    elif split_id == 1:
                        val_proteins.add(protein_to_move)
                    else:
                        test_proteins.add(protein_to_move)

                    # Update variant splits
                    for variant in protein_to_variants[protein_to_move]:
                        variant_splits[variant] = split_id

                    adjustment_count += 1
                    break

    return variant_splits, {
        'train_proteins': len(train_proteins),
        'val_proteins': len(val_proteins),
        'test_proteins': len(test_proteins),
        'adjustments_made': adjustment_count
    }


def evaluate_split_quality(all_splits, variant_info, all_categories):
    """Evaluate the randomness and quality of splits"""
    n_runs = len(all_splits)
    n_variants = len(variant_info)

    # 1. Split assignment matrix (variants x runs)
    variants = list(variant_info.keys())
    assignment_matrix = np.zeros((n_variants, n_runs))

    for run_idx, split_dict in enumerate(all_splits):
        for var_idx, variant in enumerate(variants):
            assignment_matrix[var_idx, run_idx] = split_dict[variant]

    # 2. Calculate variant split probabilities
    variant_probs = []
    expected_probs = np.array([0.7, 0.2, 0.1])  # Expected split ratios

    for var_idx in range(n_variants):
        assignments = assignment_matrix[var_idx, :]
        actual_probs = np.array([
            np.sum(assignments == 0) / n_runs,  # train
            np.sum(assignments == 1) / n_runs,  # val
            np.sum(assignments == 2) / n_runs  # test
        ])

        # Jensen-Shannon divergence from expected distribution
        js_div = jensenshannon(actual_probs, expected_probs)
        variant_probs.append({
            'variant': variants[var_idx],
            'train_prob': actual_probs[0],
            'val_prob': actual_probs[1],
            'test_prob': actual_probs[2],
            'js_divergence': js_div
        })

    # 3. Category coverage consistency
    category_coverage = {cat: {'train': [], 'val': [], 'test': []} for cat in all_categories}

    for split_dict in all_splits:
        split_categories = {0: set(), 1: set(), 2: set()}

        for variant, split_id in split_dict.items():
            for cat in variant_info[variant]['categories']:
                split_categories[split_id].add(cat)

        for cat in all_categories:
            category_coverage[cat]['train'].append(int(cat in split_categories[0]))
            category_coverage[cat]['val'].append(int(cat in split_categories[1]))
            category_coverage[cat]['test'].append(int(cat in split_categories[2]))

    # 4. Overall randomness metrics
    randomness_metrics = {
        'mean_js_divergence': np.mean([vp['js_divergence'] for vp in variant_probs]),
        'std_js_divergence': np.std([vp['js_divergence'] for vp in variant_probs]),
        'variants_with_high_bias': sum(1 for vp in variant_probs if vp['js_divergence'] > 0.1),
        'perfect_random_variants': sum(1 for vp in variant_probs if vp['js_divergence'] < 0.02)
    }

    return {
        'assignment_matrix': assignment_matrix,
        'variant_probs': variant_probs,
        'category_coverage': category_coverage,
        'randomness_metrics': randomness_metrics,
        'variants': variants
    }


if __name__ == '__main__':
    print("Loading data...")
    # Load data
    paths_df = pd.read_csv(PATH_FILE, sep="\t")
    label_class_df = pd.read_csv(LABEL_CLASSIFIED_FILE, sep="\t")

    # Create mappings
    label_to_category = dict(zip(label_class_df["term"], label_class_df["primary_category"]))
    all_categories = sorted(label_class_df["primary_category"].unique())
    category_to_id = {cat: i for i, cat in enumerate(all_categories)}
    BP_COL = "biological_process/disease"

    print(f"Loaded {len(paths_df)} path records")
    print(f"Found {len(all_categories)} categories")

    print("Analyzing variant-protein relationships...")
    variant_info, protein_to_variants = get_variant_protein_info(paths_df, label_to_category)

    print(f"Found {len(variant_info)} unique variants")
    print(f"Found {len(protein_to_variants)} unique proteins")

    # Generate 30 random splits
    print("Generating 30 random splits...")
    all_splits = []
    all_stats = []

    for seed in tqdm.tqdm(range(30)):
        split_dict, split_stats = protein_aware_random_split(
            variant_info, protein_to_variants, all_categories,
            train_ratio=0.7, val_ratio=0.2, seed=seed
        )

        all_splits.append(split_dict)

        # Calculate split statistics
        split_counts = Counter(split_dict.values())
        total_variants = len(split_dict)

        stats = {
            'seed': seed,
            'total_variants': total_variants,
            'train_variants': split_counts[0],
            'val_variants': split_counts[1],
            'test_variants': split_counts[2],
            'train_ratio': split_counts[0] / total_variants,
            'val_ratio': split_counts[1] / total_variants,
            'test_ratio': split_counts[2] / total_variants,
            **split_stats
        }
        all_stats.append(stats)

        # Save split file
        split_data = []
        for variant_id, split_id in split_dict.items():
            variant_protein = variant_id.split("_")[0]
            variant_info_str = "_".join(variant_id.split("_")[1:])

            split_data.append({
                'variant_protein': variant_protein,
                'variant_info': variant_info_str,
                'variant_id': variant_id,
                'split': split_id,
                'seed': seed
            })

        split_df = pd.DataFrame(split_data)
        split_df.to_csv(f"{SPLIT_OUTPUT}/split_seed_{seed:02d}.csv", index=False)

    print("Evaluating randomness...")
    evaluation_results = evaluate_split_quality(all_splits, variant_info, all_categories)

    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(STATS_OUTPUT, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("RANDOMNESS EVALUATION SUMMARY")
    print(f"{'='*60}")

    metrics = evaluation_results['randomness_metrics']
    print(f"Mean JS Divergence from Expected: {metrics['mean_js_divergence']:.4f}")
    print(f"Std JS Divergence: {metrics['std_js_divergence']:.4f}")
    print(f"Variants with High Bias (>0.1): {metrics['variants_with_high_bias']}/{len(variant_info)}")
    print(f"Near-Perfect Random Variants (<0.02): {metrics['perfect_random_variants']}/{len(variant_info)}")

    print(f"\nSplit Size Statistics:")
    print(f"Train - Mean: {stats_df['train_variants'].mean():.1f} ± {stats_df['train_variants'].std():.1f}")
    print(f"Val   - Mean: {stats_df['val_variants'].mean():.1f} ± {stats_df['val_variants'].std():.1f}")
    print(f"Test  - Mean: {stats_df['test_variants'].mean():.1f} ± {stats_df['test_variants'].std():.1f}")

    print(f"\nProtein Distribution:")
    print(f"Adjustments Made - Mean: {stats_df['adjustments_made'].mean():.1f} ± {stats_df['adjustments_made'].std():.1f}")

    print(f"\nFiles Generated:")
    print(f"- {SPLIT_OUTPUT}/ : 30 split CSV files")
    print(f"- {STATS_OUTPUT} : Split statistics")

    print(f"\n{'='*60}")