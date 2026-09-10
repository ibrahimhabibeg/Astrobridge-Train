#!/usr/bin/env python
"""Compare two models with a paired bootstrap test.

Usage:
    uv run eval_scripts/compare_models.py \
        --baseline eval_results/20260910_run1 \
        --treatment eval_results/20260910_run2 \
        --metric macro-f1
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from evals.metrics.classification import accuracy, macro_f1


METRIC_FUNCTIONS = {
    "accuracy": accuracy,
    "macro-f1": macro_f1,
}


def main():
    parser = argparse.ArgumentParser(description="Run a paired bootstrap test to compare two models.")
    parser.add_argument("--baseline", type=str, required=True, help="Path to the baseline model's results directory.")
    parser.add_argument("--treatment", type=str, required=True, help="Path to the treatment (fine-tuned) model's results directory.")
    parser.add_argument("--metric", type=str, choices=list(METRIC_FUNCTIONS.keys()), default="macro-f1", help="Evaluation metric to use.")
    parser.add_argument("--unknown-strategy", type=str, choices=["penalize", "drop"], default="penalize", help="How to handle UNKNOWN outputs.")
    parser.add_argument("--n-bootstraps", type=int, default=10000, help="Number of bootstrap resamples.")
    
    args = parser.parse_args()
    
    baseline_path = os.path.join(args.baseline, "results.jsonl")
    treatment_path = os.path.join(args.treatment, "results.jsonl")
    
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} does not exist.")
        return
    if not os.path.exists(treatment_path):
        print(f"Error: {treatment_path} does not exist.")
        return

    # Load data
    df_base = pd.read_json(baseline_path, lines=True)
    df_treat = pd.read_json(treatment_path, lines=True)

    # Rename columns to avoid collisions
    df_base = df_base.add_suffix('_base')
    df_base = df_base.rename(columns={'wiki_entity_id_base': 'wiki_entity_id'})
    
    df_treat = df_treat.add_suffix('_treat')
    df_treat = df_treat.rename(columns={'wiki_entity_id_treat': 'wiki_entity_id'})

    # Inner join on wiki_entity_id to ensure perfect pairing
    df = pd.merge(df_base, df_treat, on="wiki_entity_id", how="inner")
    
    if len(df) == 0:
        print("Error: No matching wiki_entity_ids found between the two result sets.")
        return
        
    print(f"Loaded {len(df)} aligned predictions.")
    
    # Stats on format errors (None / UNKNOWN predictions)
    base_unknowns = (df['model_answer_base'].isna() | (df['model_answer_base'] == 'UNKNOWN')).sum()
    treat_unknowns = (df['model_answer_treat'].isna() | (df['model_answer_treat'] == 'UNKNOWN')).sum()
    
    print("\n--- Formatting Statistics ---")
    print(f"Baseline format errors:  {base_unknowns} / {len(df)} ({(base_unknowns/len(df))*100:.2f}%)")
    print(f"Treatment format errors: {treat_unknowns} / {len(df)} ({(treat_unknowns/len(df))*100:.2f}%)")
    
    # Handle format errors based on strategy
    if args.unknown_strategy == "drop":
        mask = (
            df['model_answer_base'].notna() & (df['model_answer_base'] != 'UNKNOWN') &
            df['model_answer_treat'].notna() & (df['model_answer_treat'] != 'UNKNOWN')
        )
        df = df[mask]
        print(f"\nStrategy 'drop' applied. Kept {len(df)} clean intersecting rows.")
    else:
        # Fill None with UNKNOWN so metric functions can handle them
        df['model_answer_base'] = df['model_answer_base'].fillna('UNKNOWN')
        df['model_answer_treat'] = df['model_answer_treat'].fillna('UNKNOWN')
        print("\nStrategy 'penalize' applied. Format errors are kept and scored as incorrect.")
        
    if len(df) == 0:
        print("Error: No data left to evaluate after dropping format errors.")
        return

    # Extract numpy arrays for fast evaluation
    y_true = df['correct_answer_base'].values
    y_base = df['model_answer_base'].values
    y_treat = df['model_answer_treat'].values
    
    # Sanity check: ground truths must match
    assert (df['correct_answer_base'] == df['correct_answer_treat']).all(), "Ground truth labels disagree!"

    # Select metric function
    metric_fn = METRIC_FUNCTIONS[args.metric]

    # 1. Calculate observed metrics
    score_base = metric_fn(y_true, y_base)
    score_treat = metric_fn(y_true, y_treat)
    delta_obs = score_treat - score_base
    
    print(f"\n--- Observed Performance ({args.metric}) ---")
    print(f"Baseline Score:  {score_base:.4f}")
    print(f"Treatment Score: {score_treat:.4f}")
    print(f"Observed Delta: {delta_obs:+.4f}")

    # 2. Bootstrap loop
    print(f"\nRunning {args.n_bootstraps} paired bootstrap resamples...")
    deltas = np.zeros(args.n_bootstraps)
    n = len(y_true)
    
    np.random.seed(42)
    
    for i in tqdm(range(args.n_bootstraps), desc="Bootstrapping", leave=False):
        indices = np.random.choice(n, size=n, replace=True)
        
        y_true_boot = y_true[indices]
        y_base_boot = y_base[indices]
        y_treat_boot = y_treat[indices]
        
        b_score = metric_fn(y_true_boot, y_base_boot)
        t_score = metric_fn(y_true_boot, y_treat_boot)
        
        deltas[i] = t_score - b_score
        
    # 3. P-value and confidence interval
    p_value = np.mean(deltas <= 0)
    
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    
    print("\n--- Bootstrap Results ---")
    print(f"95% Confidence Interval for Delta: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"p-value (H0: Treatment <= Baseline): {p_value:.8f}")
    
    if p_value < 0.05:
        print("\n Conclusion: The treatment is statistically significantly BETTER than the baseline (p < 0.05).")
    else:
        print("\n Conclusion: The difference is NOT statistically significant (p >= 0.05).")

if __name__ == "__main__":
    main()
