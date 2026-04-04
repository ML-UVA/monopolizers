#!/usr/bin/env python3
"""Strategy analysis and clustering for Monopoly RL agents.

Takes episode traces (from evaluate_agent.py) and performs:
1. Feature extraction (9 strategy dimensions per episode)
2. KMeans clustering with automatic k selection via silhouette score
3. PCA visualization and archetype labeling

Usage:
    python scripts/analyze_strategies.py \
        --trace-dir runs/dqn_dense_networth_seed42/analysis/traces \
        --output-dir runs/dqn_dense_networth_seed42/analysis
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.trace_utils import extract_all_features, FEATURE_NAMES


# Archetype labels keyed by dominant feature
ARCHETYPE_MAP = {
    "build_rate": "Builder",
    "trade_frequency": "Trader",
    "cash_conservation": "Hoarder",
    "aggression_score": "Aggressive Buyer",
    "endgame_dominance": "Dominator",
    "peak_net_worth_ratio": "Peak Performer",
    "mortgage_rate": "Leverager",
    "property_diversity": "Diversifier",
    "time_to_first_monopoly": "Late Bloomer",
}


def select_best_k(features_scaled, k_range=(3, 7), random_state=42):
    """Select best k for KMeans by silhouette score."""
    best_k = k_range[0]
    best_score = -1

    n_samples = features_scaled.shape[0]
    # Need at least k+1 samples for k clusters
    max_k = min(k_range[1], n_samples)

    if max_k < k_range[0]:
        return k_range[0]

    for k in range(k_range[0], max_k):
        if n_samples <= k:
            break
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(features_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def label_archetypes(centroids, feature_names):
    """Assign archetype names to clusters based on their dominant features."""
    labels = []
    for centroid in centroids:
        # Find the feature with the highest absolute value
        dominant_idx = np.argmax(np.abs(centroid))
        dominant_feature = feature_names[dominant_idx]
        label = ARCHETYPE_MAP.get(dominant_feature, "Generalist")
        # Avoid duplicate labels
        if label in labels:
            # Use second-highest feature
            sorted_idxs = np.argsort(-np.abs(centroid))
            for idx in sorted_idxs:
                alt_label = ARCHETYPE_MAP.get(feature_names[idx], "Generalist")
                if alt_label not in labels:
                    label = alt_label
                    break
        labels.append(label)
    return labels


def plot_pca_clusters(features_scaled, labels, archetype_names, output_path):
    """2D PCA scatter plot colored by cluster."""
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(set(labels))))

    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        name = archetype_names[cluster_id] if cluster_id < len(archetype_names) else f"C{cluster_id}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[cluster_id]], label=name, alpha=0.7, s=40, edgecolors="k", linewidth=0.3)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Strategy Clusters (PCA Projection)")
    ax.legend(title="Archetype", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_radar(centroids, feature_names, archetype_names, output_path):
    """Radar chart showing cluster centroids."""
    n_features = len(feature_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(centroids)))

    for i, (centroid, name) in enumerate(zip(centroids, archetype_names)):
        values = centroid.tolist() + centroid[:1].tolist()
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    # Shorten labels for readability
    short_labels = [n.replace("_", "\n") for n in feature_names]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, size=7)
    ax.set_title("Cluster Centroids (Normalized Features)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_win_rates(cluster_labels, wins, archetype_names, output_path):
    """Bar chart of win rate by cluster."""
    unique_clusters = sorted(set(cluster_labels))
    win_rates = []
    names = []
    counts = []
    for c in unique_clusters:
        mask = cluster_labels == c
        wr = wins[mask].mean()
        win_rates.append(wr)
        names.append(archetype_names[c] if c < len(archetype_names) else f"C{c}")
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, win_rates, color=plt.cm.Set2(np.linspace(0, 1, len(names))),
                  edgecolor="black")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Strategy Archetype")
    ax.set_ylim(0, 1)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5, label="Random baseline (25%)")
    for bar, wr, cnt in zip(bars, win_rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, wr + 0.02,
                f"{wr:.0%}\n(n={cnt})", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Monopoly agent strategies via clustering")
    parser.add_argument("--trace-dir", required=True, help="Directory containing episode traces")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: parent of trace-dir)")
    parser.add_argument("--min-k", type=int, default=3, help="Minimum clusters (default: 3)")
    parser.add_argument("--max-k", type=int, default=7, help="Maximum clusters (default: 7)")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    output_dir = Path(args.output_dir) if args.output_dir else trace_dir.parent
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Strategy Analysis & Clustering")
    print("=" * 60)
    print(f"  Trace dir: {trace_dir}")
    print(f"  Output:    {output_dir}")

    # Extract features
    print("\nExtracting features...")
    features_df = extract_all_features(str(trace_dir))
    if features_df.empty or len(features_df) < 3:
        print(f"Need at least 3 episodes for clustering, found {len(features_df)}. Exiting.")
        sys.exit(1)

    print(f"  Extracted {len(features_df)} episodes x {len(FEATURE_NAMES)} features")

    # Save raw features
    features_path = output_dir / "strategy_features.csv"
    features_df.to_csv(features_path, index=False)
    print(f"  Features saved to {features_path}")

    # Prepare feature matrix
    X = features_df[FEATURE_NAMES].values
    # Replace NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select best k
    print("\nSelecting optimal k...")
    best_k = select_best_k(X_scaled, k_range=(args.min_k, args.max_k))
    print(f"  Best k: {best_k}")

    # Fit final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Label archetypes
    archetype_names = label_archetypes(kmeans.cluster_centers_, FEATURE_NAMES)
    print(f"  Archetypes: {archetype_names}")

    # Save cluster assignments
    features_df["cluster_id"] = labels
    features_df["archetype"] = [archetype_names[l] for l in labels]
    clusters_path = output_dir / "strategy_clusters.csv"
    features_df[["episode_id", "cluster_id", "archetype", "agent_win"]].to_csv(
        clusters_path, index=False
    )
    print(f"  Clusters saved to {clusters_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_pca_clusters(X_scaled, labels, archetype_names,
                      str(plots_dir / "strategy_clusters_pca.png"))
    plot_radar(kmeans.cluster_centers_, FEATURE_NAMES, archetype_names,
               str(plots_dir / "feature_radar.png"))

    wins = features_df["agent_win"].values.astype(float)
    plot_cluster_win_rates(labels, wins, archetype_names,
                           str(plots_dir / "cluster_win_rates.png"))

    # Print cluster summary
    print("\n" + "=" * 60)
    print("Cluster Summary")
    print("=" * 60)
    for c in range(best_k):
        mask = labels == c
        count = mask.sum()
        wr = wins[mask].mean() if count > 0 else 0
        print(f"  {archetype_names[c]:20s}: {count:4d} episodes, win rate {wr:.1%}")

        # Show top distinguishing features
        centroid = kmeans.cluster_centers_[c]
        top_idxs = np.argsort(-np.abs(centroid))[:3]
        top_feats = [(FEATURE_NAMES[i], centroid[i]) for i in top_idxs]
        for fname, fval in top_feats:
            direction = "+" if fval > 0 else "-"
            print(f"    {direction} {fname}: {fval:+.2f}σ")

    print("=" * 60)
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
