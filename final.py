import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_results(report_data):
    # Data transformation for plotting
    rows = []
    for model, metrics in report_data.items():
        for metric, value in metrics.items():
            rows.append({"Model": model, "Metric": metric, "Value": value})
    
    df = pd.DataFrame(rows)
    
    # Separate Recall and MRR
    recall_df = df[df["Metric"].str.contains("Recall")]
    mrr_df = df[df["Metric"] == "MRR"]
    
    # Set visual style
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Algorithm Recommendation System Benchmark Analysis", fontsize=20, fontweight='bold')

    # 1. Bar Chart: Recall at different K
    sns.barplot(x="Metric", y="Value", hue="Model", data=recall_df, ax=axes[0, 0])
    axes[0, 0].set_title("Recall@K Comparison (Higher is Better)", fontsize=14)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Score")
    
    # 2. Bar Chart: MRR Comparison
    sns.barplot(x="Model", y="Value", data=mrr_df, ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title("MRR (Mean Reciprocal Rank) Comparison", fontsize=14)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_ylabel("Score")
    
    # 3. Line Chart: Cumulative Recall Curve
    pivot_recall = recall_df.pivot(index="Metric", columns="Model", values="Value")
    # Sort index to ensure Recall@1, 5, 10 order
    pivot_recall = pivot_recall.reindex(["Recall@1", "Recall@5", "Recall@10"])
    pivot_recall.plot(kind='line', marker='o', ax=axes[1, 0], linewidth=3, markersize=8)
    axes[1, 0].set_title("Search Quality Trend (Recall Curve)", fontsize=14)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_xlabel("Recall Level")

    # 4. Heatmap: Detailed Metric Table
    pivot_all = df.pivot(index="Model", columns="Metric", values="Value")
    pivot_all = pivot_all[["Recall@1", "Recall@5", "Recall@10", "MRR"]]
    sns.heatmap(pivot_all, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title("Performance Metric Heatmap", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_comparison.png", dpi=300)
    print("Visualization saved as 'benchmark_comparison.png'")
    plt.show()

if __name__ == "__main__":
    # Actual 100-set benchmarking results provided by user
    results = {
        "OpenAI-v3": {
            "Recall@1": 0.22,
            "Recall@5": 0.50,
            "Recall@10": 0.65,
            "MRR": 0.354
        },
        "Jina-v3": {
            "Recall@1": 0.17,
            "Recall@5": 0.47,
            "Recall@10": 0.63,
            "MRR": 0.319
        },
        "BGE-M3": {
            "Recall@1": 0.15,
            "Recall@5": 0.40,
            "Recall@10": 0.53,
            "MRR": 0.272
        }
    }
    
    visualize_results(results)