import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualize_comprehensive_results(data):
    # Data transformation to DataFrame
    rows = []
    for model, metrics in data.items():
        for metric, value in metrics.items():
            rows.append({"Model": model, "Metric": metric, "Value": value})
    
    df = pd.DataFrame(rows)
    
    # Visual theme settings
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"
    
    # Setup visualization layout (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle("RAG Model Performance Comparison Analysis (100-Set Benchmark)", fontsize=22, fontweight='bold', y=0.98)

    # 1. Recall@K Grouped Bar Chart
    recall_df = df[df["Metric"].str.contains("Recall")]
    sns.barplot(x="Metric", y="Value", hue="Model", data=recall_df, ax=axes[0, 0], palette="muted")
    axes[0, 0].set_title("1. Recall@K Comparison by Model", fontsize=15, pad=10)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("Performance Score")

    # 2. MRR (Mean Reciprocal Rank) Comparison
    mrr_df = df[df["Metric"] == "MRR"]
    sns.barplot(x="Model", y="Value", data=mrr_df, ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title("2. MRR (Ranking Quality) Comparison", fontsize=15, pad=10)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_ylabel("MRR Score")

    # 3. Cumulative Recall Curve (Performance Trend)
    pivot_recall = recall_df.pivot(index="Metric", columns="Model", values="Value")
    pivot_recall = pivot_recall.reindex(["Recall@1", "Recall@5", "Recall@10"])
    pivot_recall.plot(kind='line', marker='o', ax=axes[1, 0], linewidth=3, markersize=10)
    axes[1, 0].set_title("3. Probability of Success by Search Scope (K)", fontsize=15, pad=10)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].set_ylabel("Cumulative Recall Score")
    axes[1, 0].set_xlabel("Search Scope (K)")

    # 4. Metric Heatmap (Detailed Data Table)
    pivot_all = df.pivot(index="Model", columns="Metric", values="Value")
    pivot_all = pivot_all[["Recall@1", "Recall@5", "Recall@10", "MRR"]]
    sns.heatmap(pivot_all, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[1, 1], cbar=False, annot_kws={"size": 13})
    axes[1, 1].set_title("4. Detailed Metric Summary (Heatmap)", fontsize=15, pad=10)

    # 5. Radar Chart (Overall Performance Balance)
    categories = ["Recall@1", "Recall@5", "Recall@10", "MRR"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax_radar = fig.add_subplot(3, 2, 5, polar=True)
    for model in data.keys():
        values = [data[model][cat] for cat in categories]
        values += values[:1]
        ax_radar.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax_radar.fill(angles, values, alpha=0.1)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    ax_radar.set_title("5. Model Performance Balance (Radar Chart)", fontsize=15, pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Disable original subplot 5 position to make room for Radar chart
    axes[2, 0].axis('off')

    # 6. Precision Loss Analysis (Ratio of Recall@1 to Recall@10)
    # Measures the efficiency of finding the exact match given it is within Top-10
    precision_loss = []
    for model in data.keys():
        ratio = data[model]["Recall@1"] / data[model]["Recall@10"]
        precision_loss.append({"Model": model, "Top1_Efficiency": ratio})
    
    loss_df = pd.DataFrame(precision_loss)
    sns.barplot(x="Model", y="Top1_Efficiency", data=loss_df, ax=axes[2, 1], palette="flare")
    axes[2, 1].set_title("6. Search Precision Efficiency (Recall@1 / Recall@10)", fontsize=15, pad=10)
    axes[2, 1].set_ylabel("Efficiency Ratio")
    axes[2, 1].set_ylim(0, 1.0)
    axes[2, 1].axhline(y=1.0, color='r', linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("rag_performance_analysis.png", dpi=300)
    print("Performance analysis visualization saved as 'rag_performance_analysis.png'.")
    plt.show()

if __name__ == "__main__":
    # Benchmark results provided from current evaluation
    benchmark_results = {
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
            "MRR": 0.273
        }
    }
    
    visualize_comprehensive_results(benchmark_results)