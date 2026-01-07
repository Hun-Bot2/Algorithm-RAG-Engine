import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from collections import Counter

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DatasetVisualizer:
    def __init__(self, bj_path, lc_path, gt_path, pgt_path):
        self.bj_path = bj_path
        self.lc_path = lc_path
        self.gt_path = gt_path
        self.pgt_path = pgt_path
        
        # Load Data
        print("Loading datasets...")
        self.bj_data = self._load_jsonl(bj_path)
        self.lc_data = self._load_jsonl(lc_path)
        with open(gt_path, 'r', encoding='utf-8') as f:
            self.gt_data = json.load(f)
        with open(pgt_path, 'r', encoding='utf-8') as f:
            self.pgt_data = json.load(f)
            
        # Initialize light embedding model for visualization
        print("Loading MiniLM model for visualization...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _load_jsonl(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def plot_tsne_alignment(self, ax):
        print("Computing t-SNE projection...")
        bj_texts = [d['embedding_text'] for d in self.bj_data]
        lc_texts = [d['embedding_text'] for d in self.lc_data]
        
        bj_embs = self.model.encode(bj_texts, show_progress_bar=False)
        lc_embs = self.model.encode(lc_texts, show_progress_bar=False)
        
        all_embs = np.vstack([bj_embs, lc_embs])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        reduced = tsne.fit_transform(all_embs)
        
        n_bj = len(bj_embs)
        ax.scatter(reduced[:n_bj, 0], reduced[:n_bj, 1], alpha=0.6, label='Baekjoon', c='#1f77b4', s=40)
        ax.scatter(reduced[n_bj:, 0], reduced[n_bj:, 1], alpha=0.4, label='LeetCode', c='#ff7f0e', marker='x', s=40)
        ax.set_title("1. Embedding Space Alignment (t-SNE)", fontsize=14, fontweight='bold')
        ax.legend()

    def plot_tag_comparison(self, ax):
        bj_tags = []
        for d in self.bj_data: bj_tags.extend(d.get('tags', []))
        lc_tags = []
        for d in self.lc_data: lc_tags.extend(d.get('tags', []))
        
        top_n = 12
        bj_counter = Counter(bj_tags).most_common(top_n)
        lc_counter = Counter(lc_tags).most_common(top_n)
        
        bj_df = pd.DataFrame(bj_counter, columns=['Tag', 'Count'])
        bj_df['Platform'] = 'Baekjoon'
        lc_df = pd.DataFrame(lc_counter, columns=['Tag', 'Count'])
        lc_df['Platform'] = 'LeetCode'
        
        combined_df = pd.concat([bj_df, lc_df])
        sns.barplot(x='Count', y='Tag', hue='Platform', data=combined_df, ax=ax)
        ax.set_title(f"2. Top {top_n} Algorithm Tags Comparison", fontsize=14, fontweight='bold')

    def plot_difficulty_dist(self, ax):
        # Map LeetCode Easy/Medium/Hard to numeric for visualization
        diff_map = {"Easy": 5, "Medium": 15, "Hard": 25}
        
        bj_diffs = [d['difficulty'] for d in self.bj_data]
        lc_diffs = [diff_map.get(d['difficulty'], 0) for d in self.lc_data]
        
        sns.kdeplot(bj_diffs, fill=True, label='Baekjoon (1-30)', ax=ax, color='#1f77b4')
        sns.kdeplot(lc_diffs, fill=True, label='LeetCode (Mapped)', ax=ax, color='#ff7f0e')
        ax.set_title("3. Difficulty Density Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Difficulty Level (Approx. Mapping)")
        ax.legend()

    def plot_text_length(self, ax):
        bj_lens = [len(d['embedding_text'].split()) for d in self.bj_data]
        lc_lens = [len(d['embedding_text'].split()) for d in self.lc_data]
        
        data = pd.DataFrame({
            'Word Count': bj_lens + lc_lens,
            'Platform': ['Baekjoon']*len(bj_lens) + ['LeetCode']*len(lc_lens)
        })
        sns.violinplot(x='Platform', y='Word Count', data=data, ax=ax, inner="quartile")
        ax.set_title("4. Embedding Text Word Count Analysis", fontsize=14, fontweight='bold')

    def plot_gt_coverage(self, ax):
        total_bj = len(self.bj_data)
        mapped_bj = len(self.gt_data)
        unmapped = total_bj - mapped_bj
        
        ax.pie([mapped_bj, unmapped], labels=['Mapped', 'Unmapped'], autopct='%1.1f%%', 
               colors=['#2ca02c', '#d62728'], startangle=140, explode=(0.1, 0))
        ax.set_title("5. Ground Truth Coverage Rate", fontsize=14, fontweight='bold')

    def plot_potential_candidates(self, ax):
        candidate_counts = [len(candidates) for candidates in self.pgt_data.values()]
        sns.countplot(x=candidate_counts, ax=ax, palette="viridis")
        ax.set_title("6. Candidate Count Distribution per Problem", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Potential LC Candidates")
        ax.set_ylabel("BJ Problem Count")

    def run_analysis(self):
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
        plt.subplots_adjust(hspace=0.3, wspace=0.25)
        
        self.plot_tsne_alignment(axes[0, 0])
        self.plot_tag_comparison(axes[0, 1])
        self.plot_difficulty_dist(axes[1, 0])
        self.plot_text_length(axes[1, 1])
        self.plot_gt_coverage(axes[2, 0])
        self.plot_potential_candidates(axes[2, 1])
        
        plt.savefig("comprehensive_dataset_analysis.png", dpi=300, bbox_inches='tight')
        print("Analysis complete. Report saved as 'comprehensive_dataset_analysis.png'.")
        plt.show()

if __name__ == "__main__":
    visualizer = DatasetVisualizer(
        "baekjoon_refined.jsonl",
        "leetcode_refined.jsonl",
        "ground_truth_v2.json",
        "potential_gt.json"
    )
    visualizer.run_analysis()