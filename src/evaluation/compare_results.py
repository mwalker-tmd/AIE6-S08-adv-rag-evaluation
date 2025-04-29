import pandas as pd
import matplotlib.pyplot as plt

def load_results(baseline_path: str, semantic_path: str) -> (pd.DataFrame, pd.DataFrame):
    baseline_df = pd.read_csv(baseline_path)
    semantic_df = pd.read_csv(semantic_path)
    return baseline_df, semantic_df

def compute_means(baseline_df: pd.DataFrame, semantic_df: pd.DataFrame) -> pd.DataFrame:
    baseline_means = baseline_df.mean(numeric_only=True)
    semantic_means = semantic_df.mean(numeric_only=True)

    combined_df = pd.DataFrame({
        "Metric": baseline_means.index,
        "Baseline": baseline_means.values,
        "Semantic": semantic_means.values
    })
    return combined_df

def plot_means(combined_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    x = range(len(combined_df["Metric"]))
    plt.bar(x, combined_df["Baseline"], width=0.4, label="Baseline", align="center")
    plt.bar([i + 0.4 for i in x], combined_df["Semantic"], width=0.4, label="Semantic", align="center")
    plt.xticks([i + 0.2 for i in x], combined_df["Metric"], rotation=45)
    plt.ylabel("Score")
    plt.title("Baseline vs Semantic Chunking - RAGAS Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()
