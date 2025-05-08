import matplotlib.pyplot as plt

def plot_model_comparison(f1_vit, f1_bert, f1_fusion, save_path="modality_comparison.png"):
    models = ["ViT Only", "BERT Only", "ViT + BERT"]
    scores = [f1_vit, f1_bert, f1_fusion]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, scores, color=["#3498db", "#9b59b6", "#2ecc71"])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

    plt.title("Model Comparison (F1 Score)")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nComparison chart saved as {save_path}")
