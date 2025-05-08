import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(csv_path="data/metadata.csv", save_path="class_dist.png"):
    df = pd.read_csv(csv_path)

    # Full label map for the 5 disease classes
    label_map = {
        0: "COVID",
        1: "Normal",
        2: "Viral Pneumonia",
        3: "Tuberculosis",
        4: "Lung Cancer"
    }

    # Map numeric label to name
    df["label_name"] = df["label"].map(label_map)

    # Count and print in console
    class_counts = df["label_name"].value_counts()
    print("\nClass Distribution:")
    for label, count in class_counts.items():
        print(f"• {label}: {count}")

    # Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label_name", hue="label_name", order=class_counts.index, legend=False, palette="Set2")
    plt.title("Class Distribution")
    plt.xlabel("Disease Type")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nClass distribution plot saved as {save_path}")
