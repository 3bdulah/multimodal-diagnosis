import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_auc(y_true, y_probs, save_path="roc_auc.png"):
    n_classes = 5
    y_true_bin = label_binarize(y_true, classes=[0,1,2,3,4])
    y_probs = np.array(y_probs)

    fpr = {}
    tpr = {}
    roc_auc = {}
    class_names = ["COVID", "Normal", "Viral Pneumonia", "Tuberculosis", "Lung Cancer"]
    colors = ["red", "green", "blue", "purple", "orange"]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Class")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved as {save_path}\n")
