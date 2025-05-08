import matplotlib.pyplot as plt

def plot_training_curves(logs):
    train_loss = logs["train_loss"]
    train_acc = logs["train_acc"]
    train_f1 = logs["train_f1"]
    val_acc = logs["val_acc"]
    val_f1 = logs["val_f1"]
    epochs = range(1, len(train_loss) + 1)

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o", color="red", label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.close()
    print("Training loss saved as training_loss.png")

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_acc, marker="o", color="green", label="Train Acc")
    plt.plot(epochs, val_acc, marker="x", color="darkgreen", label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks(epochs)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_accuracy.png")
    plt.close()
    print("Training accuracy saved as training_accuracy.png")

    # F1 Score
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_f1, marker="o", color="blue", label="Train F1")
    plt.plot(epochs, val_f1, marker="x", color="navy", label="Val F1")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.xticks(epochs)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_f1_score.png")
    plt.close()
    print("Training f1 score saved as training_f1_score.png")
