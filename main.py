from src.model import MultiModalClassifier
from src.dataset import MultiModalDataset
from src.train import train
from src.eval import evaluate
from src.explain import show_vit_attention_overlay, plot_bert_attention
from src.confusion_matrix_plot import plot_confusion_matrix
from src.class_distribution import plot_class_distribution
from src.plot_training_curves import plot_training_curves
from src.roc_plot import plot_roc_auc
from src.vit_only_model import ViTOnlyClassifier
from src.bert_only_model import BERTOnlyClassifier
from src.compare_models_plot import plot_model_comparison

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import json

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Split indices for train/val/test
full_dataset = MultiModalDataset(csv_path='data/metadata.csv', image_folder='data/images')
indices = list(range(len(full_dataset)))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# Create dataset instances
train_dataset = MultiModalDataset(csv_path='data/metadata.csv', image_folder='data/images', train=True)
val_dataset   = MultiModalDataset(csv_path='data/metadata.csv', image_folder='data/images', train=False)
test_dataset  = MultiModalDataset(csv_path='data/metadata.csv', image_folder='data/images', train=False)

# Create subsets from split indices
train_set = Subset(train_dataset, train_idx)
val_set   = Subset(val_dataset, val_idx)
test_set  = Subset(test_dataset, test_idx)

# DataLoaders
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=16, shuffle=False)

class_names = ["COVID", "Normal", "Viral Pneumonia", "Tuberculosis", "Lung Cancer"]

# Train and Evaluate MultiModal Model
model = MultiModalClassifier(num_classes=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_log = train(model, train_loader, val_loader, optimizer, device, num_epochs=5, save_path="best_model_fusion.pt")
f1_fusion = train_log["val_f1"][-1]

# Evaluate on Test Set
all_preds, all_labels, all_probs = evaluate(model, test_loader, device)

# Explainability (use a batch from test set)
for batch in test_loader:
    show_vit_attention_overlay(batch['image'][0], model, device)
    plot_bert_attention(model, batch['input_ids'][0], batch['attention_mask'][0])
    break

# Visualizations
plot_confusion_matrix(all_labels, all_preds, class_names=class_names)
plot_class_distribution('data/metadata.csv')
plot_training_curves(train_log)
plot_roc_auc(all_labels, all_probs)

# Train ViT-only Model
vit_model = ViTOnlyClassifier(num_classes=5).to(device)
vit_optimizer = optim.Adam(vit_model.parameters(), lr=1e-4)
train_log_vit = train(vit_model, train_loader, val_loader, vit_optimizer, device, num_epochs=3, save_path="best_model_vit.pt")
f1_vit = train_log_vit["val_f1"][-1]

# Train BERT-only Model
bert_model = BERTOnlyClassifier(num_classes=5).to(device)
bert_optimizer = optim.Adam(bert_model.parameters(), lr=1e-4)
train_log_bert = train(bert_model, train_loader, val_loader, bert_optimizer, device, num_epochs=3, save_path="best_model_bert.pt")
f1_bert = train_log_bert["val_f1"][-1]

# Compare Models
plot_model_comparison(f1_vit, f1_bert, f1_fusion)

# Save Final Log
with open("training_log.json", "w") as f:
    json.dump(train_log, f)

print("\nAll steps completed successfully.")
