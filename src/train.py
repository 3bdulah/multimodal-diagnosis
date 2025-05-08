import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import json

def train(model, train_loader, val_loader, optimizer, device, num_epochs=5, save_path="best_model.pt", early_stopping_patience=2):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    patience_counter = 0
    early_stop = False

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in loop:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect training metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loop.set_postfix(loss=loss.item())

        # Training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro')

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)

        # Validation
        val_acc, val_f1 = evaluate_on_validation(model, val_loader, device)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   — Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Best model updated — saved to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                early_stop = True
                break

    # Saving training log even if early stopping
    with open("training_log.json", "w") as f:
        json.dump(history, f)

    return history

@torch.no_grad()
def evaluate_on_validation(model, val_loader, device):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    for batch in val_loader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(images, input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1
