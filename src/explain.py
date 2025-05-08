import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from transformers import BertTokenizer
import cv2

# ViT Attention Overlay (2D heatmap)
def show_vit_attention_overlay(image_tensor, model, device, save_path="vit_overlay.png"):
    model.eval()
    model.to(device)

    image_tensor = image_tensor.unsqueeze(0).to(device)
    if image_tensor.shape[-1] != 224 or image_tensor.shape[-2] != 224:
        image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear')

    attention_weights = []

    def hook_fn(module, input, output):
        attention_weights.append(output)

    # Dynamically detect ViT component for attention hooking
    try:
        if hasattr(model, "image_model"):
            hook = model.image_model.blocks[-1].attn.register_forward_hook(hook_fn)
        elif hasattr(model, "vit"):
            hook = model.vit.blocks[-1].attn.register_forward_hook(hook_fn)
        else:
            print("Unsupported model structure for ViT attention.")
            return
    except Exception as e:
        print("Hook registration failed:", e)
        return

    # Dummy input for BERT branch (ignored if ViT-only)
    dummy_ids = torch.zeros((1, 32), dtype=torch.long).to(device)
    dummy_mask = torch.ones((1, 32), dtype=torch.long).to(device)

    with torch.no_grad():
        if hasattr(model, "image_model"):
            _ = model(image_tensor, dummy_ids, dummy_mask)
        else:
            _ = model(image_tensor)

    hook.remove()

    if not attention_weights:
        print("No attention weights captured.")
        return

    attn = attention_weights[0]
    attn_cls = attn[0].mean(0)[1:]  # Exclude CLS
    patch_count = attn_cls.shape[0]
    side = int(math.sqrt(patch_count))

    if side * side == patch_count:
        attn_map = attn_cls.detach().cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = attn_map.reshape(side, side)
        attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0).float()
        attn_map = F.interpolate(attn_map, size=(224, 224), mode='bilinear').squeeze().numpy()
        attn_map = np.uint8(255 * attn_map)
        heatmap = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)

        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

        plt.imsave(save_path, overlay)
        print(f"ViT attention overlay saved as: {save_path}")
    else:
        # Fallback if not square attention
        plt.plot(attn_cls.cpu().numpy())
        plt.title("ViT Attention (1D Fallback)")
        plt.xlabel("Patch ID")
        plt.ylabel("Attention")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Fallback 1D attention saved as: {save_path}")


# BERT Attention Visualization
def plot_bert_attention(model, input_ids, attention_mask, save_path="bert_attention.png"):
    model.eval()
    device = next(model.parameters()).device

    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    # Detect BERT submodule
    if hasattr(model, "text_model"):
        bert_module = model.text_model
    elif hasattr(model, "bert"):
        bert_module = model.bert
    else:
        print("No BERT module found.")
        return

    with torch.no_grad():
        outputs = bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    last_layer = outputs.attentions[-1]  # [batch, heads, tokens, tokens]
    cls_attention = last_layer[0, 0, 0, 1:]  # CLS to all others

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())[1:]
    scores = cls_attention.cpu().numpy()

    token_score_pairs = [
        (t, s) for t, s in zip(tokens, scores)
        if t not in ["[PAD]", "[SEP]"] and not t.startswith("##")
    ]

    if not token_score_pairs:
        print("No valid tokens to visualize.")
        return

    tokens, scores = zip(*token_score_pairs)
    scores = (np.array(scores) - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(scores)), scores, color="lightcoral")
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=9)
    plt.title("BERT [CLS] Attention on Symptom Tokens")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"BERT attention plot saved as: {save_path}")
