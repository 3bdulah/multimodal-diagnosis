import gradio as gr
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime

from src.model import MultiModalClassifier
from src.vit_only_model import ViTOnlyClassifier
from src.bert_only_model import BERTOnlyClassifier
from src.explain import show_vit_attention_overlay, plot_bert_attention
from transformers import BertTokenizer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class labels
label_map = {
    0: "COVID",
    1: "Normal",
    2: "Viral Pneumonia",
    3: "Tuberculosis",
    4: "Lung Cancer"
}

# Metadata
metadata_df = pd.read_csv("data/metadata.csv")


# Inference function
def predict_diagnosis(image, symptoms, model_type):
    if not symptoms or len(symptoms.strip()) < 3:
        return "Please enter more detailed symptoms.", None, None, None, None, None, None

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    encoded = tokenizer(symptoms, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Load model
    if model_type == "Fusion":
        model = MultiModalClassifier(num_classes=5)
        model.load_state_dict(torch.load("best_model_fusion.pt", map_location=device))
    elif model_type == "ViT-only":
        model = ViTOnlyClassifier(num_classes=5)
        model.load_state_dict(torch.load("best_model_vit.pt", map_location=device))
    elif model_type == "BERT-only":
        model = BERTOnlyClassifier(num_classes=5)
        model.load_state_dict(torch.load("best_model_bert.pt", map_location=device))
    else:
        return "Invalid model selected.", None, None, None, None, None, None

    model.to(device)
    model.eval()

    with torch.no_grad():
        if model_type == "BERT-only":
            outputs = model(input_ids, attention_mask)
        elif model_type == "ViT-only":
            outputs = model(image_tensor)
        else:  # Fusion
            outputs = model(image_tensor, input_ids, attention_mask)

        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = int(np.argmax(probs))
        confidence = probs[pred_class]

    # Top-3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_text = "\n".join([f"{label_map[i]}: {probs[i]:.2%}" for i in top3_indices])

    # Save attention maps
    vit_path = "vit_overlay.png"
    bert_path = "bert_attention.png"
    chart_path = "probabilities.png"
    report_path = "diagnosis_report.txt"

    if model_type != "BERT-only":
        show_vit_attention_overlay(image_tensor.squeeze(0), model, device, save_path=vit_path)
    else:
        vit_path = None

    if model_type != "ViT-only":
        plot_bert_attention(model, input_ids.squeeze(0), attention_mask.squeeze(0), save_path=bert_path)
    else:
        bert_path = None

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(label_map.values(), probs, color="skyblue")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    # Save diagnosis report
    with open(report_path, "w") as f:
        f.write(f"Diagnosis Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Selected Model: {model_type}\n")
        f.write(f"Prediction: {label_map[pred_class]}\n")
        f.write(f"Confidence: {confidence:.2%}\n\n")
        f.write(f"Top-3 Predictions:\n{top3_text}\n\n")
        f.write(f"Symptoms Provided: {symptoms}\n")

    return label_map[pred_class], f"{confidence:.2%}", top3_text, vit_path, bert_path, chart_path, report_path


# Build UI
def build_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Chest X-ray Diagnosis App") as demo:
        gr.Markdown(
            "## Chest X-ray Diagnosis using AI\nUpload an X-ray and type symptoms to get a predicted condition with explainability.")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Chest X-ray Image")
                text_input = gr.Textbox(label="Patient Symptoms", placeholder="e.g. fever, cough, chest pain")
                model_selector = gr.Dropdown(choices=["Fusion", "ViT-only", "BERT-only"], value="Fusion",
                                             label="Model Type")
                example_btn = gr.Button("Load Example")
                predict_btn = gr.Button("Diagnose")

            with gr.Column():
                diagnosis_label = gr.Label(label="Diagnosis")
                confidence_box = gr.Textbox(label="Confidence")
                top3_box = gr.Textbox(label="Top-3 Probabilities")
                vit_image = gr.Image(label="ViT Attention Overlay")
                bert_image = gr.Image(label="BERT Attention on Symptoms")
                prob_chart = gr.Image(label="Prediction Probabilities")
                report_file = gr.File(label="Download Diagnosis Report")

        def load_example():
            example = metadata_df.sample(1).iloc[0]

            label_index = int(example['label'])  # 0, 1, 2, etc.
            class_label = label_map[label_index]  # e.g. "COVID"

            image_name = example['image_name']
            full_image_path = f"data/images/{class_label}/{image_name}"

            symptoms = example.get('symptoms', 'fever, cough') if 'symptoms' in example else 'fever, cough'

            try:
                image = Image.open(full_image_path)
            except FileNotFoundError:
                print(f"File not found: {full_image_path}")
                return None, "Image not found.", "Fusion"

            return image, symptoms, "Fusion"

        example_btn.click(fn=load_example,
                          inputs=[],
                          outputs=[image_input, text_input, model_selector])

        predict_btn.click(fn=predict_diagnosis,
                          inputs=[image_input, text_input, model_selector],
                          outputs=[diagnosis_label, confidence_box, top3_box, vit_image, bert_image, prob_chart,
                                   report_file])

        gr.Markdown("""---  
🔬 Developed by Abdullah Alshobaki — AIN3002 Final Project  
🧠 Model: Vision Transformer + BERT | 🩺 Diseases: COVID, Pneumonia, TB, Cancer""")

    return demo


# Launch App
app = build_interface()
app.launch(share=True)
