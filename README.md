![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![Gradio](https://img.shields.io/badge/Gradio-5.25-green)

# 🧠 Multi-Modal Chest X-ray Diagnosis using Vision Transformers and BERT

This deep learning project combines chest X-ray images and patient symptom descriptions to diagnose **five respiratory diseases** using a multi-modal AI system. It leverages **Vision Transformers (ViT)** for image analysis and **BERT** for understanding patient symptoms, offering both high accuracy and strong interpretability.

---

## 🔍 Project Overview

* **Image Encoder:** `ViT (vit_base_patch16_224)`
* **Text Encoder:** `BERT (bert-base-uncased)`
* **Fusion Strategy:** Concatenation of ViT and BERT embeddings
* **Classifier:** A two-layer `MLP` with `LayerNorm`, `ReLU`, and `Dropout`
* **Explainability:**
  * **ViT:** Grad-CAM-style attention overlays on X-ray images
  * **BERT:** Token-level attention visualization
* **Deployment:** Gradio Web App for prediction and interpretability

---

## 🧠 Dataset Summary

| Disease         | Image Count |
| --------------- | ----------- |
| COVID-19        | 3616        |
| Normal          | 3000        |
| Lung Cancer     | 2340        |
| Viral Pneumonia | 1345        |
| Tuberculosis    | 662         |

* Includes **synthetic metadata**:`image_name`, `age`, `gender`, `symptoms` (e.g., "fatigue, chest pain"), and `label`.

📌 **Dataset Sources:**

🦠 **COVID-19, Normal, Viral Pneumonia:**
COVID-19 Radiography Database
([https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data))

🧫 **Tuberculosis:**
Tuberculosis Chest X-rays (Shenzhen)
([https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen/data](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen/data))

🧬 **Lung Cancer** *(selected folder: images\_001)*:
NIH Chest X-ray
([https://www.kaggle.com/datasets/nih-chest-xrays/data/data](https://www.kaggle.com/datasets/nih-chest-xrays/data/data))

**IMPORTANT:**
Due to GitHub file size limits, the full image dataset used for training and evaluation is provided via Google Drive:

📌 Download Link:
👉 https://drive.google.com/file/d/1lTrPZdmJWNK6vUdf7w0bLdjGTYIDwlTl/view?usp=sharing

After downloading:

🔻 Unzip and place the `images/` folder inside the `data/` directory:

```
multi_modal_diagnosis/
├── data/
│   └── images/
│       ├── COVID/
│       ├── Normal/
│       ├── Lung Cancer/
│       ├── Viral Pneumonia/
│       └── Tuberculosis/
```

This structure is necessary for the dataset loader to work correctly.


---

## 📂 Project Structure

| File / Folder          | Description                                       |
|------------------------|---------------------------------------------------|
| `main.py`              | Main training script (ViT, BERT, Fusion)          |
| `app.py`               | Gradio app for inference and visualization        |
| `generate_metadata.py` | Generates synthetic clinical descriptions         |
| `data/metadata.csv`    | Image names, labels, and symptoms                 |
| `data/images/`         | Subfolders for each class containing chest X-rays |
| `requirements.txt`     | Required Python packages                          |
| `training_log.json`    | Logs training metrics (loss, acc, F1)             |
| `*.pt`                 | Trained model checkpoints                         |
| `diagnosis_report.txt` | Auto-generated Gradio report                      |
| `*.png`                | Output visualizations                             |
| `src/`                 | Source code for models, plots, utils              |

### `src/` Folder Contents:

* `bert_only_model.py` – BERT-only classifier
* `vit_only_model.py` – ViT-only classifier
* `model.py` – Fusion classifier
* `explain.py` – Attention visualizations (ViT + BERT)
* `dataset.py` – Dataset loading and preprocessing
* `train.py` – Training loop
* `eval.py` – Evaluation logic
* `confusion_matrix_plot.py`, `roc_plot.py`, `class_distribution.py` – Plots
* `plot_training_curves.py` – Accuracy, loss, F1 over epochs
* `compare_models_plot.py` – Comparison across modalities

---

## 📦 Required Library Versions

```
# Not all libraries are directly used in every .py file,
# but to ensure smooth execution, install all:
pip install -r requirements.txt
```

Key dependencies:

* `torch==2.6.0`
* `transformers==4.51.3`
* `timm==1.0.15`
* `gradio==5.25.2`
* `opencv-python==4.11.0.86`
* `matplotlib==3.10.1`
* `scikit-learn==1.6.1`
* `pandas==2.2.3`
* `numpy==2.2.4`
* `Pillow==11.2.1`

---

## 🧪 Model Performance (Fusion)

| Metric    | Value  |
| --------- |--------|
| Accuracy  | 96.05% |
| Precision | 96.21% |
| Recall    | 96.05% |
| F1 Score  | 96.05% |

📍 **Note:** The fusion model was trained for up to 5 epochs with early stopping based on validation F1-score. Results reflect performance on the held-out test set. Accuracy may further improve with extended training, advanced hyperparameter tuning, or use of a more powerful compute environment. `Early stopping was applied to the fusion model during training. ViT and BERT baselines were trained for fixed 3 epochs.`

---

## 📊 Visual Outputs

* `confusion_matrix.png` — Class-wise prediction performance
* `roc_auc.png` — ROC curves for all five diseases
* `class_dist.png` — Dataset balance
* `training_loss.png`, `training_accuracy.png`, `training_f1_score.png` — Training progress
* `vit_overlay.png` — ViT attention (Grad-CAM-style image overlay)
* `bert_attention.png` — BERT token importance
* `modality_comparison.png` — ViT vs BERT vs Fusion results
* `probabilities.png` — Class prediction probabilities 
* `gradio_demo.png` — Gradio Web Interface Demo
* `model_flowchart.png` — Model Architecture (custom generated to visualize architecture and modality flow)
* `early_stop_colab.png` — Training log from Colab showing early stopping triggered after epoch 3
---

## 🦚 Model Comparison (3 Epochs)

| Model     | Accuracy | F1 Score |
| --------- |----------|----------|
| ViT-only  | 83.37%   | 0.8493   |
| BERT-only | 91.31%   | 0.8881   |
| Fusion    | 96.05% ✅ | 0.9605 ✅ |

---

## 💻 How to Run the Project Locally

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Train the models** (Optional if `.pt` files are already present):

```bash
python main.py
```

3. **Launch the Gradio Web Interface:**

```bash
python app.py
```

---

## 📁 Output Files

* `best_model_fusion.pt`, `best_model_vit.pt`, `best_model_bert.pt` — Trained weights
* `training_log.json` — Logs of all training metrics
* `diagnosis_report.txt` — Downloadable result report from app
* `.png` files — Visual outputs listed above (except `model_flowchat.png`, `gradio_demo.png`, and `early_stop_colab.png`)

---

## 📄 Final Report (AIN3002_FinalReport_AbdullahAlShobaki.pdf)

The report is located in the root directory and contains the following sections:

### 📾 Contents:

* Title & Author Information
* Abstract
* Related Work
* Methodology
  * Model Architecture (ViT + BERT fusion)
  * Full Multimodal Pipeline Flowchart
* Data Description
* Experiments & Results
  * Performance Metrics
  * Training Setup & Hyperparameters
  * Hardware Specs (Google Colab Pro with Tesla T4 GPU)
* Conclusion
* Supplementary Material
  * Attention Visualizations (ViT & BERT)
  * Training Curves & Confusion Matrix
  * ROC Curves & Implementation Notes

---

## 🧠 Acknowledgments

* 🔬 Developed by: **Abdullah Hani Abdellatif Alshobaki**
* 🎓 Course: **AIN3002 – Deep Learning Final Project**
* 👨‍🏫 Supervised by: **Dr. Arezoo Sadeghzadeh** and **Dr. Fatih Kahraman**
