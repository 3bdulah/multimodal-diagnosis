import torch
import torch.nn as nn
import timm
from transformers import BertModel

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=5, image_model_name='vit_base_patch16_224', text_model_name='bert-base-uncased'):
        super(MultiModalClassifier, self).__init__()

        # Vision Transformer (ViT)
        self.image_model = timm.create_model(image_model_name, pretrained=True)
        self.image_model.reset_classifier(0)  # Remove the original classifier
        image_output_dim = self.image_model.num_features

        # BERT for text
        self.text_model = BertModel.from_pretrained(text_model_name)
        text_output_dim = self.text_model.config.hidden_size

        # Fusion: concatenate image and text features
        self.classifier = nn.Sequential(
            nn.Linear(image_output_dim + text_output_dim, 512),
            nn.LayerNorm(512),        # Improves training stability
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Image features
        image_feat = self.image_model(image)  # shape: [batch, image_output_dim]

        # Text features
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_outputs.pooler_output  # shape: [batch, text_output_dim]

        # Combine
        combined = torch.cat((image_feat, text_feat), dim=1)

        # Final classification
        logits = self.classifier(combined)
        return logits
