import torch.nn as nn
import timm

class ViTOnlyClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name='vit_base_patch16_224'):
        super(ViTOnlyClassifier, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.vit.reset_classifier(num_classes)

    def forward(self, image, input_ids=None, attention_mask=None):
        return self.vit(image)
