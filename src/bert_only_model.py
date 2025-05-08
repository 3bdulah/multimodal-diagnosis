import torch.nn as nn
from transformers import BertModel

class BERTOnlyClassifier(nn.Module):
    def __init__(self, num_classes=5, model_name='bert-base-uncased'):
        super(BERTOnlyClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image=None, input_ids=None, attention_mask=None):
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = text_output.pooler_output
        return self.classifier(pooled_output)
