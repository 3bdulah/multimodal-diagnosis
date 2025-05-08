import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

class MultiModalDataset(Dataset):
    def __init__(self, csv_path, image_folder, max_length=32, train=False):
        self.data = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.train = train

        self.label_to_folder = {
            0: "COVID",
            1: "Normal",
            2: "Viral Pneumonia",
            3: "Tuberculosis",
            4: "Lung Cancer"
        }

        # ImageNet normalization for pretrained ViT
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if train:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row.label)

        folder_name = self.label_to_folder[label]
        image_path = os.path.join(self.image_folder, folder_name, row.image_name)

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize symptoms text
        text = row.symptoms
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': label
        }
