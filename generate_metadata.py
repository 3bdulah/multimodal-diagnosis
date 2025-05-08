import os
import csv
import random
import re

label_map = {
    "COVID": 0,
    "Normal": 1,
    "Viral Pneumonia": 2,
    "Tuberculosis": 3,
    "Lung Cancer": 4
}

symptom_pool = {
    "COVID": ["fever", "dry cough", "chest pain", "loss of taste", "shortness of breath", "fatigue"],
    "Normal": ["no symptoms", "routine check", "healthy", "no complaints"],
    "Viral Pneumonia": ["cough", "sore throat", "runny nose", "headache", "fatigue", "difficulty breathing"],
    "Tuberculosis": ["chronic cough", "night sweats", "fever", "weight loss", "coughing blood"],
    "Lung Cancer": ["persistent cough", "chest pain", "coughing blood", "hoarseness", "recurrent infections",
                    "shortness of breath"]
}

genders = ["Male", "Female"]


# Extract number from filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1


def process_class(class_name, folder_path, label):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

    if class_name in ["COVID", "Normal", "Viral Pneumonia"]:
        files.sort(key=extract_number)

    print(f"{class_name} → {len(files)} files")

    rows = []
    for fname in files:
        age = random.randint(20, 85)
        gender = random.choice(genders)

        # Base symptoms (from the correct class)
        base_symptoms = random.sample(symptom_pool[class_name], k=random.randint(2, 3))

        # Add 1 symptom from a different class with 25% chance
        if random.random() < 0.25:
            other_classes = [k for k in symptom_pool if k != class_name]
            other_class = random.choice(other_classes)
            noisy_symptom = random.choice(symptom_pool[other_class])
            base_symptoms.append(noisy_symptom)

        # Shuffle symptom order
        random.shuffle(base_symptoms)

        symptoms = "; ".join(base_symptoms)
        rows.append([fname, age, gender, symptoms, label])
    return rows



def generate_metadata(image_dir, output_csv):
    rows = [["image_name", "age", "gender", "symptoms", "label"]]

    for class_name, label in label_map.items():
        folder_path = os.path.join(image_dir, class_name)
        if not os.path.isdir(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue
        rows += process_class(class_name, folder_path, label)

    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)

    print(f"\nMetadata written to: {output_csv} — {len(rows) - 1} records")


# Run
if __name__ == "__main__":
    generate_metadata(
        image_dir=r"C:\Users\Abdullah\Desktop\AIN3002\Project\multi_modal_diagnosis\data\images",
        output_csv=r"C:\Users\Abdullah\Desktop\AIN3002\Project\multi_modal_diagnosis\data\metadata.csv"
    )
