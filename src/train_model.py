#This code performs dataset loading → preprocessing → tokenization → dataset creation → model training → evaluation → saving.

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer
from torch.utils.data import Dataset

# Load merged dataset
df = pd.read_csv("merged_dataset.csv")

# Drop rows with missing values
df = df.dropna(subset=['cleaned_desc', 'severity'])

# Encode severity labels numerically
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['severity'])

# Split into train, validation, test
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df['cleaned_desc'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")


#Load Roberta Tokenizer, converts raw text → tokens that go into the model.
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(
        batch['cleaned_desc'],
        padding='max_length',
        truncation=True,
        max_length=256
    )


#This wraps our text + labels into a PyTorch-compatible dataset.
class BugSeverityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Compute class weights
class_weights = compute_class_weight('balanced', classes=df['label'].unique(), y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loads pretrained RoBERTa n Adds classification layer for 4 severity labels
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(df['label'].unique()))
model.to(device)

# Replace loss with weighted cross entropy
def custom_loss(outputs, labels):
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn(outputs, labels)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1

#Prepare Training DataLoader in a batch of 16
train_dataset = BugSeverityDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#Training loop
model.train()
for epoch in range(num_epochs):
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}/{num_epochs} completed.")


#Evaluation on Validation Set
model.eval()
preds, truths = [], []

val_dataset = BugSeverityDataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16)

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        truths.extend(batch['labels'].cpu().numpy())

print(classification_report(truths, preds, target_names=label_encoder.classes_))

model.save_pretrained("severity_model")
tokenizer.save_pretrained("severity_model")
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model + tokenizer + label encoder saved!")






