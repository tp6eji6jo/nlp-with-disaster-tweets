# %%
import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
print("TrainingArguments file =", TrainingArguments.__module__)
# %%
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
subm = pd.read_csv('data/sample_submission.csv')

# %%
def clean_text(text:  str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)

y = train["target"].astype(int)

# %%
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train["clean_text"].tolist(),
    y.tolist(),
    test_size=0.15,
    random_state=42,
    stratify=y
)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
class TxtDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
# %%
train_ds = TxtDataset(train_texts, train_labels)
val_ds   = TxtDataset(val_texts, val_labels)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

args = TrainingArguments(
    output_dir="roberta_out",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ★ 2. tokenizer（產生的 tensor 預設在 CPU）
test_enc = tokenizer(
    test["clean_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

# ★ 3. 把所有 input tensor 搬到同一個 device
test_enc = {k: v.to(device) for k, v in test_enc.items()}

# ★ 4. 推論
model.eval()
with torch.no_grad():
    logits = model(**test_enc).logits

# ★ 5. 拉回 CPU 存成 numpy（不然 pandas 會不爽）
pred = torch.argmax(logits, dim=-1).cpu().numpy()

subm["target"] = pred
subm.to_csv("submission_RoBERTa.csv", index=False)

# %%
save_directory = "./my_roberta_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)