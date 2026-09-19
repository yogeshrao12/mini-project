import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tokenizer import get_dataloader, tokenizer, MAX_LEN
from model import DistilBertSentimentClassifier

# ================= CONFIG =================
DEVICE = torch.device("cpu")
MODEL_PATH = "saved_model/distilbert_sentiment_cpu.pt"
OUTPUT_DIR = "static/model_stats"
BATCH_SIZE = 8

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD TEST DATA =================
test_loader = get_dataloader(
    csv_path="dataset_split/test.csv",
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= LOAD MODEL =================
model = DistilBertSentimentClassifier(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded (no retraining)")

# ================= PREDICTIONS =================
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE)
        )
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ================= METRICS =================
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
cm = confusion_matrix(all_labels, all_preds)

# Save text report
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("✅ Metrics saved")

# ================= CONFUSION MATRIX IMAGE =================
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# ================= PREDICTION DISTRIBUTION =================
plt.figure(figsize=(5, 4))
sns.countplot(x=all_preds)
plt.xticks([0, 1], ["Negative", "Positive"])
plt.title("Prediction Distribution")
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/prediction_distribution.png")
plt.close()

# ================= TRUE VS PRED =================
plt.figure(figsize=(5, 4))
sns.countplot(x=all_labels)
plt.xticks([0, 1], ["Negative", "Positive"])
plt.title("True Label Distribution")
plt.xlabel("True Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/true_label_distribution.png")
plt.close()

print("✅ All images saved")
print(f"📁 Stats available in: {OUTPUT_DIR}/")
