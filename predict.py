import torch
from transformers import DistilBertTokenizerFast
from model import DistilBertSentimentClassifier

# ================= CONFIG =================
MODEL_PATH = "saved_model/distilbert_sentiment_cpu.pt"
MAX_LEN = 200
DEVICE = torch.device("cpu")

# ================= LOAD TOKENIZER =================
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

# ================= LOAD MODEL =================
model = DistilBertSentimentClassifier(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully!")

# ================= PREDICT FUNCTION =================
def predict_sentiment(text):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            encoding["input_ids"].to(DEVICE),
            encoding["attention_mask"].to(DEVICE)
        )
        prediction = torch.argmax(outputs, dim=1).item()

    return "Positive" if prediction == 1 else "Negative"


# ================= TEST =================
if __name__ == "__main__":
    print("\nEnter text (type 'exit' to stop):\n")

    while True:
        text = input("Text: ")
        if text.lower() == "exit":
            break

        result = predict_sentiment(text)
        print("Prediction:", result, "\n")
