import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# ================= PATHS =================
INPUT_DIR = "dataset_final"
OUTPUT_DIR = "dataset_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAIL_LOG = "preprocess_steps.log"
SUMMARY_LOG = "preprocess_summary.log"
FLAG_FILE = "PREPROCESS_DONE.flag"

# ================= LOGGING =================
def log(msg, file):
    print(msg)
    with open(file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def preview_df(df, n=3):
    return df.head(n).to_html(
        index=False,
        classes="table table-bordered table-sm",
        escape=False
    )



# clear logs on fresh run
open(DETAIL_LOG, "w").close()
open(SUMMARY_LOG, "w").close()

# ================= TEXT CLEANING =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean(path):
    df = pd.read_csv(path)
    df["Text"] = df["Text"].apply(clean_text)
    return df

# ================= LOAD DATA =================
log("🧹 Loading datasets...", DETAIL_LOG)

amazon_df = load_and_clean(os.path.join(INPUT_DIR, "amazon", "amazon_binary_final.csv"))
imdb_df   = load_and_clean(os.path.join(INPUT_DIR, "imdb", "imdb_binary_final.csv"))
tweets_df = load_and_clean(os.path.join(INPUT_DIR, "tweets", "tweets_binary_final.csv"))

log("📂 amazon_binary_final.csv", DETAIL_LOG)
log(f"Samples: {len(amazon_df)}", DETAIL_LOG)
log("PREVIEW_START", DETAIL_LOG)
log(preview_df(amazon_df), DETAIL_LOG)
log("PREVIEW_END", DETAIL_LOG)

log("📂 imdb_binary_final.csv", DETAIL_LOG)
log(f"Samples: {len(imdb_df)}", DETAIL_LOG)
log("PREVIEW_START", DETAIL_LOG)
log(preview_df(imdb_df), DETAIL_LOG)
log("PREVIEW_END", DETAIL_LOG)

log("📂 tweets_binary_final.csv", DETAIL_LOG)
log(f"Samples: {len(tweets_df)}", DETAIL_LOG)
log("PREVIEW_START", DETAIL_LOG)
log(preview_df(tweets_df), DETAIL_LOG)
log("PREVIEW_END", DETAIL_LOG)

# ================= CLEANING INFO =================
log(
    "🧼 Text cleaning steps:\n"
    "- Lowercasing\n"
    "- Removing URLs\n"
    "- Removing HTML tags\n"
    "- Removing mentions\n"
    "- Removing special characters\n"
    "- Normalizing spaces\n",
    DETAIL_LOG
)

# ================= SAMPLING =================
tweets_df = tweets_df.sample(n=30000, random_state=42)
amazon_df = amazon_df.sample(n=50000, random_state=42)
imdb_df   = imdb_df.sample(n=50000, random_state=42)

# ================= MERGE =================
log("🔗 Merging datasets...", DETAIL_LOG)
merged_df = pd.concat([amazon_df, imdb_df, tweets_df], ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

log(f"Total merged samples: {len(merged_df)}", DETAIL_LOG)
log(f"Preview:\n{preview_df(merged_df)}\n", DETAIL_LOG)

# ================= SPLIT =================
log("✂️ Performing stratified train-test split...", DETAIL_LOG)

train_df, test_df = train_test_split(
    merged_df,
    test_size=0.2,
    random_state=42,
    stratify=merged_df["label"]
)

log(f"Train size before downsampling: {len(train_df)}", DETAIL_LOG)
log(f"Test size before downsampling: {len(test_df)}", DETAIL_LOG)

# ================= DOWNSAMPLING =================
log("📉 Downsampling to manageable size...", DETAIL_LOG)

TRAIN_PER_CLASS = 2500
TEST_PER_CLASS = 500

train_pos = train_df[train_df["label"] == 1].sample(n=TRAIN_PER_CLASS, random_state=42)
train_neg = train_df[train_df["label"] == 0].sample(n=TRAIN_PER_CLASS, random_state=42)

test_pos = test_df[test_df["label"] == 1].sample(n=TEST_PER_CLASS, random_state=42)
test_neg = test_df[test_df["label"] == 0].sample(n=TEST_PER_CLASS, random_state=42)

train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)
test_df  = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

# ================= SAVE =================
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# ================= SUMMARY =================
log("✅ Preprocessing completed successfully", SUMMARY_LOG)
log(f"Final train size: {len(train_df)}", SUMMARY_LOG)
log(f"Final test size : {len(test_df)}", SUMMARY_LOG)
log(f"Train distribution:\n{train_df['label'].value_counts()}", SUMMARY_LOG)
log(f"Test distribution:\n{test_df['label'].value_counts()}", SUMMARY_LOG)

# ================= FLAG =================
open(FLAG_FILE, "w").close()
