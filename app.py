from flask import Flask, flash, render_template, redirect, session, url_for
import os
import subprocess
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast
from model import DistilBertSentimentClassifier
import mysql.connector

app = Flask(__name__)
app.secret_key = "secret123"

# ---------- DATABASE CONNECTION ----------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="sentiment"
)
cursor = db.cursor()

DATASET_DIR = "dataset_final"
MODEL_PATH = "saved_model/distilbert_sentiment_cpu.pt"

PREPROCESS_DONE = "PREPROCESS_DONE.flag"
DEVICE = torch.device("cpu")
MAX_LEN = 200


tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

model = DistilBertSentimentClassifier(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


def parse_logs(raw):
    lines = raw.splitlines()
    html = []
    in_preview = False

    for line in lines:
        if line == "PREVIEW_START":
            html.append('<div class="preview">')
            in_preview = True
            continue

        if line == "PREVIEW_END":
            html.append('</div><br>')
            in_preview = False
            continue

        if in_preview:
            html.append(line)
        else:
            html.append(f"<p>{line}</p>")

    return "\n".join(html)

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
        pred = torch.argmax(outputs, dim=1).item()

    return "Positive" if pred == 1 else "Negative"

# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataset_stats")
def dataset_stats():
    all_stats = []

    for source in os.listdir(DATASET_DIR):
        source_path = os.path.join(DATASET_DIR, source)

        if os.path.isdir(source_path):
            for file in os.listdir(source_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(source_path, file)

                    df = pd.read_csv(file_path)

                    stats = {
                        "source": source,
                        "file": file,
                        "rows": len(df),
                        "columns": list(df.columns),
                        "label_counts": df["label"].value_counts().to_dict(),
                        "avg_text_length": int(df["Text"].str.len().mean()),
                        "min_text_length": int(df["Text"].str.len().min()),
                        "max_text_length": int(df["Text"].str.len().max()),
                        "preview": df.head(5).to_dict(orient="records")
                    }

                    all_stats.append(stats)

    return render_template("dataset_stats.html", datasets=all_stats)

# ---------------- PREPROCESS ----------------
@app.route("/preprocess")
def preprocess_route():

    summary = ""
    details = ""

    if os.path.exists("preprocess_summary.log"):
        summary = open("preprocess_summary.log", encoding="utf-8").read()

    if os.path.exists("preprocess_steps.log"):
        raw = open("preprocess_steps.log", encoding="utf-8").read()
        details = parse_logs(raw)

    if os.path.exists(PREPROCESS_DONE):
        return render_template(
            "preprocess.html",
            done=True,
            summary=summary,
            details=details
        )

    subprocess.run(["python", "preprocess.py"], check=True)
    return redirect(url_for("preprocess_route"))


@app.route("/preprocess_again")
def preprocess_again():
    if os.path.exists(PREPROCESS_DONE):
        os.remove(PREPROCESS_DONE)

    subprocess.run(["python", "preprocess.py"], check=True)
    return redirect(url_for("preprocess_route"))

# ---------------- TRAIN ----------------

@app.route("/train")
def train_route():
    if not os.path.exists(MODEL_PATH):
        subprocess.run(["python", "train.py"], check=True)

    report = ""
    report_path = "static/model_stats/classification_report.txt"

    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.read()

    return render_template(
        "train.html",
        done=True,
        report=report,
        stats_ready=True
    )

@app.route("/train_again")
def train_again():
    subprocess.run(["python", "train.py"], check=True)
    return redirect(url_for("train_route"))

from flask import request

@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    result = None
    text = ""

    if request.method == "POST":
        text = request.form["text"].strip()
        if text:
            result = predict_sentiment(text)

            # ✅ SAVE HISTORY (only for logged-in users)
            if session.get("role") == "user":
                cursor.execute(
                    "INSERT INTO prediction_history (username, text, prediction) VALUES (%s, %s, %s)",
                    (session["username"], text, result)
                )
                db.commit()
        else:
            result = "Please enter some text"

    return render_template(
        "predict.html",
        prediction=result,
        text=text
    )

@app.route("/history")
def history():
    if session.get("role") != "user":
        return redirect("/user_login")

    cursor.execute(
        "SELECT id, text, prediction, created_at FROM prediction_history WHERE username=%s ORDER BY created_at DESC",
        (session["username"],)
    )
    records = cursor.fetchall()

    return render_template("history.html", records=records)

@app.route("/delete_history/<int:id>")
def delete_history(id):
    if session.get("role") != "user":
        return redirect("/user_login")

    cursor.execute(
        "DELETE FROM prediction_history WHERE id=%s AND username=%s",
        (id, session["username"])
    )
    db.commit()

    return redirect("/history")


# ---------- controller LOGIN ----------
@app.route("/controller_login", methods=["GET", "POST"])
def controller_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "controller" and password == "controller":
            session.clear()
            session["role"] = "controller"
            return redirect("/controller_dashboard")
        else:
            flash("Invalid Controller Credentials")

    return render_template("controller_login.html")

@app.route("/manage_users")
def manage_users():
    if session.get("role") != "controller":
        return redirect("/controller_login")

    cursor.execute("SELECT id, username FROM users")
    users = cursor.fetchall()

    return render_template("manage_users.html", users=users)

@app.route("/delete_user/<int:id>")
def delete_user(id):
    if session.get("role") != "controller":
        return redirect("/controller_login")

    # delete user's prediction history first (important)
    cursor.execute(
        "DELETE FROM prediction_history WHERE username = (SELECT username FROM users WHERE id=%s)",
        (id,)
    )

    # delete user
    cursor.execute("DELETE FROM users WHERE id=%s", (id,))
    db.commit()

    return redirect("/manage_users")

@app.route("/manage_feedbacks")
def manage_feedbacks():
    if session.get("role") != "controller":
        return redirect("/controller_login")

    cursor.execute(
        """SELECT id, username, rating, note, status, created_at
           FROM feedback
           ORDER BY created_at DESC"""
    )
    feedbacks = cursor.fetchall()

    return render_template("manage_feedbacks.html", feedbacks=feedbacks)

@app.route("/review_feedback/<int:id>", methods=["POST"])
def review_feedback(id):
    if session.get("role") != "controller":
        return redirect("/controller_login")

    controller_note = request.form.get("controller_note")

    cursor.execute(
        "UPDATE feedback SET status='Reviewed', controller_note=%s WHERE id=%s",
        (controller_note, id)
    )
    db.commit()

    return redirect("/manage_feedbacks")

@app.route("/controller_train_stats")
def controller_train_stats():
    if session.get("role") != "controller":
        return redirect("/controller_login")

    report = ""
    report_path = "static/model_stats/classification_report.txt"

    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.read()

    return render_template(
        "controller_train.html",
        report=report
    )

# ---------- TRAINER BUILDER LOGIN ----------
@app.route("/trainer_login", methods=["GET", "POST"])
def trainer_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "trainer" and password == "trainer":
            session.clear()
            session["role"] = "trainer"
            return redirect("/trainer_dashboard")
        else:
            flash("Invalid trainer Credentials")

    return render_template("trainer_login.html")


# ---------- USER REGISTRATION ----------
@app.route("/user_register", methods=["GET", "POST"])
def user_register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, password)
            )
            db.commit()
            flash("Registration Successful")
            return redirect("/user_login")

        except mysql.connector.Error:
            flash("Username already exists")

    return render_template("user_register.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if session.get("role") != "user":
        return redirect("/user_login")

    if request.method == "POST":
        rating = request.form.get("rating")
        note = request.form.get("note")

        cursor.execute(
            "INSERT INTO feedback (username, rating, note) VALUES (%s, %s, %s)",
            (session["username"], rating, note)
        )
        db.commit()
        flash("Feedback submitted")

        return redirect("/my_feedbacks")

    return render_template("feedback.html")

@app.route("/my_feedbacks")
def my_feedbacks():
    if session.get("role") != "user":
        return redirect("/user_login")

    cursor.execute(
        """SELECT rating, note, status, controller_note, created_at
           FROM feedback
           WHERE username=%s
           ORDER BY created_at DESC""",
        (session["username"],)
    )
    feedbacks = cursor.fetchall()

    return render_template("my_feedbacks.html", feedbacks=feedbacks)


# ---------- USER LOGIN ----------
@app.route("/user_login", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        user = cursor.fetchone()

        if user:
            session.clear()
            session["role"] = "user"
            session["username"] = username
            return redirect("/user_dashboard")
        else:
            flash("Invalid User Credentials")

    return render_template("user_login.html")


# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ---------- controller DASHBOARD ----------
@app.route("/controller_dashboard")
def controller_dashboard():
    if session.get("role") != "controller":
        return redirect("/controller_login")
    return render_template("controller_home.html")


# ----------TRAINER DASHBOARD ----------
@app.route("/trainer_dashboard")
def trainer_dashboard():
    if session.get("role") != "trainer":
        return redirect("/trainer_login")
    return render_template("trainer_home.html")


# ---------- USER DASHBOARD ----------
@app.route("/user_dashboard")
def user_dashboard():
    if session.get("role") != "user":
        return redirect("/user_login")
    return render_template("user_home.html")

if __name__ == "__main__":
    app.run(debug=True)
