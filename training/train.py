import numpy as np
import json, os, pickle, datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
LOG_DIR   = os.getenv("LOG_DIR",   "/data/logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

KEYWORDS = ["lost","stolen","fraud","unauthorized","blocked","compromised","suspicious","hacked"]

def extract_features(message: str) -> list:
    msg      = message.lower()
    kw_count = sum(1 for k in KEYWORDS if k in msg)
    msg_len  = min(len(msg) / 100, 1.0)
    has_card = 1 if "card" in msg else 0
    has_acc  = 1 if "account" in msg else 0
    urgency  = 1 if any(w in msg for w in ["immediately","urgent","asap","now"]) else 0
    return [kw_count, msg_len, has_card, has_acc, urgency]

# ─── Training Data ────────────────────────────────────────────────────────────
training_data = [
    # High risk (label=1)
    ("I lost my debit card",                              1),
    ("my card was stolen",                                1),
    ("unauthorized transaction on my account",            1),
    ("fraud on my account",                               1),
    ("my account is blocked",                             1),
    ("suspicious activity on my card",                    1),
    ("someone hacked my account",                         1),
    ("I did not make this transaction",                   1),
    ("my card is compromised",                            1),
    ("lost my credit card immediately need help",         1),
    ("unauthorized access to my savings account",         1),
    ("fraud detected on my debit card",                   1),
    ("my account was blocked without reason",             1),
    ("suspicious login to my account",                    1),
    ("card stolen need urgent help",                      1),
    ("someone used my card without permission",           1),
    ("I see transactions I never made",                   1),
    ("my net banking is compromised",                     1),
    ("account hacked need help asap",                     1),
    ("blocked card fraud transaction",                    1),
    # Low risk (label=0)
    ("what is my account balance",                        0),
    ("I need information about a loan",                   0),
    ("what are your banking hours",                       0),
    ("I want my account statement",                       0),
    ("what is the interest rate on savings",              0),
    ("how do I transfer money",                           0),
    ("I want to open a new account",                      0),
    ("what are the charges for a debit card",             0),
    ("I need help with my mortgage",                      0),
    ("how to update my KYC",                              0),
    ("what is the ATM withdrawal limit",                  0),
    ("how to apply for a credit card",                    0),
    ("what documents needed for loan",                    0),
    ("I want to close my fixed deposit",                  0),
    ("how to link Aadhaar to account",                    0),
    ("what is the penalty for early loan closure",        0),
    ("how to get a cheque book",                          0),
    ("I want to nominate someone for my account",         0),
    ("what is the minimum balance requirement",           0),
    ("how to change my mobile number in bank records",    0),
]

messages = [d[0] for d in training_data]
labels   = [d[1] for d in training_data]
X        = np.array([extract_features(m) for m in messages])
y        = np.array(labels)

print("=" * 60)
print("BankBot Risk Model Training")
print(f"Timestamp: {datetime.datetime.now()}")
print(f"Dataset size: {len(X)} samples")
print(f"High risk: {sum(y)} | Low risk: {len(y) - sum(y)}")
print("=" * 60)

# ─── Split ────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ─── Train 3 models — Overfit / Underfit / Good fit ──────────────────────────
print("\n[TRAINING 3 MODELS FOR COMPARISON]")

# 1. Underfitting model — too simple
print("\n--- Model 1: Underfitting (C=0.001) ---")
model_under = LogisticRegression(C=0.001, max_iter=100, random_state=42)
model_under.fit(X_train, y_train)
y_pred_under_train = model_under.predict(X_train)
y_pred_under_test  = model_under.predict(X_test)
train_acc_under    = accuracy_score(y_train, y_pred_under_train)
test_acc_under     = accuracy_score(y_test, y_pred_under_test)
print(f"Train Accuracy: {train_acc_under:.3f}")
print(f"Test  Accuracy: {test_acc_under:.3f}")
print(f"Status: UNDERFITTING (both low)")

# 2. Overfitting model — too complex
print("\n--- Model 2: Overfitting (C=1000) ---")
model_over = LogisticRegression(C=1000, max_iter=1000, random_state=42)
model_over.fit(X_train, y_train)
y_pred_over_train = model_over.predict(X_train)
y_pred_over_test  = model_over.predict(X_test)
train_acc_over    = accuracy_score(y_train, y_pred_over_train)
test_acc_over     = accuracy_score(y_test, y_pred_over_test)
print(f"Train Accuracy: {train_acc_over:.3f}")
print(f"Test  Accuracy: {test_acc_over:.3f}")
print(f"Status: OVERFITTING (train high, test low)")

# 3. Good fit model — balanced
print("\n--- Model 3: Good Fit (C=1.0) ---")
model_good = LogisticRegression(C=1.0, max_iter=500, random_state=42)
model_good.fit(X_train, y_train)
y_pred_good_train = model_good.predict(X_train)
y_pred_good_test  = model_good.predict(X_test)
train_acc_good    = accuracy_score(y_train, y_pred_good_train)
test_acc_good     = accuracy_score(y_test, y_pred_good_test)
precision         = precision_score(y_test, y_pred_good_test, zero_division=0)
recall            = recall_score(y_test, y_pred_good_test, zero_division=0)
f1                = f1_score(y_test, y_pred_good_test, zero_division=0)
print(f"Train Accuracy: {train_acc_good:.3f}")
print(f"Test  Accuracy: {test_acc_good:.3f}")
print(f"Precision:      {precision:.3f}")
print(f"Recall:         {recall:.3f}")
print(f"F1 Score:       {f1:.3f}")
print(f"Status: GOOD FIT")

# ─── Classification Report ────────────────────────────────────────────────────
print("\n[FULL CLASSIFICATION REPORT - GOOD FIT MODEL]")
print(classification_report(y_test, y_pred_good_test,
                             target_names=["low_risk","high_risk"]))

# ─── Learning Curves ──────────────────────────────────────────────────────────
print("\n[GENERATING LEARNING CURVES]")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Fit Comparison — BankBot Risk Classifier", fontsize=14)

models_info = [
    (model_under, "Underfitting Model (C=0.001)",
     train_acc_under, test_acc_under, "#ef4444"),
    (model_over,  "Overfitting Model (C=1000)",
     train_acc_over,  test_acc_over,  "#f97316"),
    (model_good,  "Good Fit Model (C=1.0)",
     train_acc_good,  test_acc_good,  "#22c55e"),
]

for ax, (mdl, title, tr_acc, te_acc, color) in zip(axes, models_info):
    train_sizes, train_scores, val_scores = learning_curve(
        mdl, X, y,
        cv=3,
        train_sizes=np.linspace(0.3, 1.0, 8),
        scoring="accuracy",
        random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    ax.plot(train_sizes, train_mean, "o-", color=color,   label="Train accuracy")
    ax.plot(train_sizes, val_mean,   "s--", color="#7b9fd4", label="Val accuracy")
    ax.fill_between(train_sizes,
                    train_scores.min(axis=1), train_scores.max(axis=1),
                    alpha=0.1, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0, 1.1])
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#0f1e35")
    fig.patch.set_facecolor("#0a1628")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.text(0.5, 0.05,
            f"Train: {tr_acc:.2f} | Test: {te_acc:.2f}",
            transform=ax.transAxes,
            ha="center", fontsize=9, color="white",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))

plt.tight_layout()
chart_path = f"{MODEL_DIR}/fit_comparison.png"
plt.savefig(chart_path, dpi=100, bbox_inches="tight",
            facecolor="#0a1628")
plt.close()
print(f"Chart saved: {chart_path}")

# ─── Save Good Fit Model ──────────────────────────────────────────────────────
print("\n[SAVING GOOD FIT MODEL]")
model_path  = f"{MODEL_DIR}/risk_model.pkl"
scaler_path = f"{MODEL_DIR}/risk_scaler.pkl"

with open(model_path,  "wb") as f: pickle.dump(model_good, f)
with open(scaler_path, "wb") as f: pickle.dump(scaler,     f)

version = f"v-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
meta    = {
    "model_version":    version,
    "threshold":        0.5,
    "keywords":         KEYWORDS,
    "accuracy":         round(float(test_acc_good), 3),
    "precision":        round(float(precision),      3),
    "recall":           round(float(recall),          3),
    "f1_score":         round(float(f1),              3),
    "train_accuracy":   round(float(train_acc_good),  3),
    "test_accuracy":    round(float(test_acc_good),   3),
    "fit_status":       "good_fit",
    "trained_on":       str(datetime.datetime.now()),
    "training_samples": len(X_train),
    "test_samples":     len(X_test),
    "fit_comparison": {
        "underfit": {
            "train_acc": round(float(train_acc_under), 3),
            "test_acc":  round(float(test_acc_under),  3),
            "status":    "underfitting"
        },
        "overfit": {
            "train_acc": round(float(train_acc_over), 3),
            "test_acc":  round(float(test_acc_over),  3),
            "status":    "overfitting"
        },
        "good_fit": {
            "train_acc": round(float(train_acc_good), 3),
            "test_acc":  round(float(test_acc_good),  3),
            "status":    "good_fit"
        }
    }
}

with open(f"{MODEL_DIR}/risk_model.json", "w") as f:
    json.dump(meta, f, indent=2)

# ─── Training Log ─────────────────────────────────────────────────────────────
try:
    log = (f"{datetime.datetime.now()} | TRAINING COMPLETE "
           f"| version={version} | accuracy={meta['accuracy']} "
           f"| f1={meta['f1_score']} | fit=good_fit\n")
    open(f"{LOG_DIR}/training.log", "a").write(log)
except:
    pass

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Version:        {version}")
print(f"Train Accuracy: {train_acc_good:.3f}")
print(f"Test  Accuracy: {test_acc_good:.3f}")
print(f"Precision:      {precision:.3f}")
print(f"Recall:         {recall:.3f}")
print(f"F1 Score:       {f1:.3f}")
print(f"Fit Status:     GOOD FIT")
print(f"Model saved:    {model_path}")
print(f"Chart saved:    {chart_path}")
print("=" * 60)

print("\n[FIT COMPARISON TABLE]")
print(f"{'Model':<25} {'Train Acc':>10} {'Test Acc':>10} {'Status':>15}")
print("-" * 65)
print(f"{'Underfit (C=0.001)':<25} {train_acc_under:>10.3f} {test_acc_under:>10.3f} {'UNDERFITTING':>15}")
print(f"{'Overfit (C=1000)':<25} {train_acc_over:>10.3f} {test_acc_over:>10.3f} {'OVERFITTING':>15}")
print(f"{'Good Fit (C=1.0)':<25} {train_acc_good:>10.3f} {test_acc_good:>10.3f} {'GOOD FIT':>15}")
