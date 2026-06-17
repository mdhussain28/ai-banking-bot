from fastapi import FastAPI
import random, socket, os, json, datetime, pickle
import numpy as np

app = FastAPI()

APP_ENV   = os.getenv("APP_ENV", "dev")
LOG_DIR   = os.getenv("LOG_DIR", "/data/logs")
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# High-risk keywords
HIGH_RISK_KEYWORDS = [
    "lost", "stolen", "fraud", "unauthorized",
    "blocked", "compromised", "suspicious", "hacked"
]

# Safe/normal banking queries
LOW_RISK_KEYWORDS = [
    "balance", "loan", "statement", "interest",
    "deposit", "withdrawal", "atm", "transfer",
    "kyc", "mortgage", "cheque", "account"
]


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────
def extract_features(message: str) -> list:
    msg = message.lower()

    kw_count = sum(1 for k in HIGH_RISK_KEYWORDS if k in msg)
    msg_len  = min(len(msg) / 100, 1.0)

    has_card = 1 if "card" in msg else 0
    has_acc  = 1 if "account" in msg else 0

    urgency = 1 if any(
        w in msg for w in ["immediately", "urgent", "asap", "now"]
    ) else 0

    return [kw_count, msg_len, has_card, has_acc, urgency]


# ─────────────────────────────────────────────
# Load model + scaler
# ─────────────────────────────────────────────
def load_model():
    model_path  = f"{MODEL_DIR}/risk_model.pkl"
    scaler_path = f"{MODEL_DIR}/risk_scaler.pkl"
    meta_path   = f"{MODEL_DIR}/risk_model.json"

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        return model, scaler, meta

    except Exception as e:
        print(f"Model load failed: {e}")

        return None, None, {
            "model_version": "fallback-rule-engine",
            "threshold": 0.80,
            "accuracy": 0.0,
            "fit_status": "fallback"
        }


# ─────────────────────────────────────────────
# Rule-based override
# ─────────────────────────────────────────────
def rule_based_check(message: str):
    msg = message.lower()

    # obvious fraud/high risk
    if any(k in msg for k in HIGH_RISK_KEYWORDS):
        return {
            "risk_score": 0.95,
            "flag": "high_risk"
        }

    # obvious normal banking
    if any(k in msg for k in LOW_RISK_KEYWORDS):
        return {
            "risk_score": 0.10,
            "flag": "low_risk"
        }

    return None


# ─────────────────────────────────────────────
# Risk endpoint
# ─────────────────────────────────────────────
@app.get("/risk")
def score(message: str = ""):

    # First use rule engine
    rule = rule_based_check(message)

    if rule:
        sc = rule["risk_score"]
        flag = rule["flag"]

        return {
            "risk_score": sc,
            "flag": flag,
            "model_version": "rule-based-override",
            "env": APP_ENV,
            "served_by": socket.gethostname()
        }

    # Otherwise use ML model
    model, scaler, meta = load_model()

    threshold = 0.80
    model_ver = meta.get("model_version", "unknown")

    if model and scaler:
        try:
            features = extract_features(message)

            X = np.array([features])

            # IMPORTANT FIX
            X = scaler.transform(X)

            prob = model.predict_proba(X)[0][1]

            sc = round(float(prob), 2)

        except Exception as e:
            print(f"Inference error: {e}")
            sc = 0.30

    else:
        sc = round(random.uniform(0.20, 0.40), 2)

    flag = "high_risk" if sc > threshold else "low_risk"

    # Logging
    try:
        log = (
            f"{datetime.datetime.now()} | "
            f"score={sc} | "
            f"flag={flag} | "
            f"model={model_ver} | "
            f"msg={message}\n"
        )

        with open(f"{LOG_DIR}/risk.log", "a") as f:
            f.write(log)

    except Exception:
        pass

    return {
        "risk_score": sc,
        "flag": flag,
        "model_version": model_ver,
        "env": APP_ENV,
        "served_by": socket.gethostname()
    }


# ─────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    _, _, meta = load_model()

    return {
        "status": "ok",
        "env": APP_ENV,
        "pod": socket.gethostname(),
        "fit_status": meta.get("fit_status", "unknown")
    }


# ─────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────
@app.get("/model-info")
def model_info():
    _, _, meta = load_model()
    return meta
