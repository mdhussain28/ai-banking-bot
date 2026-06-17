
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import requests, socket, os, json, datetime, jwt, hashlib
from passlib.context import CryptContext
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2, io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Config from ConfigMap/Secrets ───────────────────────────────────────────
PROMPT        = os.getenv("SYSTEM_PROMPT",   "You are a banking support assistant.")
RISK_URL      = os.getenv("RISK_API_URL",    "http://risk-service:8001")
BOT           = os.getenv("BOT_NAME",        "BankBot")
APP_ENV       = os.getenv("APP_ENV",         "dev")
LOG_DIR       = os.getenv("LOG_DIR",         "/data/logs")
MODEL_DIR     = os.getenv("MODEL_DIR",       "/data/models")
OLLAMA_URL    = os.getenv("OLLAMA_URL",      "http://ollama-service:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL",    "llama3.2")
CHROMA_HOST   = os.getenv("CHROMA_HOST",     "chromadb-service")
CHROMA_PORT   = int(os.getenv("CHROMA_PORT", "8000"))
JWT_SECRET    = os.getenv("JWT_SECRET",      "changeme-secret-key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM",   "HS256")
JWT_EXPIRE    = int(os.getenv("JWT_EXPIRE",  "3600"))

# ─── Users from Secret ────────────────────────────────────────────────────────
USERS_JSON = os.getenv("USERS_JSON", '{"admin":"admin123","user":"user123"}')
USERS      = json.loads(USERS_JSON)

pwd_ctx    = CryptContext(schemes=["bcrypt"], deprecated="auto")
security   = HTTPBearer(auto_error=False)

# ─── ChromaDB Setup ───────────────────────────────────────────────────────────
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    emb_fn        = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    knowledge_col = chroma_client.get_or_create_collection(
        name="banking_knowledge",
        embedding_function=emb_fn
    )
    RAG_AVAILABLE = True
    print("ChromaDB connected successfully")
except Exception as e:
    RAG_AVAILABLE = False
    knowledge_col = None
    print(f"ChromaDB not available: {e}")

# ─── Load memory ──────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

try:
    conversation_memory = json.load(open(f"{MODEL_DIR}/memory.json"))
    print(f"Loaded {len(conversation_memory)} memory entries")
except:
    conversation_memory = []

# ─── FIX 1: Expanded Banking Keywords (added Indian banking terms) ────────────
BANKING_KEYWORDS = [
    # Core banking
    "account", "balance", "card", "loan", "transfer", "payment",
    "fraud", "stolen", "lost", "transaction", "bank", "credit",
    "debit", "atm", "interest", "mortgage", "deposit", "withdrawal",
    "unauthorized", "blocked", "statement", "pin", "limit", "fee",
    "cheque", "swift", "iban", "overdraft", "savings", "current",
    "penalty", "closure", "kyc", "nominee", "passbook", "ifsc",
    # Indian banking terms (NEW)
    "upi", "neft", "rtgs", "imps", "nri", "emi", "fd", "rd",
    "fixed deposit", "recurring deposit", "rupay", "netbanking",
    "net banking", "mobile banking", "aadhaar", "pan", "cibil",
    "nbfc", "rbi", "sbi", "hdfc", "icici", "axis", "kotak",
    "gpay", "phonepe", "paytm", "bhim", "vpa", "bank account",
    "locker", "remittance", "forex", "insurance", "mutual fund",
    "cheque book", "demand draft", "dd", "noc", "lien", "pledge",
]

def is_banking_related(message: str) -> bool:
    msg_lower = message.lower()
    return any(k in msg_lower for k in BANKING_KEYWORDS)

# ─── Intent Detection ─────────────────────────────────────────────────────────
def detect_intent(message: str) -> str:
    msg = message.lower()
    if "balance" in msg:                                 return "balance_check"
    if "lost" in msg and "card" in msg:                  return "lost_card"
    if "card" in msg:                                    return "card_issue"
    if "loan" in msg or "mortgage" in msg or "emi" in msg: return "loan_query"
    if "fraud" in msg or "stolen" in msg:                return "fraud_alert"
    if "transfer" in msg or "payment" in msg:            return "payment_query"
    if "upi" in msg or "neft" in msg or "rtgs" in msg or "imps" in msg:
                                                         return "payment_query"
    if "unauthorized" in msg:                            return "fraud_alert"
    if "block" in msg or "locked" in msg:                return "account_blocked"
    if "statement" in msg:                               return "statement_request"
    if "interest" in msg or "rate" in msg:               return "interest_query"
    if "deposit" in msg or "withdrawal" in msg or "fd" in msg or "rd" in msg:
                                                         return "transaction_query"
    if "atm" in msg:                                     return "atm_query"
    if "penalty" in msg or "closure" in msg:             return "loan_closure"
    if "kyc" in msg or "aadhaar" in msg or "pan" in msg: return "kyc_query"
    if "savings" in msg or "account" in msg:             return "account_info"
    return "general_banking"

# ─── Auth Functions ───────────────────────────────────────────────────────────
def create_token(username: str) -> str:
    payload = {
        "sub":  username,
        "exp":  datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRE),
        "iat":  datetime.datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload  = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─── RAG Search ───────────────────────────────────────────────────────────────
def rag_search(query: str, n_results: int = 3) -> str:
    if not RAG_AVAILABLE or not knowledge_col:
        return ""
    try:
        count = knowledge_col.count()
        if count == 0:
            return ""
        results = knowledge_col.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )
        docs = results.get("documents", [[]])[0]
        if docs:
            return "\n\n".join([f"[Policy]: {d}" for d in docs])
    except Exception as e:
        print(f"RAG search error: {e}")
    return ""

# ─── Ollama Call with RAG Context ─────────────────────────────────────────────
def call_ollama(message: str, intent: str, risk_flag: str, context: str = "") -> str:
    rag_section = f"\n\nRelevant banking policies:\n{context}" if context else ""
    system_prompt = f"""You are BankBot, a professional banking support AI assistant.
You ONLY answer questions related to banking and financial services.
If not banking related, politely decline.

Customer intent: {intent}
Risk level: {risk_flag}
{rag_section}

Rules:
- Use the policy context above when available to give accurate answers
- If risk is high_risk, escalate immediately
- Never share passwords or sensitive account details
- Keep responses concise and professional
- Always recommend official channels for sensitive matters"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": f"{system_prompt}\n\nCustomer: {message}\nBankBot:",
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
    return None

# ─── Fallback Replies ─────────────────────────────────────────────────────────
def fallback_reply(intent: str, risk_flag: str) -> str:
    if risk_flag == "high_risk":
        return " High-risk activity detected. Escalated to fraud team. Please call our 24/7 helpline immediately."
    responses = {
        "balance_check":     "Your balance request is being processed. Use our secure app or visit a branch.",
        "lost_card":         "We'll block your card immediately and arrange a replacement within 3-5 business days.",
        "card_issue":        "Our card support team is reviewing your issue. Resolution within 24 hours.",
        "loan_query":        "Loan advisors available Mon-Fri, 9AM-6PM. Shall I arrange a callback?",
        "fraud_alert":       "Fraud alert registered. Our security team will contact you within 1 hour.",
        "payment_query":     "Payment queries handled by 24/7 support. We'll review your transaction history.",
        "account_blocked":   "Account blocks reviewed immediately. Please verify your identity at any branch.",
        "statement_request": "Statements available in our mobile app or request at a branch.",
        "interest_query":    "Current rates available on our website or speak to an advisor.",
        "transaction_query": "Recent transactions viewable in our mobile app.",
        "atm_query":         "ATM locator available on our website and mobile app.",
        "loan_closure":      "Early loan closure penalties depend on your loan type. Contact us for details.",
        "kyc_query":         "KYC updates can be done at any branch or through our mobile app.",
        "account_info":      "A savings account earns interest on your deposits. Visit us or use our app to open one.",
        "general_banking":   "Our banking support team is here to help. How can I assist you?"
    }
    return responses.get(intent, "Our support team will assist you shortly.")

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/login")
def login(username: str, password: str):
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(username)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "username":     username,
        "expires_in":   JWT_EXPIRE
    }

@app.get("/chat")
def chat(message: str = "hello", username: str = Depends(verify_token)):
    timestamp = str(datetime.datetime.now())
    intent    = detect_intent(message)

    if not is_banking_related(message):
        return {
            "bot":           BOT,
            "reply":         "I'm BankBot, your banking assistant. I can help with accounts, cards, loans, transfers, UPI, and more. How can I assist you?",
            "intent":        "out_of_scope",
            "risk_score":    0,
            "risk_flag":     "none",
            "model_version": "N/A",
            "memory_size":   len(conversation_memory),
            "environment":   APP_ENV,
            "served_by":     socket.gethostname(),
            "llm_used":      False,
            "rag_used":      False,
            "user":          username
        }

    # Load model metadata
    model_ver = "not-loaded"
    try:
        model     = json.load(open(f"{MODEL_DIR}/risk_model.json"))
        model_ver = model.get("model_version", "unknown")
    except:
        pass

    # ─── FIX 2: Risk API fallback uses 'low_risk' instead of 'unknown' ────────
    try:
        risk = requests.get(
            f"{RISK_URL}/risk",
            params={"message": message},
            timeout=3
        ).json()
        if "flag" not in risk:
            raise ValueError("Missing flag in risk response")
    except Exception as e:
        print(f"Risk API error: {e}")
        risk = {"risk_score": 0.1, "flag": "low_risk"}   # FIX: was 'unknown'

    risk_flag = risk.get("flag", "low_risk")

    # RAG search
    rag_context = rag_search(message)
    rag_used    = bool(rag_context)

    # Ollama with context
    llm_reply = call_ollama(message, intent, risk_flag, rag_context)
    reply     = llm_reply if llm_reply else fallback_reply(intent, risk_flag)
    llm_used  = llm_reply is not None

    # Memory
    conversation_memory.append({
        "time":    timestamp,
        "user":    username,
        "message": message,
        "intent":  intent,
        "risk":    risk_flag,
        "reply":   reply[:100]
    })

    try:
        with open(f"{MODEL_DIR}/memory.json", "w") as f:
            json.dump(conversation_memory[-100:], f)
    except:
        pass

    try:
        log = f"{timestamp} | user={username} | intent={intent} | risk={risk_flag} | rag={rag_used} | llm={llm_used} | msg={message}\n"
        open(f"{LOG_DIR}/chat.log", "a").write(log)
    except:
        pass

    return {
        "bot":           BOT,
        "reply":         reply,
        "intent":        intent,
        "risk_score":    risk.get("risk_score"),
        "risk_flag":     risk_flag,
        "model_version": model_ver,
        "memory_size":   len(conversation_memory),
        "environment":   APP_ENV,
        "served_by":     socket.gethostname(),
        "llm_used":      llm_used,
        "rag_used":      rag_used,
        "user":          username
    }

# ─── FIX 3 + 4: Knowledge Upload — fixed duplicate add + correct chunk storage ─
@app.post("/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    username: str = Depends(verify_token)
):
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChromaDB not available")

    content = await file.read()
    text    = ""

    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    else:
        text = content.decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content found in file")

    # Chunk text into 500 char pieces
    chunks = [text[i:i+500] for i in range(0, len(text), 500) if text[i:i+500].strip()]

    if not chunks:
        raise HTTPException(status_code=400, detail="No content chunks generated")

    ts = datetime.datetime.now().timestamp()

    # FIX: was adding `ids` as documents (bug). Now only add actual text chunks once.
    ids = [f"{file.filename}_{i}_{ts}" for i in range(len(chunks))]
    knowledge_col.add(
        documents=chunks,   # actual text content
        ids=ids
    )

    try:
        log = f"{datetime.datetime.now()} | KNOWLEDGE UPLOAD | user={username} | file={file.filename} | chunks={len(chunks)}\n"
        open(f"{LOG_DIR}/knowledge.log", "a").write(log)
    except:
        pass

    return {
        "status":      "uploaded",
        "file":        file.filename,
        "chunks":      len(chunks),
        "uploaded_by": username
    }

@app.get("/knowledge/search")
def search_knowledge(query: str, username: str = Depends(verify_token)):
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    context = rag_search(query, n_results=5)
    return {
        "query":   query,
        "results": context,
        "found":   bool(context)
    }

@app.get("/knowledge/stats")
def knowledge_stats(username: str = Depends(verify_token)):
    if not RAG_AVAILABLE:
        return {"status": "ChromaDB not available", "total_documents": 0}
    try:
        count = knowledge_col.count()
        return {"total_documents": count, "status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "bot":           BOT,
        "env":           APP_ENV,
        "rag_available": RAG_AVAILABLE
    }

@app.get("/ready")
def ready():
    try:
        r = requests.get(f"{RISK_URL}/health", timeout=2)
        if r.status_code == 200:
            return {"ready": True, "risk_api": "reachable"}
    except:
        pass
    return {"ready": False, "risk_api": "unreachable"}

@app.get("/memory")
def memory(username: str = Depends(verify_token)):
    user_memory = [m for m in conversation_memory if m.get("user") == username]
    return {
        "total":    len(user_memory),
        "last_10":  user_memory[-10:],
        "username": username
    }

@app.get("/config")
def config(username: str = Depends(verify_token)):
    return {
        "bot":          BOT,
        "env":          APP_ENV,
        "risk_url":     RISK_URL,
        "ollama_url":   OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "rag_enabled":  RAG_AVAILABLE,
        "log_dir":      LOG_DIR,
        "model_dir":    MODEL_DIR,
        "chroma_host":  CHROMA_HOST,
        "chroma_port":  CHROMA_PORT
    }

# ─── NEW: ChromaDB debug endpoint ─────────────────────────────────────────────
@app.get("/knowledge/debug")
def knowledge_debug(username: str = Depends(verify_token)):
    """Returns raw ChromaDB state for debugging"""
    if not RAG_AVAILABLE:
        return {"rag_available": False, "chroma_host": CHROMA_HOST, "chroma_port": CHROMA_PORT}
    try:
        count = knowledge_col.count()
        sample = knowledge_col.peek(limit=3) if count > 0 else {}
        return {
            "rag_available": True,
            "chroma_host":   CHROMA_HOST,
            "chroma_port":   CHROMA_PORT,
            "doc_count":     count,
            "sample_docs":   sample.get("documents", [])[:3] if sample else []
        }
    except Exception as e:
        return {"rag_available": True, "error": str(e)}
