import re
import mysql.connector
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
from langchain_community.embeddings import FakeEmbeddings
import os
from datetime import datetime

# ---------------- Load FAISS Vector Store ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_PATH = os.path.join(BASE_DIR, "..", "vectorstore")

# 👉 embeddings fake pour éviter torch
embeddings = FakeEmbeddings(size=384)

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# ---------------- LLM (LLAMA CPP) ----------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "tinyllama.gguf")

print("Loading model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=2
)

print("Model loaded!")

# ---------------- RAG FUNCTION ----------------
def generate_answer(query, context):
    prompt = f"""
    You are an industrial maintenance expert.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and professionally:
    """
    output = llm(prompt, max_tokens=200)

    return output["choices"][0]["text"].strip()


def rag_query(query):

    docs = retriever.invoke(query)

    # On limite à 2 passages
    docs = docs[:2]

    context = "\n".join(
        [doc.page_content[:500] for doc in docs]
    )

    answer = generate_answer(
        query,
        context
    )

    return answer, docs

# ---------------- MYSQL CONNECTION ----------------
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database=""
    )

# ---------------- DATA RETRIEVAL ----------------
def get_readings(machine_id: int, n: int = None, start_date=None, end_date=None):
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM machine_sensor_data WHERE machine_id = %s"
    params = [machine_id]

    if start_date and end_date:
        query += " AND timestamp BETWEEN %s AND %s"
        params.extend([start_date, end_date])

    query += " ORDER BY timestamp DESC"
    if n:
        query += " LIMIT %s"
        params.append(n)

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    return rows[::-1]

# ---------------- SIMPLE TREND ----------------
def compute_trend(values):
    if len(values) < 2:
        return "stable", 0
    slope = (values[-1] - values[0]) / len(values)
    if abs(slope) < 0.01:
        return "stable", slope
    elif slope > 0:
        return "increasing", slope
    else:
        return "decreasing", slope

# ---------------- CLASSIFICATION ----------------
def classify_query(query: str):
    q = query.lower()
    if "predict" in q or "failure" in q:
        return "prediction"
    if "compare" in q:
        return "comparison"
    if "maintenance" in q or "manual" in q:
        return "knowledge"
    return "general"

# ---------------- SIMPLE FAILURE PREDICTION ----------------
def predict_failure(values, metric="vibration", threshold=None):

    thresholds = {
        "temperature": 120,
        "vibration": 30,
        "pressure": 10
    }

    if threshold is None:
        threshold = thresholds.get(metric, 100)

    if len(values) < 2:
        return f"Not enough data for {metric} prediction."

    latest = values[-1]

    # Simple trend
    slope = (values[-1] - values[0]) / len(values)

    # Already critical
    if latest >= threshold:
        return (
            f"⚠️ {metric.capitalize()} already exceeded "
            f"critical threshold ({threshold})."
        )

    # Stable
    if slope <= 0:
        return (
            f"✅ {metric.capitalize()} stable. "
            f"No immediate failure risk detected."
        )

    # Estimate readings until threshold
    remaining = (threshold - latest) / slope

    return (
        f"⚠️ Estimated {metric} threshold reach "
        f"in approximately {int(remaining)} readings."
    )


# ---------------- MACHINE COMPARISON ----------------
def compare_machines():

    results = []

    for machine_id in [1, 2, 3]:

        rows = get_readings(machine_id, n=20)

        if not rows:
            continue

        avg_vibration = sum(
            r["vibration"] for r in rows
        ) / len(rows)

        results.append(
            (machine_id, avg_vibration)
        )

    if not results:
        return "⚠️ No machine data available."

    worst_machine = max(
        results,
        key=lambda x: x[1]
    )

    return (
        f"⚠️ Machine {worst_machine[0]} "
        f"shows the highest average vibration "
        f"({worst_machine[1]:.2f} mm/s)."
    )

# ---------------- MAIN RESPONSE ----------------
def get_rag_response(query: str, machine_id: int = None):

    try:
        # 👉 Force toutes les requêtes à passer par le RAG
        answer, docs = rag_query(query)

        # 👉 Extraction des sources
        sources = [
            d.metadata.get("source", "PDF")
            for d in docs
        ]

        return {
            "answer": answer,
            "alerts": [],
            "sources": sources
        }

    except Exception as e:

        # 👉 fallback temporaire pour debug
        return {
            "answer": f"⚠️ RAG error: {str(e)}",
            "alerts": [],
            "sources": []
        }

def get_ml_context(
    failure_probability=None
):

    if failure_probability is None:
        return ""

    return f"""
Current machine failure probability:
{failure_probability:.1%}

Use this prediction in your answer.
"""