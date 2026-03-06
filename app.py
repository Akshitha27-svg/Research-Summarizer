from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# ==========================================
# 🚀 Initialize FastAPI App
# ==========================================
app = FastAPI(title="Hybrid RAG Research Assistant")

# ==========================================
# 📁 Mount PDF Static Folder
# ==========================================
PDF_FOLDER = "pdfs"

if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

app.mount("/pdfs", StaticFiles(directory=PDF_FOLDER), name="pdfs")

# ==========================================
# 🔹 Load Models and Data (Startup)
# ==========================================
print("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("faiss_index.index")

print("Loading metadata...")
metadata = pd.read_csv("metadata.csv")

print("Loading LLM...")
model_name = os.getenv("MODEL_NAME", "google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ Backend Ready!")

# ==========================================
# 📄 Request Model
# ==========================================
class QueryRequest(BaseModel):
    question: str
    selected_paper: str


# ==========================================
# 📚 Get Available Papers (Dynamic Dropdown)
# ==========================================
@app.get("/papers")
def get_papers():
    papers = metadata["paper_name"].unique().tolist()
    return {"papers": papers}


# ==========================================
# 🤖 Ask Endpoint
# ==========================================
@app.post("/ask")
def ask_question(request: QueryRequest):

    query = request.question
    selected_paper = request.selected_paper

    # Filter only selected paper
    paper_chunks = metadata[metadata["paper_name"] == selected_paper]

    if paper_chunks.empty:
        return {"error": "Selected paper not found."}

    chunk_texts = paper_chunks["chunk"].tolist()

    # Encode query
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Encode selected paper chunks
    chunk_embeddings = embedding_model.encode(chunk_texts)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")

    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    chunk_norm = chunk_embeddings / np.linalg.norm(
        chunk_embeddings, axis=1, keepdims=True
    )

    similarities = np.dot(chunk_norm, query_norm.T).flatten()
    max_similarity = float(np.max(similarities))

    # ==========================================
    # 🚫 Out-of-Context Detection
    # ==========================================
    THRESHOLD = 0.45

    if max_similarity < THRESHOLD:
        return {
            "out_of_context": True,
            "message": "This question is out of context for the selected paper."
        }

    # ==========================================
    # 🔎 Retrieve Top Relevant Chunks
    # ==========================================
    top_indices = similarities.argsort()[-5:][::-1]
    retrieved_chunks = [chunk_texts[i] for i in top_indices]
    retrieved_context = "\n\n".join(retrieved_chunks)

    # ==========================================
    # 📝 Prompt for Generation
    # ==========================================
    prompt = f"""
Write a detailed academic explanation in ONE well-structured paragraph
(minimum 200 words) using ONLY the provided context.

Question:
{query}

Context:
{retrieved_context}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            min_new_tokens=200,
            temperature=0.8,
            do_sample=True,
            repetition_penalty=1.2
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ==========================================
    # 📊 Confidence Score (Semantic Alignment)
    # ==========================================
    query_emb = embedding_model.encode(query, convert_to_numpy=True)
    answer_emb = embedding_model.encode(generated_text, convert_to_numpy=True)

    query_emb /= np.linalg.norm(query_emb)
    answer_emb /= np.linalg.norm(answer_emb)

    confidence = round(float(np.dot(query_emb, answer_emb)) * 100, 2)

    return {
        "out_of_context": False,
        "summary": generated_text,
        "confidence": confidence

    }
