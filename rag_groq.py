from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="sochai_chroma_db")
collection = chroma_client.get_or_create_collection("sochai_papers")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Retrieval ---
def retrieve_context(question: str, k: int = 3):
    q_emb = embed_model.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return res.get("documents", [[]])[0] if res else []

SYSTEM_PROMPT = """You are SochAI, an assistant for answering questions from research documents.
Answer ONLY using the provided context. If the answer is not in the context,
respond with: "I couldn't find that in the document."
Be concise and precise.
"""
def build_prompt(context_chunks, question):
    joined = "\n\n----\n\n".join(context_chunks)
    return f"{SYSTEM_PROMPT}\n\nContext:\n<<<\n{joined}\n>>>\n\nQuestion: {question}\nAnswer:"

# --- Answer using Groq LLM ---
def answer(question: str, k: int = 3):
    ctx = retrieve_context(question, k)
    if not ctx:
        return "I couldn't find that in the document."

    prompt = build_prompt(ctx, question)

    response = client.chat.completions.create(
        model="llama3-8b-8192",  # free Groq-hosted Llama3
        messages=[
            {"role": "system", "content": "You are SochAI, a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# --- Interactive loop ---
if __name__ == "__main__":
    print("SochAI (Groq-powered) — type your question (or 'exit'):\n")
    while True:
        q = input("❓ Your question: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Bye! 👋")
            break
        ans = answer(q, k=3)
        print("\n🧠 Answer:", ans, "\n")
