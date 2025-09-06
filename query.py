from sentence_transformers import SentenceTransformer
import chromadb

# Connect to persistent DB
chroma_client = chromadb.PersistentClient(path="sochai_chroma_db")

# Load collection
collection = chroma_client.get_or_create_collection("sochai_papers")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Query text
query = "What is the main idea of the speech?"
query_embedding = model.encode(query).tolist()

# Query Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("=== Raw Results ===")
print(results)   # 👈 debug

print("\n=== Top Matches ===")
if results["documents"] and results["documents"][0]:
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i+1}. {doc[:300]}...\n")
else:
    print("⚠️ No matching documents found in collection.")
