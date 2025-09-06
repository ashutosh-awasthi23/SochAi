from sentence_transformers import SentenceTransformer
import chromadb
from chunker import chunk_text
from pdf_reader import extract_text_from_pdf

# Persistent storage
chroma_client = chromadb.PersistentClient(path="sochai_chroma_db")
collection = chroma_client.get_or_create_collection(name="sochai_papers")

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text
text = extract_text_from_pdf("Aug_Assembly_Speech.pdf")
print("📄 Extracted text length:", len(text))

# Chunk text
chunks = chunk_text(text)
print("🔹 Number of chunks:", len(chunks))

# Store chunks
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"speech-{i}"]   # 👈 safer unique IDs
    )
    print(f"✅ Added chunk {i}: {chunk[:80]}...")

print("🎉 Stored PDF chunks in Chroma")
