from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from chunker import chunk_text
from pdf_reader import extract_text_from_pdf


model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client  = chromadb.Client()
collection = chroma_client.create_collection(name = "sochai_papers")
text = extract_text_from_pdf("Aug_Assembly_Speech.pdf")
chunks = chunk_text(text)

for i,chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        documents = [chunk],
        embeddings = [embedding],
        ids = [str(i)]
    )
print("Store PDF chunks in Chroma")
