from dotenv import load_dotenv

load_dotenv()  # harus sebelum import txtai

from txtai import Embeddings

data = [
    "Python adalah bahasa pemrograman populer untuk AI",
    "FAISS adalah library vector search dari Meta",
    "Nasi goreng adalah makanan khas Indonesia",
    "Transformer mengubah NLP lewat mekanisme attention",
    "Rendang berasal dari Minangkabau",
    "HNSW adalah algoritma graph untuk approximate nearest neighbor",
]

emb = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True)
emb.index(data)

print(emb.search("makanan", 2))
print(emb.search("select text, score from txtai where similar('vector database') and score >= 0.2"))

emb.save("learning/idx"); emb.load("learning/idx")
print(emb.count())
