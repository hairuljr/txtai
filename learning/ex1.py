from dotenv import load_dotenv

load_dotenv()  # harus sebelum import txtai

from txtai import Embeddings

data = [
    "Python is a famous AI programming language",
    "FAISS is a library vector search from Meta",
    "Nasi goreng is a famous Indonesian dish",
    "Transformer changes NLP through the attention mechanism",
    "Rendang is a traditional Minangkabau dish",
    "HNSW is a graph-based algorithm for approximate nearest neighbor search",
]

emb = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True)
emb.index(data)

print(emb.search("makanan", 2))
print(emb.search("indonesian dish", 1))
print(emb.search("select text, score from txtai where similar('vector database') and score >= 0.2"))

emb.save("learning/idx"); emb.load("learning/idx")
print(emb.count())
