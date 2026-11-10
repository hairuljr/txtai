from dotenv import load_dotenv

load_dotenv()

import time

from txtai import Embeddings

data = [
    "Python adalah bahasa pemrograman populer untuk AI",
    "FAISS adalah library vector search dari Meta",
    "Nasi goreng adalah makanan khas Indonesia",
    "Transformer mengubah NLP lewat mekanisme attention",
    "Rendang berasal dari Minangkabau",
    "HNSW adalah algoritma graph untuk approximate nearest neighbor",
]

for backend in ["faiss", "hnsw", "numpy", "torch"]:
    emb = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True, backend=backend)
    emb.index(data)
    start = time.perf_counter()
    result = emb.search("makanan", 2)
    ms = (time.perf_counter() - start) * 1000
    print(f"{backend:6} {type(emb.ann).__name__:8} {ms:6.1f}ms", [(r["text"][:20], round(r["score"], 3)) for r in result])
