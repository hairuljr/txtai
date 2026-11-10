# Latihan txtai (4 langkah, ±1 hari)

## Setup (sekali)
Python 3.15 mungkin belum didukung torch. Pakai 3.10–3.12:

```bash
cd /Users/hairuljr/Codes/txtai
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[pipeline,database,graph]" pytest
```

## Latihan 1: Index, search, SQL (±1 jam)
Simpan sebagai `learning/ex1.py`. Target: paham `index`, `search`, `save/load`, SQL.

```python
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
```
Pertanyaan: file apa saja yang muncul di `learning/idx`? (Ini jawaban notebook 29.) Jangan di-commit: `echo "learning/idx/" >> .git/info/exclude`.

## Latihan 2: Trace dengan debugger (±2 jam)
Pasang breakpoint (`breakpoint()` atau IDE) di titik berikut, jalankan ex1.py:

1. `src/python/txtai/embeddings/base.py` → `index()`
2. `embeddings/index/transform.py` → bagaimana teks jadi vektor (batch, dimensi)
3. `ann/dense/factory.py` → backend ANN apa yang dibuat? (default faiss)
4. `database/sqlite.py` → bagaimana dokumen disimpan
5. `embeddings/base.py` → `search()` → `embeddings/search/base.py`

Catat alurnya dalam 5 baris. Lalu ganti config: `backend="hnsw"`, `"numpy"`, `"torch"` dan bandingkan hasilnya.

## Latihan 3: Baca test (±1 jam)
```bash
pytest test/python/testembeddings.py -k "testsearch or testsave" -x -q
```
Baca `test/python/testembeddings.py`, `testdatabase/`, `testscoring/`. Pilih 3 test, jelaskan apa yang diuji dan kenapa.

## Latihan 4: Mini proyek RAG hybrid (±3 jam)
Simpan sebagai `learning/ex4.py`. Pakai dokumen milik kamu (folder .md/.txt).

```python
from txtai import Embeddings, RAG

emb = Embeddings(hybrid=True, content=True)          # dense + BM25
emb.index(open("catatan.txt").read().split("\n\n"))   # ganti sumbernya

rag = RAG(emb, "Qwen/Qwen2.5-0.5B-Instruct", output="flatten")
print(rag("Ringkas inti dokumen ini"))
```
Tantangan bertahap:
1. Bandingkan `hybrid=False` vs `True` pada 5 query (cek hasil `search`).
2. Tambah reranker: `pipeline/text/reranker.py`, `Reranker(emb, "BAAI/bge-reranker-base")`.
3. Baca `pipeline/llm/rag.py`, ubah template prompt, lihat efeknya.
4. Ukur: latency, dan kualitas jawaban pada 10 pertanyaan buatanmu sendiri.

## Checklist paham
- [ ] Bisa jelaskan alur index → search tanpa melihat kode
- [ ] Tahu bedanya dense, sparse, hybrid, dan kapan hybrid menang
- [ ] Tahu trade-off faiss / hnsw / numpy
- [ ] Bisa jelaskan alur RAG di `rag.py`
- [ ] Punya satu proyek kecil yang jalan
