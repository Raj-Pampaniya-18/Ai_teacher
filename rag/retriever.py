import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rag/teacher.index")

def retrieve(query, docs, k=5):
    q_emb = model.encode([query])
    distances, indices = index.search(q_emb, k)

    context = []
    scores = []

    for i, d in zip(indices[0], distances[0]):
        context.append(docs[i])
        scores.append(d)

    avg_distance = sum(scores) / len(scores)
    score = round(1 / (1 + avg_distance), 2)

    return "\n".join(context), score
