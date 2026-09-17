"""Retrieval layer.

Implements a local, self-hosted retrieval mechanism over the structured
knowledge base using TF-IDF + cosine similarity (scikit-learn). This keeps
the system fully self-hosted with no external embedding API calls or
network dependency, in line with the project's "no external paid
dependencies" constraint, while still giving a genuine retrieve-then-reason
pipeline (as opposed to stuffing the whole KB into a prompt).

Swap-in note: `KnowledgeRetriever` exposes the same `.query()` interface a
vector-DB-backed retriever (e.g. Chroma/FAISS + sentence-transformers)
would, so the TF-IDF backend can be replaced without touching the
reasoning/conversation layers.
"""
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge_base import knowledge_base


class KnowledgeRetriever:
    def __init__(self, kb=knowledge_base):
        self.kb = kb
        self.entries = kb.all()
        self.corpus = [kb.searchable_text(e) for e in self.entries]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Return (entry, score) pairs ranked by cosine similarity to query_text."""
        if not query_text.strip():
            return []
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [(self.entries[i], float(sims[i])) for i in ranked_idx if sims[i] > 0]


retriever = KnowledgeRetriever()
