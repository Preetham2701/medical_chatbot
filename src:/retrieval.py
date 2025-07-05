import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, embeddings_path, csv_path):
        self.df = pd.read_csv(csv_path)
        self.sentences = self.df['Sentence'].tolist()
        self.embeddings = np.load(embeddings_path)
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
        self.doc_id_to_text = {i: text for i, text in enumerate(self.sentences)}
        self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def get_relevant(self, query, top_k=1):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.doc_id_to_text[idx] for idx in indices[0]]
