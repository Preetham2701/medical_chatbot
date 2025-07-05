import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pytest

# load your local data
@pytest.fixture
def local_data():
    df = pd.read_csv("/Users/preethams/Documents/medical_chatbot/assignment/Assignment Data Base.csv")
    sentences = df['Sentence'].tolist()
    embeddings = np.load("/Users/preethams/Documents/medical_chatbot/assignment/medical_embeddings.npy")
    return sentences, embeddings

def test_local_retrieval(local_data):
    sentences, embeddings = local_data
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query = "What should I do if I feel shaky and my sugar is 55?"
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    
    distances, indices = index.search(query_embedding, 1)
    retrieved = sentences[indices[0][0]]
    
    assert isinstance(retrieved, str)
    assert len(retrieved) > 5  

def test_web_search():
    import requests
    SERPER_API_KEY = "ef022dcc4d28b08e2b7c5b1e0a0609f66f50c703"
    query = "hypoglycemia first aid"
    
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = {"q": query}
    response = requests.post(url, headers=headers, json=data)
    results = response.json()
    
    organic = results.get("organic", [])[:3]
    assert isinstance(organic, list)
    assert len(organic) >= 1


