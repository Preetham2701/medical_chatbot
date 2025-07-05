import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import ollama

# 1. Load local knowledge base
df = pd.read_csv("/Users/preethams/Documents/medical_chatbot/assignment/Assignment Data Base.csv")
sentences = df['Sentence'].tolist()

# 2. Load precomputed embeddings
embeddings = np.load("/Users/preethams/Documents/medical_chatbot/assignment/medical_embeddings.npy")

# 3. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

doc_id_to_text = {i: text for i, text in enumerate(sentences)}

# 4. Serper web search
SERPER_API_KEY = "ef022dcc4d28b08e2b7c5b1e0a0609f66f50c703"

def web_search(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"q": query}
    response = requests.post(url, headers=headers, json=data)
    results = response.json()
    organic = results.get("organic", [])[:1]
    evidence = []
    for res in organic:
        title = res.get("title")
        link = res.get("link")
        snippet = res.get("snippet")
        evidence.append(f"{title}: {snippet} (source: {link})")
    return evidence

# 5. Retrieve local docs
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def retrieve_relevant_docs(query, top_k=1):  
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [doc_id_to_text[idx] for idx in indices[0]]

# 6. Answer generation
def ask_medical_question(query):
    retrieved = retrieve_relevant_docs(query)
    web_results = web_search(query)
    system_prompt = (
        "You are a helpful medical safety assistant. Always include disclaimers. "
        "Keep answers under 250 words. "
        "Always list first-aid steps clearly on separate lines as bullet points or numbers, "
        "and do not embed them inside paragraphs."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""⚠️ DISCLAIMER: This information is for educational purposes only and not a substitute for professional medical advice.

The user asked:
'{query}'

Relevant local knowledge:
{chr(10).join(retrieved)}

Relevant web evidence:
{chr(10).join(web_results)}

Please provide a clear, concise, medically appropriate first-aid answer including:
- the likely condition
- first-aid steps
- key medicine(s)
- references to the above sources
in fewer than 250 words.
"""}
    ]

    response = ollama.chat(model="llama2", messages=messages)
    return response['message']['content']

# 7. Streamlit interface
st.title("🩺 Medical Safety Q&A Bot")

user_query = st.text_input("Ask a medical question:")

if user_query:
    with st.spinner("Thinking..."):
        answer = ask_medical_question(user_query)
    st.markdown("### Answer:")
    st.write(answer)
