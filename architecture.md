Overview
This medical chatbot combines:
Local knowledge: CSV file with pre-defined medical sentences and first-aid recommendations
Semantic retrieval: FAISS index built on sentence embeddings to find relevant local documents
Web search: Serper API to retrieve top 3 Google results for additional context
Answer generation: Uses a local large language model (Ollama + LLaMA2) to produce a concise first-aid answer
User interface: Streamlit app to interact with the chatbot

Components
app.py : Main Streamlit app logic
Assignment Data Base.csv : Local knowledge base
medical_embeddings.npy : Precomputed embeddings of local knowledge
FAISS : Fast similarity search over embeddings
Serper API : Adds fresh evidence from Google
Ollama : Local LLM to generate final answers
Streamlit : User-facing web app

Workflow
User asks a medical question in Streamlit
The query is embedded and compared against the local FAISS index
Top 1 local document is retrieved
Top 3 web results from Serper are fetched
All evidence is sent to the local LLM (Ollama)
The LLM generates a bullet-pointed first-aid answer with disclaimers
Answer is displayed in Streamlit