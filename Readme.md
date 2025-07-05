Medical QA Chatbot

This project is a medical question-answering chatbot that combines local knowledge with web search and uses an LLM to provide concise, medically safe first-aid responses. It uses Streamlit for the user interface, FAISS for local semantic search, and Ollama for local large language model generation.

Features
Loads local knowledge base from a CSV  
Uses precomputed sentence embeddings for fast similarity search  
Integrates Serper API to fetch the top 3 Google results  
Generates answers using a local LLM via Ollama  
Provides clear bullet-point first-aid steps  
Includes a Streamlit app for easy interaction



# Setup

1. Install dependencies:  
   pip install -r requirements.txt

2. Usage
1.The chatbot will answer medical first-aid questions based on its local database and web evidence.
2.It will always include a disclaimer.
3.It will present first-aid steps clearly as bullet points.

3. Design Trade-offs
1.FAISS is used for local fast semantic retrieval, trading off deep generative retrieval for speed.
2.Ollama local LLM ensures privacy but might require enough hardware resources.
3.Serper is used for up-to-date web data without storing large amounts of scraped text locally.

4. To run the streamlit: 
streamlit run /Users/preethams/Documents/medical_chatbot/app.py

