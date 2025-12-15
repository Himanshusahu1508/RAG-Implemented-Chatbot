import streamlit as st
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re

#Adding memory seeing the app performance for 3 turns
if "history" not in st.session_state:
    st.session_state.history = []



index = faiss.read_index("faiss_index.bin")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


genai.configure(api_key="AIzaSyDi164J3DwvtxKA1UXAjVQU-QuAxa7nCgU")
model = genai.GenerativeModel("models/gemini-2.5-flash-lite")



def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    retrieved_docs = [texts[i] for i in indices[0]]
    return "\n".join(retrieved_docs)

def generate_response(query):
    # Step 1️: Retrieve RCA context
    context = retrieve_context(query)
    context = re.sub(r'[^\x00-\x7F]+',' ', context)
    context = re.sub(r'\s+', ' ', context).strip()
    context = context[:3500]  # avoid token overflow

    # Step 2️: Add short memory (last 3 turns)
    conversation_history = "\n".join([
        f"User: {m['user']}\nAssistant: {m['bot']}"
        for m in st.session_state.history[-3:]
    ])

    # Step 3️: Build final prompt
    prompt = f"""
    You are a customer support RCA assistant.
    You explain customer complaint trends clearly and concisely.

    Conversation so far:
    {conversation_history}

    RCA Context:
    {context}

    Current Question: {query}

    Keep the tone analytical and professional.
    """

    # Generate response from Gemini
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        answer = response.text.strip() if response and response.text else "No response generated."
    except Exception as e:
        answer = f"Gemini API Error: {str(e)}"

    #Save this exchange in memory
    st.session_state.history.append({"user": query, "bot": answer})

    return answer




st.title("Customer Support RCA Assistant")

st.markdown("Ask about delivery delays, refunds, or app crashes. The bot remembers your last 3 questions!")

# Previous conversation
for message in st.session_state.history:
    with st.chat_message("user"):
        st.write(message["user"])
    with st.chat_message("assistant"):
        st.write(message["bot"])

# Chat input box
user_query = st.chat_input("Ask your question about complaints...")
if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    with st.spinner("Analyzing RCA context..."):
        response = generate_response(user_query)
    with st.chat_message("assistant"):
        st.write(response)

