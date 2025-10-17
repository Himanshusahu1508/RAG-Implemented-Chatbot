import streamlit as st
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re


index = faiss.read_index("faiss_index.bin")
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


genai.configure(api_key="AIzaSyDx0CbhRG_cuKyU7_rzejfLmaG-C2ZQr8s")
model = genai.GenerativeModel("models/gemini-2.5-pro")


def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), top_k)
    retrieved_docs = [texts[i] for i in indices[0]]
    return "\n".join(retrieved_docs)

def generate_response(query):
    context = retrieve_context(query)
    context = re.sub(r'[^\x00-\x7F]+',' ', context)
    context = re.sub(r'\s+', ' ', context).strip()
    context = context[:3500]  # avoid Gemini token overflow

    prompt = f"""
    You are a customer support assistant.
    Use the following RCA context to explain the user's issue briefly and professionally.

    RCA context:
    {context}

    Question: {query}
    """

    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(prompt)
        return response.text if response and response.text else "No response generated."
    except Exception as e:
        return f"Gemini API Error: {str(e)}"



st.title("Customer Support RCA Assistant")
st.write("Ask about delivery delays, refunds, app issues, or support trends.")

query = st.text_input("Enter your question:")
if st.button("Get Response"):
    with st.spinner("Analyzing complaints..."):
        answer = generate_response(query)
    st.subheader("Response:")
    st.write(answer)
