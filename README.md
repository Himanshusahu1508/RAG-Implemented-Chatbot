# AI-driven Root Cause Analysis (RCA) Chatbot

This repository contains an end-to-end Generative AI application that automates root-cause analysis on Zomato’s Play Store reviews.  
It combines data analytics, semantic embeddings, and a Retrieval-Augmented Generation (RAG) pipeline to surface and explain key customer pain points.  
The chatbot retrieves relevant reviews from a FAISS vector index and uses Google Gemini to generate concise, contextual RCA summaries through a Streamlit interface.

---

## Project Overview

Customer reviews contain valuable signals about product performance and user experience, but analyzing thousands of reviews manually is time-consuming.  

This project automates that process through the following steps:
1. Scrape Play Store reviews of the Zomato app using the Google Play Scraper API.  
2. Clean and preprocess the text to remove noise and stopwords.  
3. Perform manual root cause analysis (RCA) to identify key complaint categories such as delivery issues, payment failures, and app bugs.  
4. Build visual analytics to explore complaint trends by app version, rating, and time.  
5. Create a knowledge base of categorized complaints and generate vector embeddings using the SentenceTransformers model.  
6. Store and retrieve relevant reviews using FAISS (Facebook AI Similarity Search).  
7. Integrate Google Gemini to generate human-like analytical summaries based on retrieved context.  
8. Deploy an interactive Streamlit chatbot that allows users to ask RCA questions and receive contextual insights in real time.  

---

## Key Features

- Scrapes real Play Store reviews from the Zomato app.  
- Cleans and categorizes 10,000+ complaints into data-driven categories.  
- Generates embeddings for semantic similarity search.  
- Uses FAISS to retrieve contextually relevant complaints.  
- Integrates Google Gemini for RCA-style response generation.  
- Includes evaluation using Recall@3 for retrieval accuracy.  
- Provides an interactive Streamlit chatbot interface with short-term memory.  

---

## Project Workflow

1. **Data Collection:** Extract reviews using `google-play-scraper`.  
2. **Data Cleaning:** Remove noise, URLs, and stopwords using regex and NLTK.  
3. **Manual RCA:** Identify and tag complaint types like "Late Delivery" or "Payment Issues".  
4. **EDA and Visualization:** Create trend plots, category distributions, and word clouds.  
5. **Knowledge Base Generation:** Merge cleaned reviews, categories, and app versions into a structured dataset.  
6. **Embedding and Indexing:** Encode text using `SentenceTransformers` and build a FAISS index for retrieval.  
7. **Retrieval-Augmented Generation:** Retrieve relevant context and generate explanations using Gemini 2.5 Pro.  
8. **Evaluation:** Compute Recall@3 to validate the quality of retrieval.  
9. **Deployment:** Use Streamlit for a conversational chatbot interface that automates RCA.  

---

## Tech Stack

| Layer | Tools & Libraries |
|-------|-------------------|
| Data Collection | `google-play-scraper`, `pandas` |
| Cleaning & Preprocessing | `re`, `nltk`, `stopwords` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| NLP & Embeddings | `sentence-transformers` |
| Retrieval | `faiss-cpu` |
| LLM Integration | `google-generativeai` (Gemini 2.5 Pro) |
| Application | `streamlit` |
| Evaluation | `scikit-learn`, custom Recall@3 metric |
| Storage | `pickle`, `csv` |

---

## Sample Insights

During manual RCA, the following insights were discovered:

1. “Late Delivery” was the most frequent complaint, especially in app version 17.0.  
2. “Payment/Refund Issues” spiked during July 2024 across versions 16.8 and 17.1.  
3. “App / Technical Issues” correlated with version updates that introduced new UI features.  

These findings formed the basis of the chatbot’s knowledge base for automated RCA.  

---

## Example Chatbot Interaction

**User:** Why are users complaining about late deliveries?  
**Bot:** Many users reported delayed delivery during lunch and dinner peak hours, especially in version 17.0. Complaints indicate poor ETA predictions and slower partner assignment times.  

**User:** What about payment issues?  
**Bot:** Most refund and payment complaints were raised in version 16.8 due to transaction failures and delayed wallet credits.  

---

## Evaluation Metric

To test retrieval accuracy, a small ground-truth mapping of complaint categories was created, and a **Recall@3** score was calculated.  

This measured how often the correct complaint category appeared among the top 3 retrieved results.  
Example result:  
`Recall@3 = 0.67` (indicating 67% correct retrievals within top 3).

---

## Summary
This project demonstrates how Generative AI can be used for real-world customer analytics.
By combining manual RCA, embeddings, FAISS retrieval, and Gemini-based summarization, the system transforms unstructured reviews into structured insights that help product and CX teams act faster.

It bridges data analytics and GenAI turning raw feedback into explainable, actionable insights.

## Architecture Overview

The following diagram shows the end-to-end flow of the RCA chatbot:

                 ┌────────────────────────────┐
                 │  Play Store Reviews (Zomato) │
                 └──────────────┬───────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │ Data Cleaning & EDA │
                     │  (pandas, nltk, re) │
                     └────────────┬─────────┘
                                │
                                ▼
                 ┌────────────────────────────────┐
                 │ Root Cause Categorization (RCA) │
                 │   Keyword rules + manual tags   │
                 └────────────┬────────────────────┘
                                │
                                ▼
                 ┌────────────────────────────────┐
                 │ Knowledge Base Creation         │
                 │ (category, review, appVersion)  │
                 └────────────┬────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────┐
            │ SentenceTransformer Embeddings          │
            │ (multi-qa-MiniLM-L6-cos-v1)             │
            └────────────┬────────────────────────────┘
                                │
                                ▼
                       ┌─────────────────────┐
                       │ FAISS Vector Index  │
                       │ (semantic retrieval)│
                       └────────────┬────────┘
                                │
                                ▼
          ┌──────────────────────────────────────────┐
          │ User Query → Retrieve Top-K Contexts     │
          │ (similar complaint reviews)               │
          └────────────┬──────────────────────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │ Google Gemini 2.5 Pro (LLM)         │
             │ Generates RCA Explanation            │
             └────────────┬─────────────────────────┘
                                │
                                ▼
                  ┌───────────────────────────────┐
                  │ Streamlit Chat Interface       │
                  │ (3-turn memory, RCA insights)  │
                  └───────────────────────────────┘



