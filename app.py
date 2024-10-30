import streamlit as st
import pandas as pd
import cohere
from rank_bm25 import BM25Okapi
import re
import logging

# Setup
st.set_page_config(layout="wide")
logging.basicConfig(filename='rerank_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Custom CSS
st.markdown("""
<style>
body {
   background-color: #fafafa;
}

/* Search results */
.results-container {
   display: flex;
   gap: 20px;
   margin-top: 20px;
}

/* BM25 results */
.css-1r6slb0 > div:nth-of-type(1) {
   background-color: #f5f5f5;
   border-radius: 8px;
   padding: 20px;
}

/* Rerank results */
.css-1r6slb0 > div:nth-of-type(2) {
   background-color: #d4d9d4;
   border-radius: 8px;
   padding: 20px;
}

/* Suggested questions buttons */
.stButton button {
   background-color: #f5f4f2;
   color: #5c5a53;
   border: none;
}

/* Response text */
.stMarkdown p {
   color: #696969;
}
            
.result-box {
    background-color: #d4d9d4;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}
            
/* Target BM25 and Rerank results text */
.element-container:nth-of-type(n+1) p {
    color: #696969 !important;
}

/* Target result-box text specifically */
.result-box p {
    color: #696969 !important;
}
</style>
""", unsafe_allow_html=True)

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load data
faqs_banking_df = pd.read_csv('faqs_banking.csv')
enriched_faqs_df = faqs_banking_df.dropna(subset=['question', 'answer']).reset_index(drop=True)
enriched_faqs_df = enriched_faqs_df[enriched_faqs_df['question'] != enriched_faqs_df['answer']]

# Define customer queries
suggested_questions = [
    "My payment thingy is busted—how can I grab a new one?",
    "What ways can I grow my money with you?",
    "Is it possible to set up a bank account for my kid?",
    "What's the easiest way to send cash abroad?",
    "Why should I choose a high-interest savings plan?",
    "How can I qualify for a mortgage if my credit isn't great?"
]

def get_bm25_results(query, faqs, top_n=3):
    tokenized_faqs = [simple_tokenize(faq) for faq in faqs['question']]
    bm25 = BM25Okapi(tokenized_faqs)
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    faqs = faqs.copy()
    faqs['bm25_score'] = scores
    return faqs.nlargest(top_n, 'bm25_score')[['question', 'answer']]

def get_rerank_results(query, faqs, top_n=3):
    co = cohere.Client(st.secrets["cohere"]["api_key"], base_url="https://stg.api.cohere.com")
    rerank_input = [{'text': row['question']} for _, row in faqs.iterrows()]
    
    response = co.rerank(
        query=query,
        documents=rerank_input,
        model="rerank-english-v3.0",
        top_n=top_n
    )
    
    if response.results:
        faqs = faqs.copy()
        for result in response.results:
            faqs.at[result.index, 'rerank_score'] = result.relevance_score
        return faqs.nlargest(top_n, 'rerank_score')[['question', 'answer']]
    return faqs.head(top_n)[['question', 'answer']]

# Initialize session state for the query
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''

# Main UI
st.title("Banking FAQ Search Comparison")

# Search input
query = st.text_input("Enter your question:", value=st.session_state.current_query)

# Suggested questions
st.write("Suggested questions:")
cols = st.columns(2)
for i, question in enumerate(suggested_questions):
    col_idx = i % 2
    if cols[col_idx].button(question, key=f"q_{i}", use_container_width=True):
        st.session_state.current_query = question
        st.rerun()

# Use the query from either direct input or button selection
current_query = query or st.session_state.current_query

# Process search when query is present
if current_query:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### BM25 Results")
        bm25_results = get_bm25_results(current_query, enriched_faqs_df.copy())
        for _, row in bm25_results.iterrows():
            with st.container():
                st.markdown(f"**Q:** {row['question']}")
                st.markdown(f"**A:** {row['answer']}")
                st.divider()
    
    with col2:
        st.markdown("### Cohere Rerank Results")
        rerank_results = get_rerank_results(current_query, enriched_faqs_df.copy())
        for _, row in rerank_results.iterrows():
            st.markdown("""
                <div class="result-box">
                    <p><strong>Q:</strong> {}</p>
                    <p><strong>A:</strong> {}</p>
                </div>
            """.format(row['question'], row['answer']), unsafe_allow_html=True)