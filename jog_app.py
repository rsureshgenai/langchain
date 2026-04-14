import streamlit as st
import pandas as pd
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Job AI", layout="wide")

# ===== GLOBAL STYLE =====
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}

/* Remove default Streamlit alerts */
div[data-testid="stAlert"] {
    display: none;
}

/* Search input styling */
input {
    border: 2px solid #1E3A8A !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ===== TOP SPACE =====
st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

# ===== HEADER =====
col1, col_space, col2 = st.columns([3, 8, 4])

with col1:
    st.image("logo.png", width=300)

with col2:
    st.markdown("""
        <div style='text-align:right; white-space:nowrap; margin-top:10px;'>
            <span style='color:#FFC107; font-size:42px; font-weight:bold;'>
                Job Search
            </span>
            <span style='color:#1E3A8A; font-size:28px; font-weight:bold; margin-left:8px;'>
                AI
            </span>
            <p style='color:gray; font-size:14px; margin-top:5px;'>
                Find the best jobs using AI-powered search
            </p>
        </div>
    """, unsafe_allow_html=True)

# ===== SPACE =====
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ===== BANNER =====
st.image("banner.jpg", use_container_width=True)

st.markdown("---")

# ===== LOAD EXCEL =====
df = pd.read_excel("jobs.xlsx")

# ===== CONVERT TO TEXT =====
texts = []
for _, row in df.iterrows():
    text = f"{row['Job Title']} job in {row['Country']} with salary {row['Salary']} and {row['Benefits']}"
    texts.append(text)

# ===== EMBEDDING =====
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding = load_embedding()

# ===== DATABASE =====
@st.cache_resource
def load_db(texts):
    if not os.path.exists("chroma_db"):
        db = Chroma.from_texts(texts, embedding, persist_directory="chroma_db")
        db.persist()
    else:
        db = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    return db

db = load_db(texts)

# ===== BRAND SUCCESS BOX (BLUE TICK) =====
st.markdown("""
<div style="
    background: linear-gradient(90deg, #FFC107, #1E3A8A);
    padding:15px;
    border-radius:10px;
    margin-top:10px;
    color:white;
    font-weight:500;
    font-size:16px;
    box-shadow: 0 0 15px rgba(255,193,7,0.3);
    display:flex;
    align-items:center;
    gap:10px;
">
    <span style="
        background:#1E3A8A;
        color:white;
        border-radius:50%;
        width:22px;
        height:22px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:14px;
        box-shadow: 0 0 8px #1E3A8A;
    ">
        ✓
    </span>
    Jobs loaded successfully
</div>
""", unsafe_allow_html=True)

# ===== SEARCH =====
query = st.text_input("🔍 Search jobs")

# ===== RESULTS =====
if query:
    results = db.similarity_search(query)

    st.markdown("""
    <h3 style='color:#FFC107;'>🎯 Results</h3>
    """, unsafe_allow_html=True)

    for r in results:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1e1e1e, #2c2c2c);
                padding:18px;
                border-radius:12px;
                margin-bottom:12px;
                border-left:4px solid #FFC107;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            ">
                <p style="color:white;font-size:16px; margin:0;">
                {r.page_content}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )