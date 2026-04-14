import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Read Excel
df = pd.read_excel("jobs.xlsx")

# 2. Convert rows → text
texts = []
for _, row in df.iterrows():
    text = f"{row['Job Title']} job in {row['Country']} with salary {row['Salary']} and {row['Benefits']}"
    texts.append(text)

# 3. Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Create Chroma DB
db = Chroma.from_texts(
    texts,
    embedding,
    persist_directory="chroma_db"
)

# 5. Save DB
db.persist()

print("✅ Data stored in Chroma")

# 6. Search
query = input("Ask job query: ")

results = db.similarity_search(query)

print("\n🔍 Results:")
for r in results:
    print("-", r.page_content)