from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="data/chroma_store", embedding_function=embeddings)
results = db.similarity_search("soil organic carbon and biodiversity", k=3)
for r in results:
    print(r.page_content[:200], "\n---")