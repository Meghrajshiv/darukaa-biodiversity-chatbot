import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from structured_data import load_reference_table, query_reference_table

load_dotenv()

CHROMA_DIR = "data/chroma_store"

SYSTEM_PROMPT = """You are an AI environmental scientist specializing in biodiversity and soil health.

RULES:
1. NEVER give generic advice like "use sustainable practices." Every recommendation must be specific and actionable.
2. Every recommendation MUST include:
   - What to do (specific action)
   - Why it works (scientific reasoning)
   - Which environmental metric it improves
   - A reference to a study/report/source (use the provided context)
   - Time horizon (short/medium/long term)
3. You MUST connect at least 2 environmental variables together (e.g. soil health + water availability, or land use + biodiversity). Never give single-variable answers.
4. If the user's input is missing key details (soil organic carbon %, rainfall, land use type, region), ASK a clarifying question instead of guessing.
5. Base your answer on the provided context below. If the context doesn't cover something, say so rather than inventing a citation.
"""

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def build_context(user_query, db, k=5):
    results = db.similarity_search(user_query, k=k)
    context = "\n\n".join([f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.page_content}" for r in results])
    return context

def get_structured_context(user_query):
    df = load_reference_table()
    relevant_rows = []
    for _, row in df.iterrows():
        if row["metric"].replace("_", " ") in user_query.lower():
            relevant_rows.append(f"- {row['metric']}: if {row['condition']} {row['threshold']} -> {row['impact']}, action: {row['recommended_action']} (source: {row['source']})")
    return "\n".join(relevant_rows) if relevant_rows else "No structured data matched."

def ask(user_query, db, llm, conversation_history):
    doc_context = build_context(user_query, db)
    structured_context = get_structured_context(user_query)

    full_prompt = f"""{SYSTEM_PROMPT}

RETRIEVED KNOWLEDGE (from research documents):
{doc_context}

STRUCTURED REFERENCE DATA:
{structured_context}

CONVERSATION HISTORY:
{conversation_history}

USER QUERY:
{user_query}

Respond following the RULES above."""

    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(full_prompt)
            content = response.content
            if isinstance(content, list):
                content = "\n".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"(Model busy, retrying in 5 seconds... attempt {attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                return f"Sorry, the model is currently unavailable after {max_retries} attempts. Please try again shortly. (Error: {e})"

if __name__ == "__main__":
    db = get_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", google_api_key=os.getenv("GOOGLE_API_KEY"))

    print("AI Biodiversity Assistant ready. Type 'exit' to quit.\n")
    history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        answer = ask(user_input, db, llm, history)
        print(f"\nAssistant: {answer}\n")
        history += f"User: {user_input}\nAssistant: {answer}\n"