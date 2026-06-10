
import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask me anything about company HR policies!")

# ── Load API keys ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Set it in your environment variables.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── HR keywords for guardrail ─────────────────────────────────
HR_KEYWORDS = [
    "leave", "salary", "policy", "employee", "work from home",
    "remote", "performance", "review", "conduct", "harassment",
    "travel", "expense", "onboarding", "separation", "benefits",
    "compensation", "probation", "joining", "it", "data", "security",
    "appraisal", "pip", "wfh", "maternity", "paternity", "sick",
    "holiday", "reimbursement", "device", "zyro", "hr", "handbook"
]

def is_hr_question(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in HR_KEYWORDS)

# ── Build RAG pipeline (cached so it only runs once) ──────────
@st.cache_resource(show_spinner="📚 Loading HR documents & building knowledge base...")
def build_rag():
    # Load PDFs
    pdf_paths = glob.glob("/kaggle/input/**/*.pdf", recursive=True)
    if not pdf_paths:
        pdf_paths = glob.glob("./**/*.pdf", recursive=True)

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)

    # Embed + Vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30}
    )

    # LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY based on the context provided below.
If the answer is not in the context, say exactly:
"I can only answer HR-related questions from Zyro Dynamics policy documents."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

chain, retriever = build_rag()

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Source chunks used"):
                for s in msg["sources"]:
                    st.caption(f"**{s['source']}** (page {s['page']})")
                    st.text(s["content"])

# ── Chat input ────────────────────────────────────────────────
if user_input := st.chat_input("Ask an HR question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            if not is_hr_question(user_input):
                answer = "I can only answer HR-related questions from Zyro Dynamics policy documents."
                sources = []
            else:
                answer = chain.invoke(user_input)
                # Get sources
                source_docs = retriever.invoke(user_input)
                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown").split("/")[-1],
                        "page": doc.metadata.get("page", "?"),
                        "content": doc.page_content[:300]
                    }
                    for doc in source_docs
                ]

        st.markdown(answer)

        if sources:
            with st.expander("📄 Source chunks used"):
                for s in sources:
                    st.caption(f"**{s['source']}** (page {s['page']})")
                    st.text(s["content"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources if sources else []
    })
