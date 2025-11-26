# main.py
import os
import tempfile
import traceback

import streamlit as st

# Load keys from environment (recommended) or set here for dev only
# Recommendation: set these in .env and load with python-dotenv (not shown here)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB1bcemrjNRk53GPbY_xNDzgdsIzyW5E9o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")  # optional fallback

# Basic checks

# LangChain / community imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# langchain-google-genai integration
from langchain_google_genai import ChatGoogleGenerativeAI

# ახალი, სწორი იმპორტი langchain.schema-ის ნაცვლად
from langchain_core.documents import Document


# --- UI ---
st.set_page_config(page_title="Medic RAG Chatbot", layout="centered")
st.title("Medic RAG Chatbot 🇬🇪")
st.write("ატვირთეთ PDF ან TXT (ქართული ჩათვლით), დააჭირეთ 'Process' — შემდეგ დაუსვით კითხვა.")

uploaded_file = st.file_uploader("ატვირთეთ PDF ან TXT ფაილი", type=["pdf", "txt"], accept_multiple_files=True)

# User input for query
query = st.text_input("შეიყვანეთ კითხვა (მაგ. რა მძიმე ნერვული სისტემის სიმპტომებია?):")

# Optional: model selection
model_option = st.selectbox("GenAI model (ჩამოტვირთეთ თქვენს ხელმისაწვდომობას):",
                            options=["gemini-2.5-flash", "gemini-2.5", "chat-bison-001"],
                            index=0)

# --- helper functions ---
def load_documents_from_uploaded(uploaded_files):
    """Convert list of Streamlit UploadedFile -> list[Document]"""
    docs = []
    for uploaded_file in uploaded_files:
        try:
            # write to temp file and use loaders (PyPDFLoader / TextLoader)
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            st.error(f"ფაილის '{uploaded_file.name}' ჩატვირთვის დროს მოხდა შეცდომა: {e}")
            st.write(traceback.format_exc())
    return docs

def build_vectorstore(documents, embedding_model_name="all-MiniLM-L6-v2"):
    emb = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    vs = FAISS.from_documents(documents, emb)
    return vs

def call_genai_with_context(llm, context_text: str, question: str):
    # According to langchain-google-genai docs use .invoke(messages) or .invoke(prompt)
    # We'll pass messages tuple-style: (("system", ...), ("human", ...))
    messages = [
        ("system", "You are a helpful medical assistant. Answer concisely and cite sources."),
        ("human", f"{context_text}\n\nQuestion: {question}")
    ]
    # invoke returns an AIMessage object; cast to text via .content (depends on version)
    resp = llm.invoke(messages)
    # resp may be an AIMessage or list — try to extract content robustly
    content = getattr(resp, "content", None)
    if content is None:
        # try mapping .text or resp[0].content
        if isinstance(resp, (list, tuple)) and len(resp) > 0:
            content = getattr(resp[0], "content", str(resp[0]))
        else:
            content = str(resp)
    return content

# --- Main flow ---
if st.button("Process files") and uploaded_file:
    with st.spinner("დოკუმენტები ჩაიტვირთება და индексირდება (FAISS)..."):
        docs = load_documents_from_uploaded(uploaded_file)
        if not docs:
            st.error("არცერთი დოკუმენტი არ ჩაიტვირთა/დადებული.")
        else:
            st.success(f"ჩატვირთულია {len(docs)} დოკუმენტი. გაყოფის შემდეგ ქმნა chunks და embeddings...")
            # Optional: show sample
            if len(docs) > 0:
                st.write("პირველი დოკუმენტის პატარა მაგალითი:")
                st.write(docs[0].page_content[:500])

            try:
                # build vectorstore
                vectorstore = build_vectorstore(docs, embedding_model_name="all-MiniLM-L6-v2")
                st.success("FAISS ინიციალიზირებულია.")
                st.session_state["vectorstore"] = vectorstore
            except Exception as e:
                st.error("ვექტორული ბაზის შექმნისას მოხდა შეცდომა:")
                st.write(e)
                st.write(traceback.format_exc())

# Query execution
if query:
    if "vectorstore" not in st.session_state:
        st.warning("გთხოვთ ჯერ დააჭიროთ 'Process files' რათა დოკუმენტები ინდექსირდეს.")
    else:
        vectorstore = st.session_state["vectorstore"]
        with st.spinner("კვლავდება მსგავს დოკუმენტები და გამოითვლება პასუხი..."):
            try:
                similar_docs = vectorstore.similarity_search(query, k=3)
                if not similar_docs:
                    st.info("დოკუმენტებში მსგავსი ფრაგმენტი ვერ მოიძებნა.")
                # build context text with simple citations (file source or excerpt)
                context_text = ""
                for i, d in enumerate(similar_docs, 1):
                    src = d.metadata.get("source", f"doc_{i}")
                    excerpt = d.page_content[:800].replace("\n", " ")
                    context_text += f"[Source: {src}]\n{excerpt}\n\n"

                # instantiate LLM (langchain-google-genai Chat model)
                llm = ChatGoogleGenerativeAI(
                    google_api_key=GOOGLE_API_KEY,
                    model=model_option,
                    temperature=0.0
                )

                answer_text = call_genai_with_context(llm, context_text, query)
                st.subheader("🤖 პასუხი")
                st.write(answer_text)

                st.subheader("📄 მოსაგროვებელი წყაროები (excerpts)")
                for d in similar_docs:
                    st.write(f"- {d.metadata.get('source', 'unknown')}: {d.page_content[:300].replace(chr(10),' ')}...")

            except Exception as e:
                st.error("LLM/რეგულაციის დროს მოხდა შეცდომა:")
                st.write(str(e))
                st.write(traceback.format_exc())
