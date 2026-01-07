import streamlit as st
import os
import shutil
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:14b"
EMBEDDING_MODEL = "mxbai-embed-large"
DB_PATH = "./chroma_db"

st.set_page_config(page_title="WhisperDocs", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(#1c2333 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(26, 28, 36, 0.75);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3 {
        color: #e0e0e0;
        text-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

st.title("WhisperDocs")

def get_embedding_function():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

def load_db():
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=get_embedding_function())
    return None

def clear_database():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        st.session_state.vector_db = None
        st.session_state.messages = [] 
        st.success("Database cleared!")
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state or st.session_state.vector_db is None:
    st.session_state.vector_db = load_db()

with st.sidebar:
    st.header("Controls")
    
    if st.session_state.vector_db:
        st.success("Database Loaded")
        if st.button("Clear Database (Start Fresh)", type="primary"):
            clear_database()
    else:
        st.warning("Database Empty")

    st.markdown("---")
    
    st.subheader("Upload Doc")
    uploaded_file = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Embedding..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                if st.session_state.vector_db:
                    st.session_state.vector_db.add_documents(splits)
                    st.info(f"Added {len(splits)} chunks to existing DB.")
                else:
                    st.session_state.vector_db = Chroma.from_documents(
                        documents=splits,
                        embedding=get_embedding_function(),
                        persist_directory=DB_PATH
                    )
                    st.success(f"Created new DB with {len(splits)} chunks.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                st.rerun()

if not st.session_state.vector_db:
    st.info("Upload a document to start chatting!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
            
            template = """
            Answer strictly based on the context provided below. 
            Context: {context}
            Question: {question}
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            response_container = st.empty()
            full_response = ""
            
            for chunk in chain.stream(prompt):
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})