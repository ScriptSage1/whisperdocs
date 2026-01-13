import streamlit as st
import os
import shutil
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "qwen/qwen3-32b"
EMBEDDING_MODEL = "text-embedding-mxbai-embed-large-v1"
DB_PATH = "./chroma_db"

st.set_page_config(
    page_title="WhisperDocs",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("WhisperDocs")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if "process_clicked" not in st.session_state:
    st.session_state.process_clicked = False


def get_embedding_function():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=LMSTUDIO_BASE_URL,
        openai_api_key="lm-studio",
    )

def load_db():
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=get_embedding_function()
        )
    return None

def clear_database():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    st.session_state.vector_db = None
    st.session_state.messages = []
    st.success("Database cleared!")
    st.rerun()

if st.session_state.vector_db is None:
    st.session_state.vector_db = load_db()



with st.sidebar:
    st.header("Controls")

    if st.session_state.vector_db:
        st.success("Database Loaded")
        if st.button("Clear Database"):
            clear_database()
    else:
        st.warning("Database Empty")

    st.markdown("---")

    st.subheader("Upload Doc")

    uploaded_file = st.file_uploader(
        "Choose PDF or TXT",
        type=["pdf", "txt"],
        key="uploader"
    )

    if st.button("Process Document"):
        st.session_state.process_clicked = True


if st.session_state.process_clicked and uploaded_file:

    with st.spinner("Embedding document..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(docs)

            if st.session_state.vector_db:
                st.session_state.vector_db.add_documents(splits)
                st.success(f"Added {len(splits)} chunks to existing DB.")
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

    
    st.session_state.process_clicked = False
    st.rerun()



if not st.session_state.vector_db:
    st.info("Upload a document to start chatting!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("assistant"):
            llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_base=LMSTUDIO_BASE_URL,
                openai_api_key="lm-studio",
                temperature=0,
                streaming=True
            )

            retriever = st.session_state.vector_db.as_retriever(
                search_kwargs={"k": 5}
            )

            template = """
            Answer strictly based on the context below.

            Context:
            {context}

            Question:
            {question}
            """

            prompt_template = ChatPromptTemplate.from_template(template)

            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )

            response_box = st.empty()
            full_response = ""

            for chunk in chain.stream(prompt):
                full_response += chunk
                response_box.markdown(full_response + "▌")

            response_box.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )