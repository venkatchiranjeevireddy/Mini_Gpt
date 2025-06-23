pip install langchain llama-cpp-python faiss-cpu sentence-transformers unstructured pdfminer.six python-docx

pip install -U langchain-community

from huggingface_hub import login, hf_hub_download
# Replace with your actual token
HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)
# Download quantized LLaMA 2 7B chat model (GGML format)
model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGML",
    filename="llama-2-7b-chat.ggmlv3.q4_0.bin"
)
print("Model downloaded at:", model_path)
pip install llama-cpp-python
from huggingface_hub import login, hf_hub_download
from langchain.llms import LlamaCpp
# Replace with your actual token
HUGGINGFACE_TOKEN = "hf_RCPHowCTgRJlmGPTXmhkUcPRaHbArsg" # Your token here
login(token=HUGGINGFACE_TOKEN)
# Download quantized LLaMA 2 7B chat model (GGUF format)
# Changed repo_id to get GGUF models
model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)
print("Model downloaded at:", model_path)
# Instantiate LlamaCpp with the new model_path
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=512,
    top_p=1,
    n_ctx=2048,
    verbose=False
)
model_path = "/root/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q4_K_M.gguf"
from google.colab import files
print("Upload your document (.pdf, .txt, .docx, .csv)")
uploaded = files.upload()
file_path = next(iter(uploaded))
import os
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
# 1️Load the document (PDF, TXT, DOCX, CSV)
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == ".csv":
        return CSVLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
# 2️Create or load FAISS vector store
def prepare_or_load_vectorstore(file_path, db_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(db_path):
        print("Vector DB found — loading")
        return FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    print("Processing document and creating vector store...")
    docs = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_path)
    return db
# 3️Load LLaMA 2 model (GGUF)
def load_llama_model(model_path):
    print("Loading LLaMA 2 7B Chat model...")
    return LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        n_ctx=4096,  # Full context
        verbose=False
    )
# 4️Chat with your document
def chat_with_document(vectorstore, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    print("\nChat is ready! Ask questions (type 'exit' to stop)")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print(" Bye bro!")
            break

        answer = qa_chain.invoke(query)
        print("\nLLaMA 2 says:\n" + "="*40)
        print(answer)
        print("="*40 + "\n")
# 5️Put it all together
model_path = "/root/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q4_K_M.gguf"
if not os.path.exists("faiss_index"):
    vectorstore = prepare_or_load_vectorstore(file_path)
else:
    vectorstore = prepare_or_load_vectorstore("placeholder.txt")  # dummy path, won't be used
llm = load_llama_model(model_path)
chat_with_document(vectorstore, llm)
