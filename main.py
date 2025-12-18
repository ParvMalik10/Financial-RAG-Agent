import os
import sys
from dotenv import load_dotenv

# --- IMPORTS ---
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# NEW: Import Ollama instead of Google
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

print("--- 1. LOADING DATA ---")
if not os.path.exists("apple_10k.pdf"):
    print("Downloading sample PDF...")
    os.system('curl -L -o apple_10k.pdf "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/b4266e40-1de6-4a34-9dfb-8632b8bd57e0.pdf"')

loader = PyPDFLoader("apple_10k.pdf")
docs = loader.load()[:50] # Limit to 50 pages for speed

print(f"--- 2. SPLITTING {len(docs)} PAGES ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

print("--- 3. BUILDING VECTOR STORE (Local HuggingFace) ---")
# Uses your CPU/GPU to convert text to math
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

print("--- 4. INITIALIZING LOCAL LLM (Ollama) ---")
# Uses the model you pulled via 'ollama pull llama3.2'
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

system_prompt = (
    "You are a financial analyst assistant. "
    "Use the provided context to answer the question. "
    "If the answer is not in the context, say 'I cannot find this in the report'. "
    "Keep the answer professional and concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

query = "What were the total net sales for 2023? Summarize the main risk factors."
print(f"\n--- ASKING LOCAL AI: {query} ---")
print("(This runs 100% offline on your Mac. No internet needed.)")

try:
    response = rag_chain.invoke({"input": query})
    print(f"\nANSWER:\n{response['answer']}")
except Exception as e:
    print(f"\nERROR: {e}")
    print("Ensure you ran 'ollama serve' in a separate terminal!")
