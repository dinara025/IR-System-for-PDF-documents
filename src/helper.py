import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACE_API_KEY

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits a large text into manageable chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store using Hugging Face embeddings.
    """
    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain using Hugging Face models.
    """
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("Hugging Face API key is missing. Please set it in the .env file.")

    # Initialize Hugging Face LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Example Hugging Face model
        model_kwargs={"temperature": 0.7, "max_length": 512}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain
