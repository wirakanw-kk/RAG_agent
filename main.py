import os
import getpass
import requests

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

load_dotenv()

'''#get azure API
from langchain_openai import AzureOpenAIEmbeddings
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)'''

def get_model():
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


#create and store vector
def create_vector_store(
    doc_path: str,
    chunk_size: int = 200,
    overlap: int = 20,
    embedding_model: str = "models/gemini-embedding-001",
) -> InMemoryVectorStore:

    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

    # Read the document
    with open(doc_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    documents = [Document(page_content=text_content, metadata={"source": doc_path})]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = text_splitter.split_documents(documents)

    # Create and return the in-memory vector store
    return InMemoryVectorStore.from_documents(
        documents=chunks, embedding=embeddings
    )


@tool(parse_docstring=True)
def search_data_base(
    vector_store: InMemoryVectorStore,
    query: str,
    k: int = 2
) ->str:
    """Search the document database"""
    retrieved_info = vector_store.similarity_search(query, k=k)

    if not retrieved_info:
        return "NO_RESULT_FOUND: No relevent content match the query"

    return

