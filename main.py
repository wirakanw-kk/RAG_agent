''' langchain, using agent-as-tool

Data Receiver
 - Samantic Search
 - return RawData
Report Generator
 - Synthesise easy to read for user
 - return WhatUserRead
'''

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

#use gemini api to test (embedding)
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

doc_path = "knowledge_base.txt"

with open(doc_path, "r", encoding="utf-8") as f:
    text_content = f.read()
document = [Document(page_content=text_content, metadata={"source": doc_path})]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 125, chunk_overlap = 20)
chunck = text_splitter.split_documents(document)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore.from_documents(documents=chunck,embedding=embeddings) #in memory for small data

@tool
def search_data_base(query: str) ->str:
    """Search the document database for user queries."""
    retrieved_info = vector_store.similarity_search(query, k=3)

    if not retrieved_info:
        return "NO_RESULT_FOUND: No relevent content match the query"

    return

@tool
def generate_report()