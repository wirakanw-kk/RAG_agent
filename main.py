import os
import getpass
import re


from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_agent

'''#get azure API
from langchain_openai import AzureOpenAIEmbeddings
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)'''

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
API_KEY = os.environ.get("OPENAI_API_KEY")

#create and store vector
def create_vector_store(
    doc_path: str,
    chunk_size: int = 200,
    overlap: int = 20,
) -> tuple[InMemoryVectorStore, list[Document]]:

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    with open(doc_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text_content)
    document = [Document(page_content=c,metadata={"chunk_id": i}) for i, c in enumerate(chunks)]

    store = InMemoryVectorStore(embeddings)
    store.add_documents(document)
    return store, document


class DataRetrieverAgent:
    """Retrieves raw snippet, does not answer query"""

    NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*\b")

    def __init__(self, vectorstore: InMemoryVectorStore, chunks: list[Document], llm: ChatOpenAI, k: int = 2):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.k = k
        self.llm = llm

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for text relevant to the query."""
            semantic_results = self.vectorstore.similarity_search(query, k=self.k)

            #for numerical matching (question concerning years and numbers)
            numbers = self.NUMBER_PATTERN.findall(query)
            keyword_results = (
                [d for d in self.chunks if any(n in d.page_content for n in numbers)]
                if numbers
                else []
            )

            seen = set()
            combined = []
            for d in semantic_results + keyword_results:
                key = d.page_content.strip()
                if key not in seen:
                    seen.add(key)
                    combined.append(d)

            if not combined:
                return "NO_RESULTS_FOUND"
            return "\n\n---\n\n".join(
                f"[Snippet {i + 1}]\n{d.page_content}" for i, d in enumerate(combined)
            )

        self.tools = [search_knowledge_base]

        prompt =(
            "You are a Data Retriever. Your ONLY job is to call the "
            "search_knowledge_base tool to find text snippets relevant to the "
            "user's query, then return those snippets verbatim, labeled and "
            "grouped. Do NOT answer the question, summarize, interpret, or add "
            "opinions. You may call the tool more than once with different "
            "phrasings if the first search seems incomplete. Return only the "
            "raw snippets you found, with duplicates removed."
                )

        self.agent = create_agent(model=self.llm, tools=self.tools, system_prompt=prompt)

    def retrieve(self, query: str) -> str:
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

class ReportGeneratorAgent:
    """Synthesizes answer from the retreived data and query"""

    def __init__(self, llm: ChatOpenAI):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Report Generator. You receive a user's question and a "
                    "set of raw text snippets retrieved from a knowledge base. Your job:\n"
                    "1. Synthesize the snippets into a single, cohesive answer.\n"
                    "2. Remove redundancy - never repeat the same fact twice.\n"
                    "3. Only use information present in the snippets; if they don't "
                    "answer the question, say so explicitly rather than guessing.\n"
                    "4. Format clearly (short paragraphs and/or bullet points).\n"
                    "5. Do not mention 'snippets' or the retrieval process - write a "
                    "standalone, paragraph.",
                ),
                ("human", "Question: {question}\n\nRetrieved snippets:\n{snippets}"),
            ]
        )
        self.chain = self.prompt | llm | StrOutputParser()

    def generate(self, question: str, snippets: str) -> str:
        return self.chain.invoke({"question": question, "snippets": snippets})


class TwoAgentRAGSystem:
    def __init__(self, txt_path: str):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.vectorstore, self.chunks = create_vector_store(txt_path)
        self.retriever = DataRetrieverAgent(self.vectorstore, self.chunks, self.llm)
        self.generator = ReportGeneratorAgent(self.llm)

    def answer(self, query: str, verbose: bool = True) -> str:
        snippets = self.retriever.retrieve(query)
        if verbose:
            print("=" * 70)
            print("DATA RETRIEVER OUTPUT (raw snippets)")
            print("=" * 70)
            print(snippets, "\n")

        report = self.generator.generate(query, snippets)
        if verbose:
            print("=" * 70)
            print("REPORT GENERATOR OUTPUT (final answer)")
            print("=" * 70)
            print(report)

        return report

if __name__ == "__main__":
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
    system = TwoAgentRAGSystem(kb_path)

    query = "What happended it 2002"
    system.answer(query)

