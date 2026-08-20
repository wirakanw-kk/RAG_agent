import os
import getpass
import re


from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# --- Custom gateway configuration (from given curl payload)
if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for the gateway: ")

if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = input(
        "Enter gateway base endpoint (e.g. https://<resource>.azure-api.net): "
    )

AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

# Defensive normalization: if someone pastes the full path from the curl
_raw_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
for _suffix in ("/llm/responses", "/llm/embeddings", "/llm"):
    if _raw_endpoint.endswith(_suffix):
        _raw_endpoint = _raw_endpoint[: -len(_suffix)]
        break

# The OpenAI SDK appends the operation name
GATEWAY_BASE_URL = _raw_endpoint + "/llm/"
LLM_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-5-mini")


def load_chunks(doc_path: str, chunk_size: int = 200, overlap: int = 20) -> list[Document]:
    """Read and chunk the knowledge base file."""
    with open(doc_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text_content)
    return [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)]


class DataRetrieverAgent:
    """Retrieves raw snippet, does not answer query.

    Using keyword search due to the availability of the model
    """

    WORD_PATTERN = re.compile(r"\w+")

    def __init__(self, doc_path: str, llm: ChatOpenAI, k: int = 3):
        self.chunks = load_chunks(doc_path)
        self.k = k
        self.llm = llm

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for text relevant to the query."""
            results = self._keyword_search(query)
            if not results:
                return "NO_RESULTS_FOUND"
            return "\n\n---\n\n".join(
                f"[Snippet {i + 1}]\n{d.page_content}" for i, d in enumerate(results)
            )

        self.tools = [search_knowledge_base]

        prompt = (
            "You are a Data Retriever. Call search_knowledge_base once to "
            "find snippets relevant to the query, then return them verbatim, "
            "labeled and deduplicated. Do not answer, summarize, or "
            "interpret."
        )

        self.agent = create_agent(model=self.llm, tools=self.tools, system_prompt=prompt)

    def _keyword_search(self, query: str) -> list[Document]:
        """Score each chunk by how many distinct query words it contains,
        case-insensitive, and return the top-k chunks with a nonzero score."""
        query_words = set(self.WORD_PATTERN.findall(query.lower()))
        if not query_words:
            return []

        scored = []
        for doc in self.chunks:
            content_lower = doc.page_content.lower()
            score = sum(1 for w in query_words if w in content_lower)
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in scored[: self.k]]

    def retrieve(self, query: str) -> str:
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].text


class ReportGeneratorAgent:
    """Synthesizes an answer to the user's question.

    Coordinates with the Data Retriever using the AGENT-AS-TOOL pattern.
    """

    def __init__(self, retriever_agent: DataRetrieverAgent, llm: ChatOpenAI):
        self.retriever_agent = retriever_agent

        @tool
        def retrieve_from_knowledge_base(query: str) -> str:
            """Retrieve relevant raw text snippets from the knowledge base."""
            return self.retriever_agent.retrieve(query)

        self.tools = [retrieve_from_knowledge_base]

        system_prompt = (
            "You are a Report Generator. Call retrieve_from_knowledge_base "
            "once to gather information, then answer the question using only "
            "that information. If any part contains information relevant "
            "to the question - even partial or indirect - use it to write a "
            "concise, cohesive answer; do not refuse just because the "
            "answer isn't stated in full detail. Only say you can't answer "
            "if the information are genuinely unrelated to the question - if "
            "so, say that in one brief sentence and stop there. Never "
            "speculate about other things the user might have meant or list "
            "alternate interpretations of the query. Don't mention snippets "
            "or tools."
        )

        self.agent = create_agent(model=llm, tools=self.tools, system_prompt=system_prompt)

    def generate(self, question: str, verbose: bool = False) -> str:
        result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result["messages"]

        if verbose:
            for m in messages:
                if m.__class__.__name__ == "ToolMessage":
                    print("=" * 70)
                    print("[Data Retriever agent called via tool]")
                    print("=" * 70)
                    print(m.content, "\n")

        return messages[-1].text


class TwoAgentRAGSystem:
    def __init__(self, txt_path: str):
        self.llm = ChatOpenAI(
            base_url=GATEWAY_BASE_URL,
            api_key="in_headers",
            default_headers={"api-key": AZURE_OPENAI_API_KEY},
            model=LLM_DEPLOYMENT,
            use_responses_api=True,
            reasoning_effort="minimal",
            max_tokens=300,  #due to sandbox rate limit
            max_retries=6,  
        )
        self.retriever = DataRetrieverAgent(txt_path, self.llm)
        self.generator = ReportGeneratorAgent(self.retriever, self.llm)

    def answer(self, query: str, verbose: bool = True) -> str:
        report = self.generator.generate(query, verbose=verbose)
        print("QUERY: ")
        print(query)
        if verbose:
            print("=" * 70)
            print("REPORT GENERATOR OUTPUT (final answer)")
            print("=" * 70)
            print(report)

        return report


if __name__ == "__main__":
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
    system = TwoAgentRAGSystem(kb_path)

    query = "Who is the organization that purview MAS?"
    system.answer(query)