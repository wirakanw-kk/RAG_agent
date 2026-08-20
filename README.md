# Two-Agent RAG Report System — Two Submitted Versions

**Agent 1 — Data Retriever (`DataRetrieverAgent`):** owns its own knowledge
base setup and retrieves raw, relevant snippets only — it never answers the
question via keyword scoring, since the sandbox
gateway has no embedding deployment. The assignment explicitly allows either
("performs a simple keyword or basic semantic search").

**Agent 2 — Report Generator (`ReportGeneratorAgent`):** synthesizes the
final answer. Coordinates with the Data Retriever using the **agent-as-tool**
pattern: the Data Retriever agent is wrapped as a single callable tool
(`retrieve_from_knowledge_base`), which the Report Generator's own
tool-calling loop invokes autonomously.

## Setup

```bash
pip install -r requirements.txt
```

## Use with your own data / query

```python
from main import TwoAgentRAGSystem

system = TwoAgentRAGSystem("path/to/knowledge_base.txt")
answer = system.answer("Your question here")
```

## Repo Structure

- `script.py` - Main execution code.
- `Result/` - Contains previous outputs