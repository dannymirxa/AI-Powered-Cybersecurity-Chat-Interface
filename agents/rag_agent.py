"""
rag_agent.py
────────────
Specialised agent: answers questions grounded in the cybersecurity
knowledge base (NIST CSF 2.0 docs stored in Milvus).

Model  : qwen2.5:7b  — small but strong at instruction following
Tools  : search_knowledge_base  (from MCP server)
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_agent

SYSTEM_PROMPT = """You are a Cybersecurity Knowledge Expert specialised in the
NIST Cybersecurity Framework 2.0, risk management, and security best practices.

Your responsibilities:
- Answer questions using the knowledge base — always call search_knowledge_base first
- Cite the source document and chunk in every answer
- If the knowledge base returns no relevant results, say so clearly
- Never fabricate compliance requirements or framework details
- Keep answers structured: context → finding → recommendation

You do NOT handle live threat data (CVEs, IPs, passwords) — defer those to other agents.
"""


def create_rag_agent(tools: list):
    """
    Build and return the RAG react agent.

    Args:
        tools: LangChain-compatible tools loaded from the MCP server.
               Only the search_knowledge_base tool is used here;
               the others are passed but not exposed via the prompt.
    """
    # Filter to only the RAG tool so the model isn't distracted
    rag_tools = [t for t in tools if t.name == "search_knowledge_base"]

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.1,          # low temp → factual, consistent
        num_ctx=8192,
    )

    agent = create_agent(
        model=llm,
        tools=rag_tools,
        name="rag_agent",
        prompt=SYSTEM_PROMPT,
    )
    return agent