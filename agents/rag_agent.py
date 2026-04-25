"""
rag_agent.py
────────────
Specialised agent: answers questions grounded in the cybersecurity
knowledge base (NIST CSF 2.0 docs stored in Milvus).

Model  : AGENT_MODEL env var  (default: qwen2.5:3b)
Tools  : search_knowledge_base  (from MCP server)
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

AGENT_MODEL     = os.getenv("AGENT_MODEL",  "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL",   "http://localhost:11434")

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
    rag_tools = [t for t in tools if t.name == "search_knowledge_base"]

    llm = ChatOllama(
        model=AGENT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=4096,
    )
    print(f"✅ rag_agent model: {AGENT_MODEL}")

    return create_agent(
        model=llm,
        tools=rag_tools,
        name="rag_agent",
        system_prompt=SYSTEM_PROMPT,
    )
