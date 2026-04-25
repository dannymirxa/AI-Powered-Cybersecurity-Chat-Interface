"""
audit_agent.py
──────────────
Specialised agent: security audit & credential hygiene.
  - Password breach checks via HIBP (k-anonymity, no plaintext sent)
  - Policy recommendations based on breach results

Model  : AGENT_MODEL env var  (default: qwen2.5:3b)
Tools  : check_breach  (from MCP server)
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

AGENT_MODEL     = os.getenv("AGENT_MODEL",  "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL",   "http://localhost:11434")

SYSTEM_PROMPT = """You are a Security Auditor specialised in credential hygiene
and access control policy.

Your responsibilities:
- Check whether passwords have appeared in known breach databases
- Never store, log, or repeat passwords beyond what is needed to call the tool
- Explain breach risk in plain language (none / low / medium / high / critical)
- Always recommend actionable remediation steps:
    • Immediate password rotation
    • MFA enablement
    • Credential manager adoption
    • Monitoring for account takeover indicators

Privacy note: remind users not to check passwords currently protecting live accounts;
use this tool only for auditing purposes or test credentials.

You do NOT handle vulnerability lookups or knowledge-base questions — defer those.
"""


def create_audit_agent(tools: list):
    audit_tools = [t for t in tools if t.name == "check_breach"]

    llm = ChatOllama(
        model=AGENT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=4096,
    )
    print(f"✅ audit_agent model: {AGENT_MODEL}")

    return create_agent(
        model=llm,
        tools=audit_tools,
        name="audit_agent",
        system_prompt=SYSTEM_PROMPT,
    )
