"""
audit_agent.py
──────────────
Specialised agent: security audit & credential hygiene.
  - Password breach checks via HIBP (k-anonymity, no plaintext sent)
  - Policy recommendations based on breach results

Model  : qwen2.5:7b
Tools  : check_breach  (from MCP server)
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

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
    """
    Build and return the audit/credential react agent.

    Args:
        tools: Full tool list from MCP; agent is given only its relevant tool.
    """
    audit_tools = [t for t in tools if t.name == "check_breach"]

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.1,
        num_ctx=4096,
    )

    agent = create_react_agent(
        model=llm,
        tools=audit_tools,
        name="audit_agent",
        prompt=SYSTEM_PROMPT,
    )
    return agent