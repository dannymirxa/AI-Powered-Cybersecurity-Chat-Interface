"""
threat_agent.py
───────────────
Specialised agent: live threat intelligence.
  - CVE lookups from NIST NVD
  - IP reputation checks via AbuseIPDB

Model  : qwen2.5:7b
Tools  : lookup_cve, check_ip  (from MCP server)
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_agent

SYSTEM_PROMPT = """You are a Threat Intelligence Analyst with access to live
vulnerability and IP reputation databases.

Your responsibilities:
- Look up CVE details when the user mentions a vulnerability ID or asks about a product flaw
- Check IP addresses for malicious activity when an IP is mentioned or suspicious traffic is reported
- Interpret CVSS scores and explain severity in plain language
- Summarise threat findings concisely: what it is, what it affects, how critical, what to do

Always present:
  • CVE ID / IP address
  • Severity & score (for CVEs) or confidence score (for IPs)
  • Affected systems or ISP/country (for IPs)
  • Recommended action

You do NOT handle knowledge-base questions or password checks — defer those.
"""


def create_threat_agent(tools: list):
    """
    Build and return the threat intelligence react agent.

    Args:
        tools: Full tool list from MCP; agent is given only its relevant tools.
    """
    threat_tools = [
        t for t in tools if t.name in ("lookup_cve", "check_ip")
    ]

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.1,
        num_ctx=4096,
    )

    agent = create_agent(
        model=llm,
        tools=threat_tools,
        name="threat_agent",
        prompt=SYSTEM_PROMPT,
    )
    return agent