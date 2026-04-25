"""
threat_agent.py
───────────────
Specialised agent: live threat intelligence.
  - CVE lookups from NIST NVD
  - IP reputation checks via AbuseIPDB

Model  : AGENT_MODEL env var  (default: qwen2.5:3b)
Tools  : lookup_cve, check_ip  (from MCP server)
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

load_dotenv()

AGENT_MODEL     = os.getenv("AGENT_MODEL",  "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL",   "http://localhost:11434")

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
    threat_tools = [t for t in tools if t.name in ("lookup_cve", "check_ip")]

    llm = ChatOllama(
        model=AGENT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_ctx=4096,
    )
    print(f"✅ threat_agent model: {AGENT_MODEL}")

    return create_agent(
        model=llm,
        tools=threat_tools,
        name="threat_agent",
        system_prompt=SYSTEM_PROMPT,
    )
