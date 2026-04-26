"""
mcp_rag_server.py
─────────────────
MCP server exposing all 4 cybersecurity tools to any MCP-compatible client.

Tools exposed:
  1. search_knowledge_base  — RAG semantic search over Milvus
  2. lookup_cve             — CVE details from NIST NVD
  3. check_ip               — IP abuse score from AbuseIPDB
  4. check_breach           — Pwned password check via HIBP

Run:
  python mcp_rag_server.py           # stdio transport (LangGraph / Claude Desktop)
  python mcp_rag_server.py --sse     # SSE transport (HTTP-based MCP clients)

Change from previous version:
  The Milvus connection is now opened LAZILY inside each search_knowledge_base
  call rather than eagerly in the server lifespan.  This means the MCP subprocess
  starts instantly and reports its tools to the parent process without waiting
  for Milvus to be available, which was the cause of the startup hang in FastAPI.
"""

import argparse

from mcp.server.fastmcp import FastMCP

from tools.rag_tool import (
    search_cybersec_kb,
    cve_lookup,
    check_ip_reputation,
    check_password_breach,
)

# ── MCP Server ────────────────────────────────────────────────────────────
# No lifespan — Milvus client is created and closed inside each tool call.
# This lets the subprocess start immediately without waiting for Milvus.
mcp = FastMCP(
    "CyberSec Tools",
    dependencies=["pymilvus", "requests"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1 — RAG: Knowledge Base Search
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_knowledge_base(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.60,
) -> dict:
    """
    Search the cybersecurity knowledge base (NIST CSF 2.0 and related docs)
    using semantic vector search.

    Use this tool when the user asks about:
    - NIST Cybersecurity Framework functions, categories, or subcategories
    - Risk management practices or security controls
    - Incident response procedures or recovery planning
    - Real-world cybersecurity incidents referenced in the framework
    - Compliance requirements or governance topics

    Args:
        query:           The user's question or search phrase.
        top_k:           Number of results to return (default 5, max 10).
        score_threshold: Minimum cosine similarity (default 0.60).

    Returns:
        {"query": str, "results": [{text, source, chunk_index, score}, ...]}
    """
    # Milvus client opened and closed per call — no shared connection needed
    return search_cybersec_kb(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        client=None,   # search_cybersec_kb creates its own client when None
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2 — CVE Lookup
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def lookup_cve(
    cve_id: str = None,
    keyword: str = None,
    results_per_page: int = 3,
) -> dict:
    """
    Look up CVE vulnerability details from the NIST National Vulnerability Database.
    No API key required.

    Use this tool when the user:
    - Mentions a specific CVE ID (e.g. CVE-2021-44228)
    - Asks about vulnerabilities in a specific product or software
    - Wants to know the CVSS score or severity of a vulnerability

    Provide either cve_id OR keyword — not both.

    Args:
        cve_id:           Exact CVE identifier (e.g. "CVE-2021-44228").
        keyword:          Free-text search (e.g. "log4j remote code execution").
        results_per_page: Max results for keyword search (default 3).

    Returns:
        {"cves": [{id, description, severity, score, vector, published, modified, references}], "total": int}
    """
    return cve_lookup(
        cve_id=cve_id,
        keyword=keyword,
        results_per_page=results_per_page,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3 — IP Reputation Check
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def check_ip(
    ip_address: str,
    max_age_days: int = 30,
) -> dict:
    """
    Check whether an IP address has been reported for malicious activity.
    Uses AbuseIPDB (free tier: 1,000 checks/day). Requires ABUSEIPDB_API_KEY env var.

    Use this tool when the user:
    - Pastes or mentions a suspicious IP address
    - Asks whether an IP is safe or malicious
    - Is investigating a network intrusion or unusual connection

    Args:
        ip_address:   IPv4 or IPv6 address to check.
        max_age_days: How far back to look for reports (default 30 days).

    Returns:
        {ip, is_malicious, confidence_score, total_reports, country, isp, domain, usage_type, last_reported}
    """
    return check_ip_reputation(
        ip_address=ip_address,
        max_age_days=max_age_days,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 4 — Password Breach Check
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def check_breach(password: str) -> dict:
    """
    Check whether a password has appeared in known data breach dumps.
    Uses Have I Been Pwned k-anonymity API — free, no API key needed.
    The password is never sent over the network (only a 5-char SHA-1 prefix is).

    Use this tool when the user:
    - Asks whether a password is safe or has been compromised
    - Is responding to a credential leak or account takeover incident
    - Wants to audit password quality as part of a security review

    Args:
        password: The plaintext password to check.

    Returns:
        {breached, times_found, risk_level, recommendation}
    """
    return check_password_breach(password=password)


# ── Entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberSec MCP Tool Server")
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Use SSE transport instead of stdio (for HTTP-based MCP clients)",
    )
    args = parser.parse_args()

    transport = "sse" if args.sse else "stdio"
    print(f"🚀 Starting CyberSec MCP server  [transport: {transport}]")
    mcp.run(transport=transport)
