"""
mcp_rag_server.py
─────────────────
MCP server exposing all 4 cybersecurity tools to any MCP-compatible client
(Claude Desktop, LangGraph MCP adapter, Cursor, etc.).

This file is intentionally thin — all tool logic lives in tools/rag_tool.py.

Tools exposed:
  1. search_knowledge_base  — RAG semantic search over Milvus
  2. cve_lookup             — CVE details from NIST NVD
  3. check_ip_reputation    — IP abuse score from AbuseIPDB
  4. check_password_breach  — Pwned password check via HIBP

Run:
  python mcp_rag_server.py           # stdio transport (Claude Desktop / LangGraph)
  python mcp_rag_server.py --sse     # SSE transport (HTTP-based MCP clients)

Install: pip install mcp pymilvus requests
"""

import argparse
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from pymilvus import MilvusClient

from tools.rag_tool import (
    MILVUS_URI,
    get_milvus_client,
    search_cybersec_kb,
    cve_lookup,
    check_ip_reputation,
    check_password_breach,
)


# ── Lifespan: one shared Milvus connection for the whole server session ────────
@dataclass
class ServerContext:
    milvus: MilvusClient


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Open Milvus connection on startup, close cleanly on shutdown."""
    client = get_milvus_client()
    print(f"✅ MCP server connected to Milvus at {MILVUS_URI}")
    try:
        yield ServerContext(milvus=client)
    finally:
        client.close()
        print("🔌 Milvus connection closed")


# ── MCP Server ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "CyberSec Tools",
    dependencies=["pymilvus", "requests"],
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1 — RAG: Knowledge Base Search
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_knowledge_base(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.60,
    ctx: Context = None,
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
        score_threshold: Minimum cosine similarity to include a result (default 0.60).
                         Lower this (e.g. 0.50) if too few results are returned.

    Returns:
        {
          "query":   str,
          "results": [{ text, source, chunk_index, score }, ...]
        }
    """
    shared_client = ctx.request_context.lifespan_context.milvus

    return search_cybersec_kb(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        client=shared_client,
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
    - Asks about a recent patch or security advisory

    Provide either cve_id OR keyword — not both.

    Args:
        cve_id:           Exact CVE identifier (e.g. "CVE-2021-44228"). Takes
                          priority over keyword if both are provided.
        keyword:          Free-text search (e.g. "log4j remote code execution").
        results_per_page: Max results for keyword search (default 3, ignored for ID lookup).

    Returns:
        {
          "cves":  [{ id, description, severity, score, vector,
                      published, modified, references }, ...],
          "total": int
        }
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
    - Wants to know the origin or reputation of an IP

    Args:
        ip_address:   IPv4 or IPv6 address to check.
        max_age_days: How far back to look for reports (default 30 days).

    Returns:
        {
          "ip":               str,
          "is_malicious":     bool,
          "confidence_score": int   (0–100),
          "total_reports":    int,
          "country":          str,
          "isp":              str,
          "domain":           str,
          "usage_type":       str,
          "last_reported":    str or None
        }
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
    Uses Have I Been Pwned k-anonymity API — completely free, no API key needed.
    The password is never sent over the network (only a 5-char SHA-1 prefix is).

    Use this tool when the user:
    - Asks whether a password is safe or has been compromised
    - Is responding to a credential leak or account takeover incident
    - Wants to audit password quality as part of a security review
    - Mentions a specific password they want to check

    Args:
        password: The plaintext password to check.

    Returns:
        {
          "breached":       bool,
          "times_found":    int,
          "risk_level":     str  ("none" | "low" | "medium" | "high" | "critical"),
          "recommendation": str
        }
    """
    return check_password_breach(password=password)


# ── Entry point ───────────────────────────────────────────────────────────────
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
    # mcp.run()