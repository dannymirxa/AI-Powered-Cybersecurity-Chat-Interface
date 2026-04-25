"""
tools/rag_tool.py
─────────────────
All cybersecurity tools in one place — pure functions, no MCP, no server.
Each tool can be imported individually or tested from the CLI.

Tools:
  1. search_cybersec_kb     — RAG: semantic search over Milvus knowledge base
  2. cve_lookup             — CVE details from NIST NVD (no API key needed)
  3. check_ip_reputation    — IP abuse score from AbuseIPDB (free API key)
  4. check_password_breach  — Pwned password check via HIBP k-anonymity (no key)

CLI test:
  python tools/rag_tool.py rag      "What is NIST CSF Govern function?"
  python tools/rag_tool.py cve      CVE-2021-44228
  python tools/rag_tool.py ip       8.8.8.8
  python tools/rag_tool.py breach   password123
"""

import hashlib
import os
import re
import requests
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

# ── Shared config ─────────────────────────────────────────────────────────────
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
MILVUS_URI      = os.getenv("MILVUS_URI",      "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cybersec_kb")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "nomic-embed-text")
ABUSEIPDB_KEY   = os.getenv("ABUSEIPDB_API_KEY", "")

DEFAULT_TIMEOUT = 15   # seconds for all HTTP calls


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — RAG: Semantic search over Milvus knowledge base
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_TOP_K        = 5
_SCORE_THRESHOLD      = 0.35   # cosine similarity — higher = more similar


def _get_embedding(text: str) -> list[float]:
    """Embed text using Ollama /api/embeddings."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_milvus_client() -> MilvusClient:
    """Return a fresh MilvusClient. Caller is responsible for .close()."""
    return MilvusClient(uri=MILVUS_URI)


def search_cybersec_kb(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    score_threshold: float = _SCORE_THRESHOLD,
    client: MilvusClient = None,
) -> dict:
    """
    Semantic search over the cybersecurity knowledge base in Milvus.

    Creates and closes its own MilvusClient unless one is passed in.
    Pass a shared client (e.g., from MCP server lifespan) to avoid
    reconnecting on every call.

    Args:
        query:           Natural-language search string.
        top_k:           Maximum chunks to return (capped at 10).
        score_threshold: Minimum similarity (0–1). Raise to tighten results.
        client:          Optional pre-connected MilvusClient.

    Returns:
        {
          "query":   str,
          "results": [{ text, source, chunk_index, score }, ...]
        }
    """
    top_k = min(top_k, 10)
    owns_client = client is None
    if owns_client:
        client = get_milvus_client()

    try:
        vector = _get_embedding(query)
        hits = client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            limit=top_k,
            output_fields=["text", "source", "chunk_index"],
            search_params={"metric_type": "COSINE"},
        )
        results = []
        for hit in hits[0]:
            score = round(1 - hit["distance"], 4)
            if score < score_threshold:
                continue
            results.append({
                "text":        hit["entity"]["text"],
                "source":      hit["entity"]["source"],
                "chunk_index": hit["entity"]["chunk_index"],
                "score":       score,
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return {"query": query, "results": results}

    finally:
        if owns_client:
            client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — CVE Lookup via NIST NVD
# No API key required. Rate limit: 50 requests per 30 seconds (unauthenticated).
# API docs: https://nvd.nist.gov/developers/vulnerabilities
# ═══════════════════════════════════════════════════════════════════════════════

_NVD_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _parse_cve(cve: dict) -> dict:
    """Extract the most useful fields from a raw NVD CVE object."""
    descriptions = cve.get("descriptions", [])
    english_desc = next(
        (d["value"] for d in descriptions if d.get("lang") == "en"),
        "No English description available.",
    )

    # Try CVSS v3.1 first, then v3.0, then v2
    metrics = cve.get("metrics", {})
    severity = score = vector = "N/A"
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            cvss_data = entries[0].get("cvssData", {})
            severity = cvss_data.get("baseSeverity", "N/A")
            score    = cvss_data.get("baseScore",    "N/A")
            vector   = cvss_data.get("vectorString",  "N/A")
            break

    references = [r["url"] for r in cve.get("references", [])[:3]]

    return {
        "id":          cve["id"],
        "description": english_desc,
        "severity":    severity,
        "score":       score,
        "vector":      vector,
        "published":   cve.get("published", "N/A")[:10],
        "modified":    cve.get("lastModified", "N/A")[:10],
        "references":  references,
    }


def cve_lookup(
    cve_id: str = None,
    keyword: str = None,
    results_per_page: int = 3,
) -> dict:
    """
    Look up CVE vulnerability details from the NIST National Vulnerability Database.

    Provide either a CVE ID (e.g. "CVE-2021-44228") for a precise lookup,
    or a keyword (e.g. "log4j remote code execution") for a broader search.
    At least one of cve_id or keyword must be provided.

    Args:
        cve_id:           Exact CVE identifier. Takes priority over keyword.
        keyword:          Free-text search term.
        results_per_page: Max results for keyword search (ignored for ID lookup).

    Returns:
        {
          "cves":  [{ id, description, severity, score, vector,
                      published, modified, references }, ...],
          "total": int
        }

    Raises:
        ValueError:           If neither cve_id nor keyword is provided.
        requests.HTTPError:   On NVD API errors.
    """
    if not cve_id and not keyword:
        raise ValueError("Provide either cve_id or keyword.")

    params = {}
    if cve_id:
        params["cveId"] = cve_id.upper().strip()
    else:
        params["keywordSearch"] = keyword.strip()
        params["resultsPerPage"] = results_per_page

    resp = requests.get(_NVD_BASE, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()
    vulns = data.get("vulnerabilities", [])
    cves  = [_parse_cve(v["cve"]) for v in vulns]

    return {"cves": cves, "total": data.get("totalResults", len(cves))}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — IP Reputation Check via AbuseIPDB
# Free tier: 1,000 checks/day. Sign up at https://www.abuseipdb.com
# Set env var: ABUSEIPDB_API_KEY=your_key
# API docs: https://docs.abuseipdb.com
# ═══════════════════════════════════════════════════════════════════════════════

_ABUSEIPDB_BASE = "https://api.abuseipdb.com/api/v2"
_MALICIOUS_THRESHOLD = 25   # confidence score % above which we flag as malicious


def check_ip_reputation(
    ip_address: str,
    max_age_days: int = 30,
) -> dict:
    """
    Check whether an IP address has been reported for abusive behaviour.

    Uses AbuseIPDB's confidence score (0–100). Scores above 25 are flagged
    as likely malicious — adjust _MALICIOUS_THRESHOLD to tune sensitivity.

    Args:
        ip_address:   IPv4 or IPv6 address to check.
        max_age_days: How far back to look for abuse reports (default 30 days).

    Returns:
        {
          "ip":               str,
          "is_malicious":     bool,
          "confidence_score": int  (0–100),
          "total_reports":    int,
          "country":          str  (ISO country code),
          "isp":              str,
          "domain":           str,
          "usage_type":       str,
          "last_reported":    str  (ISO datetime or None),
        }

    Raises:
        EnvironmentError:   If ABUSEIPDB_API_KEY is not set.
        requests.HTTPError: On API errors.
    """

    if not ABUSEIPDB_KEY:
        raise EnvironmentError(
            "ABUSEIPDB_API_KEY environment variable is not set. "
            "Sign up free at https://www.abuseipdb.com"
        )

    resp = requests.get(
        f"{_ABUSEIPDB_BASE}/check",
        headers={"Key": ABUSEIPDB_KEY, "Accept": "application/json"},
        params={"ipAddress": ip_address.strip(), "maxAgeInDays": max_age_days},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()

    d = resp.json()["data"]
    return {
        "ip":               d["ipAddress"],
        "is_malicious":     d["abuseConfidenceScore"] >= _MALICIOUS_THRESHOLD,
        "confidence_score": d["abuseConfidenceScore"],
        "total_reports":    d["totalReports"],
        "country":          d.get("countryCode", "N/A"),
        "isp":              d.get("isp", "N/A"),
        "domain":           d.get("domain", "N/A"),
        "usage_type":       d.get("usageType", "N/A"),
        "last_reported":    d.get("lastReportedAt"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Pwned Password Check via Have I Been Pwned (HIBP)
# Completely free, no API key. Uses k-anonymity — password never sent over wire.
# API docs: https://haveibeenpwned.com/API/v3#PwnedPasswords
# ═══════════════════════════════════════════════════════════════════════════════

_HIBP_BASE = "https://api.pwnedpasswords.com/range"


def check_password_breach(password: str) -> dict:
    """
    Check whether a password has appeared in known data breach dumps.

    Uses k-anonymity: only the first 5 characters of the SHA-1 hash are
    sent to HIBP. The full hash never leaves the machine.

    Args:
        password: The plaintext password to check.

    Returns:
        {
          "breached":      bool,
          "times_found":   int   (0 if not breached),
          "risk_level":    str   ("none" | "low" | "medium" | "high" | "critical"),
          "recommendation": str
        }

    Raises:
        requests.HTTPError: On HIBP API errors.
    """
    sha1    = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix  = sha1[:5]
    suffix  = sha1[5:]

    resp = requests.get(
        f"{_HIBP_BASE}/{prefix}",
        headers={"Add-Padding": "true"},   # prevents traffic analysis
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()

    # Response is "HASH_SUFFIX:COUNT\r\n" per line
    counts = {}
    for line in resp.text.splitlines():
        parts = line.split(":")
        if len(parts) == 2:
            counts[parts[0].strip()] = int(parts[1].strip())

    times_found = counts.get(suffix, 0)

    if times_found == 0:
        risk_level     = "none"
        recommendation = "Password not found in known breach databases. Still follow good hygiene."
    elif times_found < 10:
        risk_level     = "low"
        recommendation = "Found in a small number of breaches. Consider changing this password."
    elif times_found < 100:
        risk_level     = "medium"
        recommendation = "Found in multiple breaches. Change this password immediately."
    elif times_found < 10_000:
        risk_level     = "high"
        recommendation = "Found in many breaches. This password is well-known to attackers — change it now."
    else:
        risk_level     = "critical"
        recommendation = "Extremely common in breach databases. Never use this password anywhere."

    return {
        "breached":       times_found > 0,
        "times_found":    times_found,
        "risk_level":     risk_level,
        "recommendation": recommendation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — Test any tool individually
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    USAGE = """
Usage:
  python tools/rag_tool.py rag     "<query>"
  python tools/rag_tool.py cve     <CVE-ID or keyword>
  python tools/rag_tool.py ip      <ip_address>
  python tools/rag_tool.py breach  <password>

Examples:
  python tools/rag_tool.py rag    "NIST CSF Govern function"
  python tools/rag_tool.py cve    CVE-2021-44228
  python tools/rag_tool.py cve    log4j
  python tools/rag_tool.py ip     1.2.3.4
  python tools/rag_tool.py breach password123
"""

    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    tool_name = sys.argv[1].lower()
    arg       = " ".join(sys.argv[2:])

    try:
        if tool_name == "rag":
            result = search_cybersec_kb(arg)
            if not result["results"]:
                print("❌ No results above threshold.")
            else:
                for i, r in enumerate(result["results"], 1):
                    print(f"\n── Result {i} ──────────────────────────────────────────")
                    print(f"📄 {r['source']}  chunk #{r['chunk_index']}  score {r['score']}")
                    print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))

        elif tool_name == "cve":
            cve_id  = arg if re.match(r"CVE-\d{4}-\d+", arg, re.I) else None
            keyword = None if cve_id else arg
            result  = cve_lookup(cve_id=cve_id, keyword=keyword)
            print(f"\n🔎 Found {result['total']} CVE(s)\n")
            for cve in result["cves"]:
                print(f"  ID       : {cve['id']}")
                print(f"  Severity : {cve['severity']}  (CVSS {cve['score']})")
                print(f"  Published: {cve['published']}")
                print(f"  Desc     : {cve['description'][:200]}...")
                print(f"  Vector   : {cve['vector']}")
                print()

        elif tool_name == "ip":
            result = check_ip_reputation(arg)
            flag   = "🚨 MALICIOUS" if result["is_malicious"] else "✅ CLEAN"
            print(f"\n{flag}  —  {result['ip']}")
            print(f"  Confidence : {result['confidence_score']}%")
            print(f"  Reports    : {result['total_reports']}")
            print(f"  Country    : {result['country']}")
            print(f"  ISP        : {result['isp']}")
            print(f"  Usage      : {result['usage_type']}")

        elif tool_name == "breach":
            result = check_password_breach(arg)
            flag   = "🚨 BREACHED" if result["breached"] else "✅ NOT FOUND"
            print(f"\n{flag}")
            print(f"  Times found : {result['times_found']:,}")
            print(f"  Risk level  : {result['risk_level'].upper()}")
            print(f"  Action      : {result['recommendation']}")

        else:
            print(f"Unknown tool: '{tool_name}'\n{USAGE}")
            sys.exit(1)

    except EnvironmentError as e:
        print(f"⚙️  Config error: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"🌐 API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)