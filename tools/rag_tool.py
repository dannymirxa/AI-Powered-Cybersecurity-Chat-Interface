"""
tools/rag_tool.py
─────────────────
All cybersecurity tools in one place — pure functions, no MCP, no server.
Every URL, port, model name, collection name, API key, and threshold is
readable from environment variables (see .env.example for the full list).

Tools:
  1. search_cybersec_kb     — RAG: semantic search over Milvus knowledge base
  2. cve_lookup             — CVE details from NIST NVD (no API key needed)
  3. check_ip_reputation    — IP abuse score from AbuseIPDB (free API key)
  4. check_password_breach  — Pwned password check via HIBP k-anonymity (no key)

CLI test:
  python tools/rag_tool.py rag     "What is NIST CSF Govern function?"
  python tools/rag_tool.py cve     CVE-2021-44228
  python tools/rag_tool.py ip      8.8.8.8
  python tools/rag_tool.py breach  password123
"""

import hashlib
import os
import re
import requests
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

# ── Shared config (all overridable via env / .env) ────────────────────────────
OLLAMA_URL      = os.getenv("OLLAMA_URL",          "http://localhost:11434")
MILVUS_URI      = os.getenv("MILVUS_URI",          "http://localhost:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME",     "cybersec_kb")
EMBED_MODEL     = os.getenv("EMBED_MODEL",          "nomic-embed-text")
ABUSEIPDB_KEY   = os.getenv("ABUSEIPDB_API_KEY",   "")

# Timeouts
DEFAULT_TIMEOUT  = int(os.getenv("HTTP_TIMEOUT",   "15"))    # seconds
EMBED_TIMEOUT    = int(os.getenv("EMBED_TIMEOUT",  "60"))    # embedding can be slow

# External API base URLs (override to point at proxies / mirrors)
_NVD_BASE        = os.getenv("NVD_BASE_URL",        "https://services.nvd.nist.gov/rest/json/cves/2.0")
_ABUSEIPDB_BASE  = os.getenv("ABUSEIPDB_BASE_URL",  "https://api.abuseipdb.com/api/v2")
_HIBP_BASE       = os.getenv("HIBP_BASE_URL",       "https://api.pwnedpasswords.com/range")

# Tunable thresholds
_DEFAULT_TOP_K        = int(os.getenv("RAG_TOP_K",             "5"))
_SCORE_THRESHOLD      = float(os.getenv("RAG_SCORE_THRESHOLD", "0.60"))
_MALICIOUS_THRESHOLD  = int(os.getenv("IP_MALICIOUS_THRESHOLD", "25"))  # AbuseIPDB confidence %


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_embedding(text: str) -> list[float]:
    """Embed text using Ollama /api/embeddings."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=EMBED_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def get_milvus_client() -> MilvusClient:
    """Return a fresh MilvusClient. Caller is responsible for .close()."""
    return MilvusClient(uri=MILVUS_URI)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — RAG: Semantic search over Milvus knowledge base
# ═══════════════════════════════════════════════════════════════════════════════

def search_cybersec_kb(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    score_threshold: float = _SCORE_THRESHOLD,
    client: MilvusClient = None,
) -> dict:
    """
    Semantic search over the cybersecurity knowledge base in Milvus.

    Milvus COSINE metric returns `distance` as cosine SIMILARITY (0–1),
    so we use it directly as the score — no inversion needed.
    Chunks with score < score_threshold are discarded.

    Args:
        query:           Natural-language search string.
        top_k:           Maximum chunks to return (capped at 10). Default: RAG_TOP_K.
        score_threshold: Minimum cosine similarity (0–1). Default: RAG_SCORE_THRESHOLD.
        client:          Optional pre-connected MilvusClient.

    Returns:
        {
          "query":   str,
          "results": [{ text, source, chunk_index, score }, ...]
                     sorted best-first, score is cosine similarity (0–1)
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
            score = round(hit["distance"], 4)
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
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_cve(cve: dict) -> dict:
    descriptions = cve.get("descriptions", [])
    english_desc = next(
        (d["value"] for d in descriptions if d.get("lang") == "en"),
        "No English description available.",
    )
    metrics  = cve.get("metrics", {})
    severity = score = vector = "N/A"
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            cvss_data = entries[0].get("cvssData", {})
            severity  = cvss_data.get("baseSeverity", "N/A")
            score     = cvss_data.get("baseScore",    "N/A")
            vector    = cvss_data.get("vectorString",  "N/A")
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
    Look up CVE details from the NIST NVD.
    Base URL is overridable via NVD_BASE_URL env var.
    """
    if not cve_id and not keyword:
        raise ValueError("Provide either cve_id or keyword.")

    params = {}
    if cve_id:
        params["cveId"] = cve_id.upper().strip()
    else:
        params["keywordSearch"]  = keyword.strip()
        params["resultsPerPage"] = results_per_page

    resp = requests.get(_NVD_BASE, params=params, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()

    data  = resp.json()
    vulns = data.get("vulnerabilities", [])
    cves  = [_parse_cve(v["cve"]) for v in vulns]
    return {"cves": cves, "total": data.get("totalResults", len(cves))}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — IP Reputation Check via AbuseIPDB
# ═══════════════════════════════════════════════════════════════════════════════

def check_ip_reputation(
    ip_address: str,
    max_age_days: int = 30,
) -> dict:
    """
    Check IP reputation via AbuseIPDB.
    Base URL overridable via ABUSEIPDB_BASE_URL.
    Malicious threshold overridable via IP_MALICIOUS_THRESHOLD (default 25).
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
# TOOL 4 — Pwned Password Check via Have I Been Pwned
# ═══════════════════════════════════════════════════════════════════════════════

def check_password_breach(password: str) -> dict:
    """
    Check whether a password appeared in known breach dumps.
    Base URL overridable via HIBP_BASE_URL.
    Password is never sent — only 5-char SHA-1 prefix (k-anonymity).
    """
    sha1   = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix = sha1[:5]
    suffix = sha1[5:]

    resp = requests.get(
        f"{_HIBP_BASE}/{prefix}",
        headers={"Add-Padding": "true"},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()

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
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    USAGE = """
Usage:
  python tools/rag_tool.py rag     "<query>"
  python tools/rag_tool.py cve     <CVE-ID or keyword>
  python tools/rag_tool.py ip      <ip_address>
  python tools/rag_tool.py breach  <password>

Examples:
  python tools/rag_tool.py rag    "NIST CSF Govern function"
  python tools/rag_tool.py cve    CVE-2021-44228
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
                print(f"No results above threshold ({_SCORE_THRESHOLD}).")
            else:
                for i, r in enumerate(result["results"], 1):
                    print(f"\n── Result {i} ─────────────────────")
                    print(f"{r['source']}  chunk #{r['chunk_index']}  score {r['score']}")
                    print(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))

        elif tool_name == "cve":
            cve_id  = arg if re.match(r"CVE-\d{4}-\d+", arg, re.I) else None
            keyword = None if cve_id else arg
            result  = cve_lookup(cve_id=cve_id, keyword=keyword)
            print(f"\nFound {result['total']} CVE(s)\n")
            for cve in result["cves"]:
                print(f"  {cve['id']}  {cve['severity']} (CVSS {cve['score']})")
                print(f"  {cve['description'][:200]}...\n")

        elif tool_name == "ip":
            result = check_ip_reputation(arg)
            flag   = "MALICIOUS" if result["is_malicious"] else "CLEAN"
            print(f"\n{flag}  {result['ip']}  confidence {result['confidence_score']}%")

        elif tool_name == "breach":
            result = check_password_breach(arg)
            flag   = "BREACHED" if result["breached"] else "NOT FOUND"
            print(f"\n{flag}  times={result['times_found']:,}  risk={result['risk_level'].upper()}")

        else:
            print(f"Unknown tool: '{tool_name}'\n{USAGE}")
            sys.exit(1)

    except EnvironmentError as e:
        print(f"Config error: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"API error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
