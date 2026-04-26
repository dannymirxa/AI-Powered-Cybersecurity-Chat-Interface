"""
test_supervisor.py
──────────────────
Interactive + automated smoke tests for the multi-agent supervisor.

Run modes:
  python test_supervisor.py              # run all automated tests
  python test_supervisor.py --chat       # interactive REPL
  python test_supervisor.py --test <n>   # run a single test by number

What is tested:
  1. RAG routing          — supervisor sends KB question to rag_agent
  2. CVE routing          — supervisor sends CVE ID to threat_agent
  3. IP routing           — supervisor sends IP address to threat_agent
  4. Password audit       — supervisor sends password to audit_agent
  5. Multi-agent (CVE+KB) — threat_agent then rag_agent chained
  6. Multi-turn memory    — follow-up refers to prior turn (core memory test)
  7. Off-topic rejection  — non-cybersecurity question is declined
  8. Empty input guard    — empty string handled gracefully

NOTE on stream() API
--------------------
stream() yields (kind, value) tuples:
  ("agent", label)  — which specialist agent / tool is being used
  ("text",  chunk)  — incremental answer text

The helpers below unpack these tuples correctly.
"""

import asyncio
import argparse
import sys
import uuid
from dataclasses import dataclass, field
from typing import Optional

from supervisor_agent import build_supervisor_graph, stream


# ── ANSI colours ─────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}\u2705 {msg}{RESET}")
def fail(msg): print(f"  {RED}\u274c {msg}{RESET}")
def info(msg): print(f"  {CYAN}\u2139  {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}\u26a0  {msg}{RESET}")

def header(msg):
    bar = "\u2550" * 60
    print(f"\n{BOLD}{bar}{RESET}")
    print(f"{BOLD}  {msg}{RESET}")
    print(f"{BOLD}{bar}{RESET}")

def sub(msg):
    print(f"\n{BOLD}{CYAN}\u25b6 {msg}{RESET}")


# ── Helper: collect full text + agent labels from stream() ──────────────────

async def collect_stream(
    graph,
    messages: list[dict],
    session_id: str,
    print_live: bool = True,
) -> tuple[str, list[str]]:
    """
    Drain stream() and return (full_text, agent_labels_seen).

    stream() yields (kind, value) tuples:
      ("agent", label)  — printed inline as a dim indicator
      ("text",  chunk)  — accumulated into full_text and printed live
    """
    full_text: str       = ""
    agents_seen: list[str] = []

    async for kind, value in stream(graph, messages, session_id):
        if kind == "agent":
            if value not in agents_seen:
                agents_seen.append(value)
            if print_live:
                print(f"\n    {YELLOW}[tool] {value}{RESET}")
                print(f"    {BOLD}[ASSISTANT]{RESET} ", end="", flush=True)
        elif kind == "text":
            full_text += value
            if print_live:
                print(value, end="", flush=True)

    return full_text, agents_seen


# ── Test case definition ───────────────────────────────────────────────────────

@dataclass
class TestCase:
    number:           int
    name:             str
    description:      str
    turns:            list[tuple[str, str]]   # (role, content)
    must_contain:     list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    must_use_tool:    list[str] = field(default_factory=list)  # expected agent labels
    shared_session:   bool = False            # True = all turns share one session_id


TEST_CASES = [
    # ── 1. RAG routing ────────────────────────────────────────────────────────
    TestCase(
        number=1,
        name="RAG Routing",
        description="KB question should be handled by rag_agent using Milvus search",
        turns=[
            ("user", "What does NIST CSF 2.0 say about incident recovery planning?")
        ],
        must_contain=["nist", "recover"],
        must_use_tool=["\U0001f4da NIST / Framework Q&A"],
    ),

    # ── 2. CVE routing ────────────────────────────────────────────────────────
    TestCase(
        number=2,
        name="CVE Lookup Routing",
        description="CVE ID mention should be routed to threat_agent → NVD lookup",
        turns=[
            ("user", "What is CVE-2021-44228 and how severe is it?")
        ],
        must_contain=["cve-2021-44228", "log4"],
        must_use_tool=["\U0001f50d CVE lookup"],
    ),

    # ── 3. IP reputation routing ───────────────────────────────────────────────
    TestCase(
        number=3,
        name="IP Reputation Routing",
        description="IP address mention should be routed to threat_agent → AbuseIPDB",
        turns=[
            ("user", "Can you check if the IP address 185.220.101.1 is malicious?")
        ],
        must_contain=["185.220.101.1"],
        must_use_tool=["\U0001f310 IP reputation check"],
    ),

    # ── 4. Password audit routing ───────────────────────────────────────────────
    TestCase(
        number=4,
        name="Password Breach Routing",
        description="Password check should be routed to audit_agent → HIBP",
        turns=[
            ("user", "Has the password 'password123' appeared in any data breaches?")
        ],
        must_contain=["breach", "password"],
        must_use_tool=["\U0001f511 Password breach check"],
    ),

    # ── 5. Multi-agent chaining ───────────────────────────────────────────────
    TestCase(
        number=5,
        name="Multi-Agent Chain (CVE + Framework)",
        description="CVE question + 'what controls' should chain threat_agent → rag_agent",
        turns=[
            ("user",
             "CVE-2023-23397 was exploited in the wild — "
             "what NIST CSF controls would have helped prevent or detect it?")
        ],
        must_contain=["cve-2023-23397", "nist"],
    ),

    # ── 6. Multi-turn memory ───────────────────────────────────────────────
    TestCase(
        number=6,
        name="Multi-Turn Conversation Memory",
        description="Follow-ups should reference earlier context within the same session",
        shared_session=True,
        turns=[
            ("user", "What is the NIST CSF 'Govern' function?"),
            ("user", "Can you give me a real-world example of how an organisation implements that?"),
            ("user", "And how does it relate to the 'Identify' function?"),
        ],
        must_contain=["govern", "identify"],
    ),

    # ── 7. Off-topic rejection ────────────────────────────────────────────────
    TestCase(
        number=7,
        name="Off-Topic Rejection",
        description="Non-cybersecurity question should be politely declined",
        turns=[
            ("user", "What is the best recipe for nasi lemak?")
        ],
        must_not_contain=["coconut milk", "pandan", "sambal"],
    ),

    # ── 8. Empty input guard ────────────────────────────────────────────────
    TestCase(
        number=8,
        name="Empty Input Guard",
        description="Blank message should not crash — should return a graceful response",
        turns=[
            ("user", "   ")
        ],
    ),
]


# ── Test runner ──────────────────────────────────────────────────────────────────

async def run_test(graph, test: TestCase) -> bool:
    """
    Execute a single test case against the compiled graph.
    Returns True if the test passed, False otherwise.
    """
    sub(f"[{test.number}] {test.name}")
    info(test.description)

    session_id = str(uuid.uuid4())
    last_response = ""
    all_agents_seen: list[str] = []
    passed = True

    try:
        if test.shared_session:
            for i, (role, content) in enumerate(test.turns):
                print(f"\n    {BOLD}[USER]{RESET} {content.strip()}")
                print(f"    {BOLD}[ASSISTANT]{RESET} ", end="", flush=True)
                last_response, agents = await collect_stream(
                    graph,
                    [{"role": role, "content": content.strip()}],
                    session_id=session_id,
                    print_live=True,
                )
                all_agents_seen.extend(a for a in agents if a not in all_agents_seen)
                print()
        else:
            for role, content in test.turns:
                print(f"    {BOLD}[USER]{RESET} {content.strip()}")
                print(f"    {BOLD}[ASSISTANT]{RESET} ", end="", flush=True)
                last_response, agents = await collect_stream(
                    graph,
                    [{"role": role, "content": content.strip()}],
                    session_id=session_id,
                    print_live=True,
                )
                all_agents_seen.extend(a for a in agents if a not in all_agents_seen)
                print()

    except Exception as exc:
        fail(f"Exception raised: {exc}")
        return False

    # ── Assertions ───────────────────────────────────────────────────────────────
    response_lower = last_response.lower()

    if test.must_contain:
        for keyword in test.must_contain:
            if keyword.lower() in response_lower:
                ok(f"Contains expected keyword: '{keyword}'")
            else:
                fail(f"Missing expected keyword: '{keyword}'")
                passed = False

    if test.must_not_contain:
        for keyword in test.must_not_contain:
            if keyword.lower() not in response_lower:
                ok(f"Correctly absent: '{keyword}'")
            else:
                fail(f"Should NOT contain: '{keyword}'")
                passed = False

    if test.must_use_tool:
        for expected_label in test.must_use_tool:
            if expected_label in all_agents_seen:
                ok(f"Correct tool used: '{expected_label}'")
            else:
                fail(f"Expected tool NOT used: '{expected_label}' (got: {all_agents_seen})")
                passed = False

    if not test.must_contain and not test.must_not_contain and not test.must_use_tool:
        if last_response.strip():
            ok("Returned a non-empty response (no crash)")
        else:
            fail("Returned an empty response")
            passed = False

    return passed


async def run_all_tests(test_numbers: Optional[list[int]] = None):
    """Run all (or selected) tests and print a summary."""
    header("CyberSec Multi-Agent Supervisor — Test Suite")

    cases = TEST_CASES
    if test_numbers:
        cases = [t for t in TEST_CASES if t.number in test_numbers]
        if not cases:
            print(f"{RED}No test cases match numbers: {test_numbers}{RESET}")
            sys.exit(1)

    results = {}

    graph, client = await build_supervisor_graph()

    try:
        for test in cases:
            passed = await run_test(graph, test)
            results[test.number] = (test.name, passed)
            print()
    finally:
        pass

    # ── Summary ─────────────────────────────────────────────────────────────────
    header("Results Summary")
    total  = len(results)
    n_pass = sum(1 for _, (_, p) in results.items() if p)
    n_fail = total - n_pass

    for num, (name, p) in results.items():
        status = f"{GREEN}PASS{RESET}" if p else f"{RED}FAIL{RESET}"
        print(f"  [{status}]  {num}. {name}")

    print()
    print(f"  {BOLD}Total: {total}  |  Passed: {GREEN}{n_pass}{RESET}  |  Failed: {RED}{n_fail}{RESET}{BOLD}{RESET}")

    if n_fail:
        sys.exit(1)


async def interactive_chat():
    """Simple multi-turn REPL for manual testing."""
    header("CyberSec AI — Interactive Test Chat")
    print("  Type your message and press Enter.")
    print("  Type 'new' for a new session, 'exit' or Ctrl-C to quit.\n")

    session_id = str(uuid.uuid4())
    info(f"Session ID: {session_id}")

    graph, client = await build_supervisor_graph()

    while True:
        try:
            user_input = input(f"\n{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break
        if user_input.lower() == "new":
            session_id = str(uuid.uuid4())
            info(f"New session started: {session_id}")
            continue

        print(f"\n{BOLD}Assistant:{RESET} ", end="", flush=True)
        try:
            async for kind, value in stream(
                graph,
                [{"role": "user", "content": user_input}],
                session_id=session_id,
            ):
                if kind == "agent":
                    print(f"\n  {YELLOW}[tool] {value}{RESET}\n", flush=True)
                elif kind == "text":
                    print(value, end="", flush=True)
        except Exception as exc:
            print(f"\n{RED}Error: {exc}{RESET}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Supervisor test suite / interactive chat")
    parser.add_argument("--chat", action="store_true", help="Launch interactive REPL")
    parser.add_argument("--test", type=int, nargs="+", metavar="N",
                        help="Run specific test(s) by number (e.g. --test 1 6)")
    args = parser.parse_args()

    if args.chat:
        asyncio.run(interactive_chat())
    else:
        asyncio.run(run_all_tests(args.test))


if __name__ == "__main__":
    main()
