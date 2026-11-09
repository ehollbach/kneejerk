"""Parity with llama-server: the same requests through systemone-cli and through /v1/systemone,
the probabilities must agree.

  python3 tests/parity.py --cli build/systemone-cli --model d1-2B-Q4_K_M.gguf --server http://127.0.0.1:8081/v1

Prints every question's probabilities side by side and the largest difference; exits 1 above --tol.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

CASES = [
    {
        "state": "I was charged twice this month, please refund one of them.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                                  "fraud": "Suspected unauthorised use"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "Today", "Blocking the customer now"]},
        },
    },
    {
        "state": "Petr: Sterilizace je 134 stupňů.\nJana: To číslo mi nesedí, nebylo to 121?",
        "questions": {
            "kind": {"type": "choice", "instructions": "What does the LAST message do?", "criteria": {
                "question": "asks a question about the subject",
                "request": "asks for something to be done",
                "doubt": "doubts or corrects something said",
                "stuck": "says they are stuck or lost",
                "chatter": "agreement, thanks, a joke, small talk",
            }},
            "named": {"type": "noul", "instructions": "Does the last message name a person?"},
        },
    },
    {
        "state": {"ticket": "The app crashes when I open the camera on Android 15.", "customer_tier": "pro"},
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges and refunds", "technical": "App or site faults"}},
            "severity": {"type": "score", "instructions": "How severe is it?",
                         "criteria": ["Cosmetic", "Annoying", "Blocking a feature", "Blocking the app"]},
        },
    },
    {
        "state": "Dneska je hezky, jdu si dát kafe.",
        "questions": {
            "asks": {"type": "noul", "instructions": "Does the message ask anyone for anything?"},
        },
    },
]


def via_server(url: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(url.rstrip("/") + "/systemone", json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r), (time.time() - t0) * 1000


def via_cli(cli: str, model: str, bodies: list[dict], extra: list[str]) -> list[dict]:
    inp = "\n".join(json.dumps(b, ensure_ascii=False) for b in bodies) + "\n"
    p = subprocess.run([cli, "-m", model, "--time", *extra], input=inp.encode(), capture_output=True, timeout=600)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode(errors="replace"))
        raise SystemExit(f"cli failed ({p.returncode})")
    return [json.loads(line) for line in p.stdout.decode().splitlines() if line.strip()]


def probs(answer: dict) -> list[float]:
    if "probabilities" in answer:
        pr = answer["probabilities"]
        return list(pr.values()) if isinstance(pr, dict) else list(pr)
    if "noul" in answer:
        return [answer["noul"]]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default=None, help="/v1 of a llama-server with the same model")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--ngl", default=None)
    ap.add_argument("--threads", default=None)
    a = ap.parse_args()

    extra = (["-ngl", a.ngl] if a.ngl is not None else []) + (["-t", a.threads] if a.threads is not None else [])
    ours = via_cli(a.cli, a.model, CASES, extra)
    worst = 0.0
    for i, (body, mine) in enumerate(zip(CASES, ours)):
        if "error" in mine:
            print(f"case {i}: cli error {mine['error']}")
            worst = 1.0
            continue
        theirs = via_server(a.server, body)[0] if a.server else None
        print(f"== case {i} · cli {mine.get('_wall_ms', 0):.0f} ms · {mine['usage']['input_tokens']} tokens")
        for qid, ans in mine["answers"].items():
            p_cli = probs(ans)
            line = f"  {qid:10s} cli {['%.4f' % p for p in p_cli]}"
            if theirs:
                p_srv = probs(theirs["answers"][qid])
                diff = max((abs(x - y) for x, y in zip(p_cli, p_srv)), default=1.0)
                worst = max(worst, diff)
                line += f"  server {['%.4f' % p for p in p_srv]}  diff {diff:.4f}"
            print(line)
    if a.server:
        print(f"\nlargest difference: {worst:.4f} ({'OK' if worst <= a.tol else 'ABOVE TOLERANCE'} {a.tol})")
        raise SystemExit(0 if worst <= a.tol else 1)


if __name__ == "__main__":
    main()
