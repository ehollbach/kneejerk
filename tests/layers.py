#!/usr/bin/env python3
"""The layer readout: one request with "layers": true through systemone-cli, the answer of every layer as a table.

    python3 tests/layers.py --cli build/systemone-cli --model d1-2B-Q4_K_M.gguf [--ngl 0] [--threads 8]

The last row (the last layer through the output head) should match the final answer: the difference is printed.
"""
import argparse
import json
import subprocess

CASES = [
    {
        "state": "will it rain in Brno tomorrow? ah never mind, I'll just take the umbrella",
        "questions": {
            "act": {"type": "choice", "instructions": "What should the agent do next?", "criteria": {
                "reply": "no tool call, just reply to the user",
                "weather": "call the weather API",
                "clarify": "ask the user what they meant",
            }},
            "wants_info": {"type": "noul", "instructions": "Does the user still want information?"},
        },
        "layers": True,
    },
    {
        "state": "I was charged twice this month, please refund one of them.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                                  "fraud": "Suspected unauthorised use"}},
        },
        "layers": True,
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    cmd = [a.cli, "-m", a.model, "-ngl", str(a.ngl), "--time"]
    if a.threads:
        cmd += ["-t", str(a.threads)]
    stdin = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in CASES)
    out = subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=True).stdout
    for body, line in zip(CASES, out.strip().splitlines()):
        ans = json.loads(line)
        if "error" in ans:
            print("error:", ans["error"])
            continue
        print(f"== {ans['usage']['input_tokens']} tokens · {ans.get('_wall_ms', 0):.0f} ms")
        for qid, q in ans["answers"].items():
            layers = q.get("layers", {})
            if "error" in layers:
                print(f"  {qid}: {layers['error']}")
                continue
            rows = layers["probabilities"]
            if q["type"] == "noul":
                final = q["noul"]
                print(f"  {qid} (noul) final {final:.3f}")
                print("    " + " ".join(f"{p:.2f}" if p is not None else "  - " for p in rows))
                last = next((p for p in reversed(rows) if p is not None), None)
            else:
                opts = layers["options"]
                final = q["probabilities"]
                print(f"  {qid} ({q['type']}) final " + " ".join(f"{o} {final[o]:.3f}" for o in opts))
                for i, o in enumerate(opts):
                    print(f"    {o:>10} " + " ".join(f"{r[i]:.2f}" if r else "  - " for r in rows))
                last = next((r for r in reversed(rows) if r), None)
                last = {o: last[i] for i, o in enumerate(opts)} if last else None
                final = [final[o] for o in opts]
                last = [last[o] for o in opts] if last else None
            if last is not None:
                if isinstance(final, list):
                    diff = max(abs(x - y) for x, y in zip(final, last))
                else:
                    diff = abs(final - last)
                print(f"    last layer vs final: {diff:.4f}")


if __name__ == "__main__":
    main()
