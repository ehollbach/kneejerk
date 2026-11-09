"""Accuracy on tasksource/tasksource-jev-typed-decisions, through the library and (optionally) through
llama-server: does the library decide as well as the server, on real decision data, not four examples.

  python3 tests/accuracy.py --cli build/systemone-cli --model d1-2B-Q4_K_M.gguf --n 200 \
      [--server http://127.0.0.1:8086/v1] [--config default --split validation]

Rows come from the Hugging Face datasets-server API (no download, 100 rows per call). Each row is one
typed question (choice / noul / score) over a text state with a target distribution; the answer is right
when it picks the target's argmax. Printed: accuracy of each side, agreement between them, and the
largest probability difference.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request

DATASET = "tasksource/tasksource-jev-typed-decisions"


def rows(config: str, split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": DATASET, "config": config, "split": split, "offset": offset, "length": length})
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=120) as r:
        return [x["row"] for x in json.load(r)["rows"]]


def request_of(row: dict) -> dict | None:
    """The row as a /v1/systemone body with one question, or None for a kind this test does not run."""
    kind = row["kind"]
    opts = row["options"] or []
    q: dict
    if kind == "choice":
        if len(opts) < 2:
            return None
        q = {"type": "choice", "instructions": row["question"],
             "criteria": {f"o{i}": (o if isinstance(o, str) else json.dumps(o, ensure_ascii=False)) for i, o in enumerate(opts)}}
    elif kind == "noul":
        q = {"type": "noul", "instructions": row["question"]}
    elif kind == "score":
        if len(opts) < 2:
            return None
        q = {"type": "score", "instructions": row["question"], "criteria": [str(o) for o in opts]}
    else:
        return None
    return {"state": row["state"], "questions": {"q": q}}


def target_index(row: dict) -> int | None:
    t = row["target"]
    if not t:
        return None
    if row["kind"] == "noul":
        return 1 if float(t[0]) >= 0.5 else 0  # 1 = yes
    return max(range(len(t)), key=lambda i: float(t[i]))


def answer_index(ans: dict, kind: str) -> tuple[int, list[float]]:
    if kind == "noul":
        p = float(ans["noul"])
        return (1 if p >= 0.5 else 0), [p]
    probs = ans["probabilities"]
    vals = list(probs.values()) if isinstance(probs, dict) else list(probs)
    return max(range(len(vals)), key=lambda i: vals[i]), vals


def via_cli(cli: str, model: str, bodies: list[dict], extra: list[str]) -> list[dict]:
    inp = "\n".join(json.dumps(b, ensure_ascii=False) for b in bodies) + "\n"
    p = subprocess.run([cli, "-m", model, *extra], input=inp.encode(), capture_output=True, timeout=3600)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode(errors="replace")[-2000:])
        raise SystemExit(f"cli failed ({p.returncode})")
    return [json.loads(line) for line in p.stdout.decode().splitlines() if line.strip()]


def via_server(url: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/systemone", json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--config", default="default")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-state-chars", type=int, default=6000, help="skip longer states (context)")
    ap.add_argument("--ngl", default=None)
    ap.add_argument("--threads", default=None)
    a = ap.parse_args()

    picked: list[tuple[dict, dict]] = []
    offset = a.offset
    while len(picked) < a.n:
        batch = rows(a.config, a.split, offset, 100)
        if not batch:
            break
        for row in batch:
            if len(row["state"] or "") > a.max_state_chars or target_index(row) is None:
                continue
            body = request_of(row)
            if body:
                picked.append((row, body))
            if len(picked) >= a.n:
                break
        offset += 100
    print(f"{len(picked)} questions from {DATASET} [{a.config}/{a.split}], kinds: "
          + ", ".join(f"{k} {sum(1 for r, _ in picked if r['kind'] == k)}" for k in ("choice", "noul", "score")))

    extra = (["-ngl", a.ngl] if a.ngl is not None else []) + (["-t", a.threads] if a.threads is not None else [])
    t0 = time.time()
    ours = via_cli(a.cli, a.model, [b for _, b in picked], extra)
    cli_s = time.time() - t0

    ok_cli = ok_srv = agree = n = 0
    worst = 0.0
    srv_s = 0.0
    for (row, body), mine in zip(picked, ours):
        if "error" in mine:
            continue
        n += 1
        idx, p_cli = answer_index(mine["answers"]["q"], row["kind"])
        tgt = target_index(row)
        ok_cli += idx == tgt
        if a.server:
            t1 = time.time()
            theirs = via_server(a.server, body)
            srv_s += time.time() - t1
            jdx, p_srv = answer_index(theirs["answers"]["q"], row["kind"])
            ok_srv += jdx == tgt
            agree += idx == jdx
            worst = max(worst, max(abs(x - y) for x, y in zip(p_cli, p_srv)))

    print(f"library: {ok_cli}/{n} right ({100 * ok_cli / max(n, 1):.1f} %) · {cli_s / max(n, 1) * 1000:.0f} ms per question")
    if a.server:
        print(f"server:  {ok_srv}/{n} right ({100 * ok_srv / max(n, 1):.1f} %) · {srv_s / max(n, 1) * 1000:.0f} ms per question")
        print(f"agreement library vs server: {agree}/{n} · largest probability difference {worst:.4f}")


if __name__ == "__main__":
    main()
