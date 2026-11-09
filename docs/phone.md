# Does it run on a phone?

Before any app: the same binary, on the phone, in a terminal. Android with Termux, no root.
What comes out is the number nobody has yet: how long one decision takes on *your* phone,
and how much memory it holds.

```bash
# in Termux
pkg install -y git cmake clang python
git clone --recursive --shallow-submodules https://github.com/moudrkat/kneejerk.git
cd kneejerk
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --target systemone-cli -j

# 1.1 GB, on wifi
curl -L -o d1-2B-Q4_K_M.gguf https://huggingface.co/LiquidAI/d1-2B-GGUF/resolve/main/d1-2B-Q4_K_M.gguf

# one decision, timed
echo '{"state":"I was charged twice this month, please refund one of them.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}' \
  | ./build/systemone-cli -m d1-2B-Q4_K_M.gguf --time -t 4
```

While it runs, in a second Termux session:

```bash
top -n 1 | grep systemone        # RES = memory held
```

What to write down: phone model, RAM, `_wall_ms` of the second request (the first includes
the model load), RES in MB, and whether Android killed the process. Open an issue titled
"phone: <model>" with those five lines. Thread count: `-t 4` is a good start, the big cores
of most phones; more threads are often slower.

iOS: no terminal, so this needs the Swift binding (not written yet). The library builds for
Metal with `-DGGML_METAL=ON`; the plumbing into an app is the next step, see the README.

## Measured so far

| phone | RAM | yes/no, 44 tokens | choice of four, 87 tokens | RSS | threads | date |
|---|---|---|---|---|---|---|
| Google Pixel 8a (Tensor G3) | 8 GB | 464 ms | 1 100 ms | 2.1 GB with repack | 4 | 2026-11-09 |
| Google Pixel 8a, `--no-repack` | 8 GB | 1 210 ms | | 1.1 GB | 4 | 2026-11-09 |

Notes from that run: Termux needs `pkg upgrade` first (its cmake was newer than its libc++);
build with `-j2` and `termux-wake-lock`, a full `-j` build got killed; the Hugging Face download
crawled on the phone, serving the file from a laptop over wifi took minutes.
