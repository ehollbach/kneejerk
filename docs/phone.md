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
