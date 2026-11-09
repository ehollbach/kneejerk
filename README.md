# kneejerk

I want Jev-like decision models in the device, natively, and I want it fast, because it is cool.
Not behind a server, not through an HTTP port on localhost: inside the app, on the phone, in a
Capacitor plugin. Nobody has done that yet, so there is nothing left for me but to do it.

A decision model doesn't write. It takes a *state* (text) and typed *questions* (yes/no, a
choice, a score) and returns calibrated probabilities in one forward pass. No tokens generated,
no hallucinated prose, no broken JSON, milliseconds instead of seconds. TypeSafe made the
format (`/v1/systemone`), Liquid AI's open **d1** and a dozen other families speak it, and
llama.cpp runs them all. But only in `llama-server`: the code that turns questions into a
prompt and logits into probabilities lives in the server, not in the library an app links.

kneejerk is that code as a library. The same C++ llama.cpp uses, with the server cut away.

## Try it in five minutes

You need cmake, a C++17 compiler and a decision model in GGUF. Linux and macOS for now.

```bash
git clone --recursive https://github.com/moudrkat/kneejerk.git
cd kneejerk
cmake -B build -DCMAKE_BUILD_TYPE=Release    # add -DGGML_CUDA=ON or -DGGML_METAL=ON for a GPU
cmake --build build --target systemone-cli -j

# a 1.1 GB decision model
curl -L -o d1-2B-Q4_K_M.gguf https://huggingface.co/LiquidAI/d1-2B-GGUF/resolve/main/d1-2B-Q4_K_M.gguf

echo '{"state": "I was charged twice this month, please refund one of them.",
       "questions": {
         "refund":  {"type": "noul",   "instructions": "Is the customer asking for a refund?"},
         "team":    {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                                  "fraud": "Suspected unauthorised use"}},
         "urgency": {"type": "score",  "instructions": "How urgent is this?",
                     "criteria": ["Can wait", "Today", "Blocking the customer now"]}}}' \
  | tr -d '\n' | ./build/systemone-cli -m d1-2B-Q4_K_M.gguf --time
```

```json
{"model":"d1-2B-Q4_K_M.gguf",
 "answers":{"refund":{"type":"noul","noul":0.9974},
            "team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9631,"technical":0.0173,"fraud":0.0197},"confidence":0.9631},
            "urgency":{"type":"score","score":0.0,"probabilities":{"0":0.7610,"1":0.2165,"2":0.0224},"confidence":0.7610}},
 "usage":{"input_tokens":160,"output_tokens":0}, "_wall_ms": 678}
```

That is 160 tokens in, zero out, on eight CPU threads. The request and the answer are exactly
what `llama-server`'s `/v1/systemone` takes and returns, so anything written against the server
works against the library.

## Use it from code

```c
#include "systemone.h"

systemone_params p = systemone_default_params();   // n_gpu_layers -1, n_ctx 4096
systemone * so = systemone_load("d1-2B-Q4_K_M.gguf", &p);
char * answer = systemone_decide(so, request_json);  // JSON in, JSON out
puts(answer);
systemone_free_string(answer);
systemone_free(so);
```

Link `libsystemone` (it brings `llama` with it). Six functions, C ABI, so Swift, Kotlin/JNI,
Dart and Rust bindings are thin. The Capacitor plugin that started this comes next.

## Which models

Whatever llama.cpp b12207 knows as a decision model: Liquid AI d1 (LFM3 and LFM2, text and
omni), OpenJev, IBM Granite Switch, Lev, Kev, Nimble, Laya, Clef, Perplexity decider, vLLM
decision2 and d3, H2O Lightning, MiniCPM OCR Jev. The library reads the family from the GGUF
metadata and builds that family's prompt; nothing is d1-specific here.

**Tested against `llama-server` b12207, same GGUF, CPU vs CPU** (`tests/parity.py`, the four
requests in it, every decision identical; the number is the largest probability difference):

| family | model | path through the code | largest diff |
|---|---|---|---|
| lfm3-d1 | LiquidAI/d1-2B-GGUF Q4_K_M | logits of option tokens | 0.018 |
| laya | ggml-org/Laya-GGUF Q8_0 | embeddings at option markers | 0.0095 |
| h2o-lightning | h2oai/h2o-lightning-4b-gguf IQ3_XXS | logits, H2O's own rendering | 0.0115 |
| clef | bartowski/Cloudflare_clef-flash-GGUF Q2_K | joint head, all questions in one prompt | 0.035 |

Perplexity's own `pplx-decider-flash-gguf-v1.2` carries no decision metadata
(`<arch>.decision.type`, `tokenizer.chat_template.systemone`), so neither llama-server nor this
library treats it as a decision model; a re-export with the metadata would run. OpenJev, Granite
Switch, Lev, Kev, Nimble, vLLM decision2/d3 and MiniCPM OCR Jev compile and are untested here: I
found no GGUF, or none small enough to run on a CPU box tonight. If you have one, run
`tests/parity.py` against a `llama-server` with the same file and open an issue with the output,
pass or fail. That is the most useful thing you can do for this repo right now.

Images, audio and video are not in yet: a request with files gets an error and a pointer to
`llama-server`.

## Does it run on a phone?

[docs/phone.md](docs/phone.md): the same binary in Termux, timed, before any app is touched.
Numbers from real phones are what this repo needs most; open an issue with yours.

## Parity

`tests/parity.py` sends the same requests to `systemone-cli` and to `/v1/systemone` and
compares the probabilities. On every family tested the decisions are identical and the probabilities
agree within 0.035 (most within 0.005). The remaining difference is deterministic and not yet
explained; it is issue #1. `tests/accuracy.py` runs both on
`tasksource/tasksource-jev-typed-decisions` (a million typed decisions with answers) and reports
accuracy, agreement and the largest difference.

## How it is built

`src/upstream/` is `tools/server/server-decision.{h,cpp}` from llama.cpp b12207, unchanged, so
it diffs cleanly against the next release. `src/shim/` gives it the dozen server types it
expects (a task, its tokens, a JSON helper) without the server. `src/systemone.cpp` does what
the server's loop and its `/v1/systemone` handler do around that code: one task per question
variant, the prompt through the model in one batch, the scores read back the way
`send_decision()` reads them, then `format_answer()`.

llama.cpp is a submodule pinned to b12207. MIT, like llama.cpp.
