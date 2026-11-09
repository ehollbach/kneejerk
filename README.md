# kneejerk

I want Jev-like decision models in the device, natively, and I want it fast, because it is cool.
Not behind a server, not through an HTTP port on localhost: inside the app, on the phone, in a
Capacitor plugin. Decisions on a device exist in other shapes: in the browser with Transformers.js
([open-jev](https://github.com/nico-martin/open-jev)), on a Mac through Core ML with one model
converted by hand ([FluidUse](https://github.com/FluidInference/FluidUse)), offline ONNX runtimes
([edgejev](https://github.com/yzfly/edgejev)). What nobody had is the route that is both native and general: a C
library linked into the app itself, taking every decision model llama.cpp already speaks straight
from its GGUF, no conversion, one model or the next. So there is nothing left for me but to do it.

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

Link `libsystemone` (it brings `llama` with it). A handful of functions, C ABI, so Swift, Kotlin/JNI,
Dart and Rust bindings are thin. The Capacitor plugin that started this comes next.

## Chat models too

A model is loaded for decisions by default. Any GGUF with a chat template can be loaded for text
generation instead: `params.model = SYSTEMONE_MODEL_CHAT`, then `systemone_chat()` with an
OpenAI-style `{"messages": [...]}` body. From C++ (`kneejerk.hpp`) it is one argument:

```cpp
#include "kneejerk.hpp"

kneejerk::Assistant judge("d1-2B-Q4_K_M.gguf");                               // Model::Decision
kneejerk::Assistant talker("qwen1_5-0_5b-chat-q2_k.gguf", kneejerk::Model::Chat);

std::string answer = judge.decide(request_json);
std::string reply  = talker.chat("Summarise my last message");
```

On the command line: `echo 'Hello, who are you?' | ./build/systemone-cli -m MODEL.gguf --chat --time`.
The answer carries `usage` and `timings` (`prompt_ms`, `generation_ms`). A base model without chat
training can continue raw text instead: `{"prompt": "The quick brown fox", "stop": "."}` skips the chat
template, answers with `text`, and ends at the first stop string, which is cut from the text and named
in `stop_sequence` (llama.cpp's reverse prompt, OpenAI's `stop`). Control-token texts in the
messages are broken with a zero-width space, so a message cannot open a turn of its own at the token
level. That is not a defence against prompt injection: a small model follows "reply with PWNED" written
in plain text just as well, so a reply to untrusted text (an SMS, an email) must not trigger anything on
its own. `SYSTEMONE_DUMP=1` prints the rendered prompt.

## Which models

Whatever llama.cpp b12207 knows as a decision model: Liquid AI d1 (LFM3 and LFM2, text and
omni), OpenJev, IBM Granite Switch, Lev, Kev, Nimble, Laya, Clef, Perplexity decider, vLLM
decision2 and d3, H2O Lightning, MiniCPM OCR Jev. The library reads the family from the GGUF
metadata and builds that family's prompt; nothing is d1-specific here.

**Tested against the prebuilt `llama-server` b12207, same GGUF, CPU vs CPU** (`tests/parity.py`,
the four requests in it, every decision identical; the number is the largest probability
difference, all of it CPU-kernel rounding, see Parity below):

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

Yes. First measurement, 9 November 2026, the binary built in Termux, d1-2B Q4_K_M, four threads:

| phone | RAM | yes/no, 44 tokens | choice of four, 87 tokens | process RSS |
|---|---|---|---|---|
| Google Pixel 8a (Tensor G3) | 8 GB | 464 ms | 1 100 ms | 2.1 GB (1.1 GB model mapped + a repacked copy) |
| the same with `--no-repack` | | 1 210 ms | | 1.1 GB |

Same probabilities as on the desktop to three decimals. Half a second per decision is five times a
desktop CPU, and still before a message would reach a server. Repacking the weights for the ARM
kernels costs a second copy of the model in RAM and buys 2.6× speed; pick per device. [docs/phone.md](docs/phone.md) has the
recipe; numbers from other phones are what this repo needs most, open an issue with yours.

## Parity

`tests/parity.py` sends the same requests to `systemone-cli` and to `/v1/systemone` and
compares the probabilities. Against a `llama-server` built from the submodule with the same
flags, every probability matches to four decimals (d1-2B, CPU). Against the prebuilt release
binary they differ by up to 0.035: it ships fourteen CPU backend variants and picks one at
runtime, and those kernels round differently. The decisions are the same either way.
`tests/accuracy.py` runs both on `tasksource/tasksource-jev-typed-decisions` (a million typed
decisions with answers): on 200 validation questions d1-2B Q4_K_M answers 119/200 through the
library and 119/200 through the server, 200/200 the same answer, largest difference 0.0000.

## How it is built

`src/upstream/` is `tools/server/server-decision.{h,cpp}` from llama.cpp b12207, unchanged, so
it diffs cleanly against the next release. `src/shim/` gives it the dozen server types it
expects (a task, its tokens, a JSON helper) without the server. `src/systemone.cpp` does what
the server's loop and its `/v1/systemone` handler do around that code: one task per question
variant, the prompt through the model in one batch, the scores read back the way
`send_decision()` reads them, then `format_answer()`.

llama.cpp is a submodule pinned to b12207. MIT, like llama.cpp. Whether this belongs upstream, in
`common/` or libllama, is [discussion #31938](https://github.com/ggml-org/llama.cpp/discussions/31938).
