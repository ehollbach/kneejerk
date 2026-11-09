// libsystemone: decision models (TypeSafe /v1/systemone) as a library, on llama.cpp.
//
// One request is the JSON body llama-server's /v1/systemone takes: a "state" and typed "questions"
// (noul, choice, score). The answer is the JSON llama-server returns: calibrated probabilities per
// question, no token generated. Text only: images and audio are left to the server for now.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct systemone systemone;

// what a model is loaded for
typedef enum systemone_model {
    SYSTEMONE_MODEL_DECISION = 0, // typed questions in, probabilities out: systemone_decide() (default)
    SYSTEMONE_MODEL_CHAT     = 1, // text generation with the chat template of the model: systemone_chat()
} systemone_model;

typedef struct systemone_params {
    int32_t n_gpu_layers;   // layers on the GPU, 0 = CPU only, -1 = all
    uint32_t n_ctx;         // longest prompt in tokens (state + question + options, or chat + reply); 0 = 4096
    int32_t n_threads;      // CPU threads; 0 = hardware default
    bool verbose;           // llama.cpp logs to stderr
    bool repack;            // CPU weight repacking: faster kernels, but the weights are held twice in RAM (default true)
    systemone_model model;  // SYSTEMONE_MODEL_DECISION (default) or SYSTEMONE_MODEL_CHAT
} systemone_params;

systemone_params systemone_default_params(void);

// Loads a GGUF decision model. NULL on failure, the reason is printed to stderr (there is no handle
// to ask systemone_last_error() about).
systemone * systemone_load(const char * model_path, const systemone_params * params);
void systemone_free(systemone * so);

// The decision model family llama.cpp detected, e.g. "lfm3_d1", "openjev"; "none" for a model that is not one;
// "chat" for a model loaded with SYSTEMONE_MODEL_CHAT.
const char * systemone_model_type(const systemone * so);

// Answers one request. Returns a malloc'd JSON string: {"model", "answers", "usage"} as the server, or
// {"error": {"message", "type"}}. Free it with systemone_free_string(). NULL only if out of memory.
// Safe to call from several threads on the same handle: requests run one at a time.
char * systemone_decide(systemone * so, const char * request_json);

// Chat mode: the reply to a conversation. The request is {"messages": [{"role", "content"}, ...]} in the OpenAI
// format, or {"prompt": "raw text"} to continue text without the chat template (for base models), with optional
// "max_tokens" (256), "temperature" (0.8, 0 = greedy), "top_k" (40), "top_p" (0.95), "min_p" (0.05), "seed" and
// "stop" (a string or an array of strings that end the reply; the matched string is cut from it). Returns a
// malloc'd JSON string: {"model", "message": {"role", "content"} or "text" for a prompt, "finish_reason": "stop"
// or "length", "stop_sequence" if a stop string ended it, "usage", "timings"}, or {"error": ...}. Free it with
// systemone_free_string().
// Control-token texts in the messages (e.g. "<|im_start|>") are broken with a zero-width space, so a message
// cannot open a turn of its own at the token level. The model can still follow instructions written in a
// message (prompt injection): treat a reply to untrusted text as untrusted. Same threading and NULL rules as
// systemone_decide(). SYSTEMONE_DUMP=1 prints the rendered prompt to stderr.
char * systemone_chat(systemone * so, const char * request_json);

void systemone_free_string(char * s);

// The error JSON of the last systemone_decide() or systemone_chat() on this handle, "" if it succeeded.
// The pointer is valid until the next request on the same handle, from any thread: read it from the
// thread that got the error, before another request can run.
const char * systemone_last_error(const systemone * so);

#ifdef __cplusplus
}
#endif
