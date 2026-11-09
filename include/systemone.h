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

typedef struct systemone_params {
    int32_t n_gpu_layers;   // layers on the GPU, 0 = CPU only, -1 = all
    uint32_t n_ctx;         // longest prompt in tokens (state + question + options); 0 = 4096
    int32_t n_threads;      // CPU threads; 0 = hardware default
    bool verbose;           // llama.cpp logs to stderr
} systemone_params;

systemone_params systemone_default_params(void);

// Loads a GGUF decision model. NULL on failure (the message is in systemone_last_error()).
systemone * systemone_load(const char * model_path, const systemone_params * params);
void systemone_free(systemone * so);

// The decision model family llama.cpp detected, e.g. "lfm3_d1", "openjev"; "none" for a model that is not one.
const char * systemone_model_type(const systemone * so);

// Answers one request. Returns a malloc'd JSON string: {"model", "answers", "usage"} as the server, or
// {"error": {"message", "type"}}. Free it with systemone_free_string().
char * systemone_decide(systemone * so, const char * request_json);
void systemone_free_string(char * s);

const char * systemone_last_error(const systemone * so);

#ifdef __cplusplus
}
#endif
