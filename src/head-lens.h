// The decision head of an LFM2 d1 model (d1-omni) or a ModernBERT one (Laya), re-run on the residual stream after
// any encoder block.
//
// These models do not answer through the output matrix: after the backbone, a small trained head (two
// transformer blocks and a scorer, llama.cpp: llama_model_lfm2::graph_decision::build_decision_head)
// turns the residual at each option's position into that option's score. A layer readout therefore has
// to run that head, not a norm and a dot product. This is the head, as a small ggml graph over float
// copies of its weights, exactly the ops llama.cpp builds: on the last encoder block it reproduces the
// model's own answer, which is the test.
#pragma once

#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_model;

struct lfm2_head_lens {
    ~lfm2_head_lens();

    // false if this model has no such head (then the readout is not available)
    bool init(const llama_model * model);

    // the head's scores at the given prompt positions for one question type, from the residual stream
    // after an encoder block: resid is [n_tokens * n_embd], as the layer wrote it (before the output norm),
    // or, with normed = true, the output norm's result itself (what the head receives)
    std::vector<float> run(const float * resid, int32_t n_tokens, const std::vector<int32_t> & positions, int32_t type, int n_threads, bool normed = false, struct ggml_threadpool * threadpool = nullptr);

    // the question-type vector the head adds to its input
    std::vector<float> type_row(int32_t type) const;

    bool rms_input = true; // the backbone's output norm: RMS (LFM2) or LayerNorm without bias (ModernBERT)

    int64_t n_embd = 0, n_head = 0, n_head_kv = 0, n_embd_head = 0, n_types = 0;

private:
    struct layer_w {
        ggml_tensor * attn_norm = nullptr, * attn_norm_b = nullptr;
        ggml_tensor * wqkv = nullptr, * wqkv_b = nullptr, * wo = nullptr, * wo_b = nullptr;
        ggml_tensor * ffn_norm = nullptr, * ffn_norm_b = nullptr;
        ggml_tensor * ffn_up = nullptr, * ffn_up_b = nullptr, * ffn_down = nullptr, * ffn_down_b = nullptr;
    };
    ggml_context * ctx_w = nullptr; // the float copies of the weights
    std::vector<layer_w> layers;
    ggml_tensor * output_norm = nullptr, * type_embd = nullptr;
    ggml_tensor * cls_norm = nullptr, * cls_norm_b = nullptr, * cls = nullptr, * cls_b = nullptr, * cls_out = nullptr, * cls_out_b = nullptr;
    float eps_ln = 1e-5f, eps_rms = 1e-5f;

    ggml_tensor * copy(const ggml_tensor * src, ggml_type type);
};
