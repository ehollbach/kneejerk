#include "head-lens.h"

#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

// a weight tensor of the model as floats, row by row, whatever its type and wherever it lives
void read_rows(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t n = t->ne[0], rows = ggml_nrows(t);
    out.resize(n * rows);
    const size_t row_bytes = ggml_row_size(t->type, n);
    std::vector<uint8_t> buf(row_bytes);
    const auto * traits = ggml_get_type_traits(t->type);
    for (int64_t r = 0; r < rows; r++) {
        ggml_backend_tensor_get(t, buf.data(), (size_t) r * t->nb[1], row_bytes);
        if (t->type == GGML_TYPE_F32) {
            memcpy(out.data() + r * n, buf.data(), n * sizeof(float));
        } else if (traits && traits->to_float) {
            traits->to_float(buf.data(), out.data() + r * n, n);
        } else {
            throw std::runtime_error(std::string("head lens: cannot read a ") + ggml_type_name(t->type) + " tensor");
        }
    }
}

} // namespace

lfm2_head_lens::~lfm2_head_lens() {
    if (ctx_w) {
        ggml_free(ctx_w);
    }
}

// a 1-D weight (norms, biases) as a float copy; a matrix stays the model's own tensor: reading its rows is not
// possible when llama.cpp has repacked it for fast kernels, but ggml_mul_mat on it is, repacked or not
ggml_tensor * lfm2_head_lens::copy(const ggml_tensor * src, ggml_type type) {
    if (!src) {
        return nullptr;
    }
    if (type != GGML_TYPE_F32) {
        return const_cast<ggml_tensor *>(src);
    }
    std::vector<float> f;
    read_rows(src, f);
    ggml_tensor * t = ggml_new_tensor_2d(ctx_w, type, src->ne[0], ggml_nrows(src));
    memcpy(t->data, f.data(), f.size() * sizeof(float));
    return t;
}

bool lfm2_head_lens::init(const llama_model * model) {
    const auto & hp = model->hparams;
    if ((model->arch != LLM_ARCH_LFM2 && model->arch != LLM_ARCH_MODERN_BERT) || hp.n_layer_decision == 0 ||
        !model->type_embd || !model->cls || !model->cls_out) {
        return false;
    }
    rms_input = model->arch == LLM_ARCH_LFM2;
    const int n_layer     = (int) hp.n_layer();
    const int n_layer_enc = n_layer - (int) hp.n_layer_decision;
    n_embd      = hp.n_embd;
    n_head      = hp.n_head(n_layer_enc);
    n_head_kv   = hp.n_head_kv(n_layer_enc);
    n_embd_head = hp.n_embd_head_v();
    n_types     = model->type_embd->ne[1];
    eps_ln      = hp.f_norm_eps;
    eps_rms     = hp.f_norm_rms_eps;

    // room for the float copies of the vectors (the matrices stay the model's)
    size_t bytes = 1024 * 1024;
    for (int il = n_layer_enc; il < n_layer; il++) {
        const auto & l = model->layers[il];
        for (const ggml_tensor * t : {l.attn_norm, l.attn_norm_b, l.wqkv, l.wqkv_b, l.wo, l.wo_b, l.ffn_norm, l.ffn_norm_b, l.ffn_up, l.ffn_up_b, l.ffn_down, l.ffn_down_b}) {
            if (!t) {
                return false;
            }
            if (ggml_n_dims(t) == 1) bytes += ggml_nelements(t) * 4 + 256;
        }
    }
    for (const ggml_tensor * t : {model->output_norm, model->type_embd, model->cls_norm, model->cls_norm_b, model->cls, model->cls_b, model->cls_out, model->cls_out_b}) {
        if (!t) {
            return false;
        }
        bytes += ggml_nelements(t) * 4 + 256;
    }
    ggml_init_params ip = { bytes, nullptr, false };
    ctx_w = ggml_init(ip);
    if (!ctx_w) {
        return false;
    }
    for (int il = n_layer_enc; il < n_layer; il++) {
        const auto & l = model->layers[il];
        layer_w w;
        w.attn_norm   = copy(l.attn_norm,   GGML_TYPE_F32);
        w.attn_norm_b = copy(l.attn_norm_b, GGML_TYPE_F32);
        w.wqkv        = copy(l.wqkv,        GGML_TYPE_F16);
        w.wqkv_b      = copy(l.wqkv_b,      GGML_TYPE_F32);
        w.wo          = copy(l.wo,          GGML_TYPE_F16);
        w.wo_b        = copy(l.wo_b,        GGML_TYPE_F32);
        w.ffn_norm    = copy(l.ffn_norm,    GGML_TYPE_F32);
        w.ffn_norm_b  = copy(l.ffn_norm_b,  GGML_TYPE_F32);
        w.ffn_up      = copy(l.ffn_up,      GGML_TYPE_F16);
        w.ffn_up_b    = copy(l.ffn_up_b,    GGML_TYPE_F32);
        w.ffn_down    = copy(l.ffn_down,    GGML_TYPE_F16);
        w.ffn_down_b  = copy(l.ffn_down_b,  GGML_TYPE_F32);
        layers.push_back(w);
    }
    output_norm = copy(model->output_norm, GGML_TYPE_F32);
    type_embd   = copy(model->type_embd,   GGML_TYPE_F32);
    cls_norm    = copy(model->cls_norm,    GGML_TYPE_F32);
    cls_norm_b  = copy(model->cls_norm_b,  GGML_TYPE_F32);
    cls         = copy(model->cls,         GGML_TYPE_F16);
    cls_b       = copy(model->cls_b,       GGML_TYPE_F32);
    cls_out     = copy(model->cls_out,     GGML_TYPE_F16);
    cls_out_b   = copy(model->cls_out_b,   GGML_TYPE_F32);
    return true;
}

std::vector<float> lfm2_head_lens::run(const float * resid, int32_t n_tokens, const std::vector<int32_t> & positions, int32_t type, int n_threads, bool normed, ggml_threadpool * threadpool) {
    if (!ctx_w) {
        throw std::runtime_error("head lens: not initialised");
    }
    if (type < 0 || type >= n_types) {
        throw std::runtime_error("head lens: question type out of range");
    }
    const int64_t T = n_tokens;
    const int64_t n_ff = layers.empty() ? 0 : layers[0].ffn_up->ne[1];
    // activations: generous, freed after the run
    const size_t act = (size_t) T * (16 * n_embd + 2 * n_ff + 2 * n_head * T) * sizeof(float) * (layers.size() + 1) + 8 * 1024 * 1024;
    ggml_init_params ip = { act, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        throw std::runtime_error("head lens: out of memory");
    }
    auto layer_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        x = ggml_norm(ctx, x, eps_ln);
        x = ggml_mul(ctx, x, w);
        return ggml_add(ctx, x, b);
    };

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
    memcpy(x->data, resid, (size_t) T * n_embd * sizeof(float));

    // the backbone's output norm, then the question type: what the head receives
    ggml_tensor * h = normed ? x : ggml_mul(ctx, rms_input ? ggml_rms_norm(ctx, x, eps_rms) : ggml_norm(ctx, x, eps_ln), output_norm);
    ggml_tensor * type_row = ggml_view_1d(ctx, type_embd, n_embd, (size_t) type * type_embd->nb[1]);
    h = ggml_add(ctx, h, type_row);

    const float scale = 1.0f / sqrtf((float) n_embd_head);
    for (const auto & w : layers) {
        ggml_tensor * cur = layer_norm(h, w.attn_norm, w.attn_norm_b);
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, w.wqkv, cur), w.wqkv_b);        // [3 n_embd, T]
        const int64_t n_embd_q = n_embd_head * n_head, n_embd_k = n_embd_head * n_head_kv;
        ggml_tensor * Q = ggml_view_3d(ctx, qkv, n_embd_head, n_head,    T, n_embd_head * sizeof(float), qkv->nb[1], 0);
        ggml_tensor * K = ggml_view_3d(ctx, qkv, n_embd_head, n_head_kv, T, n_embd_head * sizeof(float), qkv->nb[1], n_embd_q * sizeof(float));
        ggml_tensor * V = ggml_view_3d(ctx, qkv, n_embd_head, n_head_kv, T, n_embd_head * sizeof(float), qkv->nb[1], (n_embd_q + n_embd_k) * sizeof(float));
        // no positional encoding, no mask: every position reads every position
        ggml_tensor * q = ggml_permute(ctx, Q, 0, 2, 1, 3);                                   // [n_embd_head, T, n_head]
        ggml_tensor * k = ggml_permute(ctx, K, 0, 2, 1, 3);
        ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));                   // [T, n_embd_head, n_head_kv]
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                                           // [T_k, T_q, n_head]
        kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                                         // [n_embd_head, T_q, n_head]
        ggml_tensor * out = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), n_embd_q, T); // [n_embd, T]
        cur = ggml_add(ctx, ggml_mul_mat(ctx, w.wo, out), w.wo_b);
        ggml_tensor * ffn_inp = ggml_add(ctx, cur, h);
        cur = layer_norm(ffn_inp, w.ffn_norm, w.ffn_norm_b);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_up, cur), w.ffn_up_b);
        cur = ggml_relu(ctx, cur);
        cur = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_down, cur), w.ffn_down_b);
        h = ggml_add(ctx, cur, ffn_inp);
    }
    // the scorer
    ggml_tensor * c = layer_norm(h, cls_norm, cls_norm_b);
    c = ggml_add(ctx, ggml_mul_mat(ctx, cls, c), cls_b);
    c = ggml_gelu_erf(ctx, c);
    ggml_tensor * s = ggml_add(ctx, ggml_mul_mat(ctx, cls_out, c), cls_out_b);                // [1, T]

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(gf, s);
    ggml_cplan plan = ggml_graph_plan(gf, n_threads > 0 ? n_threads : 4, threadpool);
    std::vector<uint8_t> work(plan.work_size + 64);
    plan.work_data = work.data();
    const ggml_status st = ggml_graph_compute(gf, &plan);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("head lens: compute failed");
    }
    std::vector<float> scores;
    const float * sd = (const float *) s->data;
    for (const int32_t p : positions) {
        if (p < 0 || p >= T) {
            ggml_free(ctx);
            throw std::runtime_error("head lens: position out of range");
        }
        scores.push_back(sd[(size_t) p * (s->nb[1] / sizeof(float))]);
    }
    ggml_free(ctx);
    return scores;
}

std::vector<float> lfm2_head_lens::type_row(int32_t type) const {
    std::vector<float> row(n_embd, 0.0f);
    if (type_embd && type >= 0 && type < n_types) {
        memcpy(row.data(), (const char *) type_embd->data + (size_t) type * type_embd->nb[1], n_embd * sizeof(float));
    }
    return row;
}
