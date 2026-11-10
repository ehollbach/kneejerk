// The library: load a decision model, answer /v1/systemone requests without a server.
//
// The prompt of every family (what the model reads, where its answer is read from) is upstream
// llama.cpp code, unchanged: src/upstream/server-decision.cpp. This file does what the server's
// loop and its /v1/systemone handler do around it: one task per question variant, the prompt
// through the model in one batch, the scores read back as send_decision() reads them, then
// format_answer(). Parity with llama-server is the test (tests/parity.py).

#include "systemone.h"

#include "server-decision.h"
#include "llama-ext.h"
#include "llama-model.h"   // the output norm and matrix, the layer kinds: for the layer readout and model info
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

const char * decision_type_name(common_decision_type type) {
    switch (type) {
        case COMMON_DECISION_TYPE_NONE:            return "none";
        case COMMON_DECISION_TYPE_OPENJEV:         return "openjev";
        case COMMON_DECISION_TYPE_GRANITE_SWITCH:  return "granite_switch";
        case COMMON_DECISION_TYPE_LEV:             return "lev";
        case COMMON_DECISION_TYPE_KEV:             return "kev";
        case COMMON_DECISION_TYPE_NIMBLE:          return "nimble";
        case COMMON_DECISION_TYPE_LAYA:            return "laya";
        case COMMON_DECISION_TYPE_CLEF:            return "clef";
        case COMMON_DECISION_TYPE_PPLX_DECIDER:    return "pplx_decider";
        case COMMON_DECISION_TYPE_VLLM_DECISION2:  return "vllm_decision2";
        case COMMON_DECISION_TYPE_VLLM_D3:         return "vllm_d3";
        case COMMON_DECISION_TYPE_LFM2_D1:         return "lfm2_d1";
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:    return "lfm2_d1_omni";
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:   return "h2o_lightning";
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV: return "minicpm_ocr_jev";
        case COMMON_DECISION_TYPE_LFM3_D1:         return "lfm3_d1";
        case COMMON_DECISION_TYPE_LFM3_D1_OMNI:    return "lfm3_d1_omni";
        default:                                   return "unknown";
    }
}

void quiet_log(ggml_log_level, const char *, void *) {}

// one row of a weight tensor as floats, whatever its type and wherever it lives
void tensor_row_to_float(const ggml_tensor * t, int64_t row, std::vector<float> & out) {
    const int64_t n = t->ne[0];
    out.resize(n);
    const size_t row_bytes = ggml_row_size(t->type, n);
    std::vector<uint8_t> buf(row_bytes);
    ggml_backend_tensor_get(t, buf.data(), (size_t) row * t->nb[1], row_bytes);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), buf.data(), n * sizeof(float));
        return;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error(std::string("cannot read a ") + ggml_type_name(t->type) + " tensor");
    }
    traits->to_float(buf.data(), out.data(), n);
}

std::string error_json(const std::string & message, const char * type) {
    json err = json::object();
    err["message"] = message;
    err["type"]    = type;
    json out = json::object();
    out["error"] = err;
    return out.dump();
}

} // namespace

struct systemone {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    server_decision_context decision;
    std::string model_name;
    std::string type_name;
    std::string last_error;
    uint32_t n_ctx = 0;
    std::mutex mutex; // one request at a time: the context is shared

    // Layer readout (a logit lens): the residual stream at the last prompt position after each layer,
    // read through the model's own output norm and output matrix, restricted to the label tokens.
    bool lens_active = false;
    std::vector<std::vector<float>> lens_resid;                     // [n_layer][n_embd], empty if the layer was not seen
    std::unordered_map<llama_token, std::vector<float>> lens_rows;  // rows of the output matrix, dequantized
    std::vector<float> lens_norm_w;                                 // the output norm weight
    const ggml_tensor * lens_out = nullptr;                         // the output matrix (or the tied embedding)

    // ggml asks before each node whether its data is needed, then calls again once it is computed
    static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
        auto * so = static_cast<systemone *>(user_data);
        if (!so->lens_active) {
            return !ask;
        }
        const char * name = ggml_get_name(t);
        if (strncmp(name, "l_out-", 6) != 0) {
            return !ask;
        }
        if (ask) {
            return true;
        }
        const int il = atoi(name + 6);
        if (il < 0 || il >= (int) so->lens_resid.size() || t->type != GGML_TYPE_F32 || t->ne[1] < 1) {
            return true;
        }
        auto & row = so->lens_resid[il];
        row.resize(t->ne[0]);
        ggml_backend_tensor_get(t, row.data(), (size_t) (t->ne[1] - 1) * t->nb[1], t->ne[0] * sizeof(float));
        return true;
    }

    // the output norm and matrix, once; throws if this architecture's head is not RMS norm + matrix
    void lens_init() {
        if (lens_out) {
            return;
        }
        if (!model->output_norm || model->output_norm_b || model->output_s) {
            throw std::runtime_error("layer readout: this model's output head is not an RMS norm and a matrix");
        }
        const ggml_tensor * out = model->output ? model->output : model->tok_embd;
        if (!out) {
            throw std::runtime_error("layer readout: the model has no output matrix");
        }
        tensor_row_to_float(model->output_norm, 0, lens_norm_w);
        lens_out = out;
    }

    const std::vector<float> & lens_row(llama_token label) {
        auto it = lens_rows.find(label);
        if (it == lens_rows.end()) {
            if (lens_rows.size() > 4096) {
                lens_rows.clear();
            }
            it = lens_rows.emplace(label, std::vector<float>()).first;
            tensor_row_to_float(lens_out, label, it->second);
        }
        return it->second;
    }

    // the label logits a layer's residual would give through the output head: RMS norm, then the matrix rows
    std::vector<float> lens_scores(const std::vector<float> & resid, const std::vector<llama_token> & labels) {
        const size_t n = resid.size();
        if (n != lens_norm_w.size()) {
            throw std::runtime_error("layer readout: residual width does not match the output norm");
        }
        double ss = 0.0;
        for (const float v : resid) {
            ss += (double) v * v;
        }
        const float scale = 1.0f / sqrtf((float) (ss / n) + model->hparams.f_norm_rms_eps);
        std::vector<float> h(n);
        for (size_t i = 0; i < n; i++) {
            h[i] = resid[i] * scale * lens_norm_w[i];
        }
        std::vector<float> scores;
        for (const llama_token label : labels) {
            const auto & w = lens_row(label);
            double dot = 0.0;
            for (size_t i = 0; i < n; i++) {
                dot += (double) h[i] * w[i];
            }
            scores.push_back((float) dot);
        }
        return scores;
    }

    static std::vector<float> pool_groups(std::vector<float> scores, const std::vector<int32_t> & groups) {
        if (groups.empty()) {
            return scores;
        }
        std::vector<float> pooled;
        size_t i = 0;
        for (const int32_t n : groups) {
            if (n <= 0 || i + n > scores.size()) {
                throw std::runtime_error("label groups do not match the labels");
            }
            pooled.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
            i += n;
        }
        return pooled;
    }

    // what the model is made of, for an architecture view
    json model_info() const {
        const auto & hp = model->hparams;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        char desc[256];
        llama_model_desc(model, desc, sizeof(desc));
        json info = json::object();
        info["model"]            = model_name;
        info["name"]             = model->name;
        info["architecture"]     = model->arch_name();
        info["description"]      = desc;
        info["decision_type"]    = type_name;
        info["n_params"]         = (int64_t) llama_model_n_params(model);
        info["size_bytes"]       = (int64_t) llama_model_size(model);
        info["n_layer"]          = (int32_t) hp.n_layer();
        info["n_layer_decision"] = (int32_t) hp.n_layer_decision;
        info["n_embd"]           = (int32_t) hp.n_embd;
        info["n_head"]           = llama_model_n_head(model);
        info["n_head_kv"]        = llama_model_n_head_kv(model);
        info["n_vocab"]          = llama_vocab_n_tokens(vocab);
        info["n_ctx_train"]      = llama_model_n_ctx_train(model);
        info["n_ctx"]            = (int32_t) n_ctx;
        const bool lfm = model->arch == LLM_ARCH_LFM2 || model->arch == LLM_ARCH_LFM3;
        json layers = json::array();
        for (uint32_t il = 0; il < hp.n_layer(); il++) {
            json l = json::object();
            l["i"] = (int32_t) il;
            if (hp.is_recr(il)) {
                l["kind"] = lfm ? "shortconv" : "recurrent";
            } else {
                l["kind"] = hp.is_swa(il) ? "attention_swa" : "attention";
            }
            l["decision_head"] = hp.n_layer_decision > 0 && il >= hp.n_layer() - hp.n_layer_decision;
            layers.push_back(l);
        }
        info["layers"] = layers;
        return info;
    }

    ~systemone() {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
    }

    // The scores of one task: the prompt through the model in one batch, read as the server reads it.
    // layers, if given, receives the label scores of every layer (empty rows for layers without a readout)
    std::vector<float> run(const server_task & task, std::vector<std::vector<float>> * layers = nullptr) {
        const llama_tokens & tokens = task.tokens.tokens;
        const int32_t n_tokens = (int32_t) tokens.size();
        if (n_tokens == 0) {
            throw std::runtime_error("empty prompt");
        }
        if ((uint32_t) n_tokens > n_ctx) {
            throw std::runtime_error(string_format(
                "request (%d tokens) exceeds the context size (%u tokens), load with a larger n_ctx", n_tokens, n_ctx));
        }

        const auto & d = task.decision;
        const bool wants_embd = d.pooled || !d.markers.empty() || d.pointer >= 0 || d.n_scores > 0 || !d.order.empty();
        const int32_t pos_first = d.pos_first();

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        llama_set_embeddings(ctx, wants_embd);

        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch_guard(batch, llama_batch_ext_free);

        // row in the batch of each prompt position that has an output
        std::vector<int32_t> rows(n_tokens, -1);
        for (int32_t i = 0; i < n_tokens; i++) {
            const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            if (idx < 0) {
                throw std::runtime_error(string_format("the prompt (%d tokens) does not fit in one batch", n_tokens));
            }
            const llama_pos pos = i; // text tokens: one position each, nothing is set by default
            llama_batch_ext_set_pos(batch, idx, &pos);
            rows[i] = idx;
            const bool last = i == n_tokens - 1;
            if (!d.labels.empty() && last) {
                llama_batch_ext_set_output_logits(batch, idx, true);
            }
            if (wants_embd && (d.pooled || (pos_first >= 0 && i >= pos_first) || (d.n_scores > 0 && i < d.n_scores))) {
                llama_batch_ext_set_output_embd(batch, idx, true);
            }
            if (!d.order.empty()) {
                llama_batch_ext_set_decision_order(batch, idx, (llama_decision_order) d.order[i]);
            }
        }

        const bool lens = layers != nullptr && !d.labels.empty() && !d.pooled;
        if (lens) {
            lens_init();
            lens_resid.assign(llama_model_n_layer(model), {});
            lens_active = true;
        }
        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        lens_active = false;
        if (rc != 0) {
            throw std::runtime_error(string_format("llama_process failed (%d)", rc));
        }
        if (lens) {
            layers->clear();
            for (const auto & resid : lens_resid) {
                layers->push_back(resid.empty() ? std::vector<float>() : pool_groups(lens_scores(resid, d.labels), d.label_groups));
            }
            lens_resid.clear();
        }

        std::vector<float> scores;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool dump = getenv("SYSTEMONE_DUMP") != nullptr; // the prompt and the raw scores, to compare with llama-server
        if (dump) {
            std::string text;
            for (const llama_token t : tokens) {
                text += common_token_to_piece(vocab, t, true);
            }
            fprintf(stderr, "--- prompt (%d tokens) ---\n%s\n--- labels %zu groups %zu markers %zu pointer %d order %zu pooled %d ---\n",
                    n_tokens, text.c_str(), d.labels.size(), d.label_groups.size(), d.markers.size(), d.pointer, d.order.size(), (int) d.pooled);
        }

        if (d.pooled) {
            const float * embd = llama_get_embeddings_seq(ctx, 0);
            if (!embd) {
                throw std::runtime_error("failed to get the class scores");
            }
            scores.assign(embd, embd + llama_model_n_embd_out(model));
            return scores;
        }

        if (!d.labels.empty()) {
            const float * logits = llama_get_logits_ith(ctx, rows[n_tokens - 1]);
            if (!logits) {
                throw std::runtime_error("failed to get logits");
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : d.labels) {
                if (label < 0 || label >= n_vocab) {
                    throw std::runtime_error("label token out of the vocabulary");
                }
                scores.push_back(logits[label]);
            }
            if (!d.label_groups.empty()) {
                std::vector<float> pooled;
                size_t i = 0;
                for (const int32_t n : d.label_groups) {
                    if (n <= 0 || i + n > scores.size()) {
                        throw std::runtime_error("label groups do not match the labels");
                    }
                    pooled.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
                    i += n;
                }
                scores = std::move(pooled);
            }
            if (dump) {
                fprintf(stderr, "--- raw scores:");
                for (const float v : scores) fprintf(stderr, " %.5f", v);
                fprintf(stderr, "\n");
            }
            return scores;
        }

        auto get_embd = [&](int32_t pos) -> const float * {
            return pos >= 0 && pos < n_tokens ? llama_get_embeddings_ith(ctx, rows[pos]) : nullptr;
        };

        // joint head: the scores are the first rows
        for (int32_t i = 0; i < d.n_scores; i++) {
            const float * embd = get_embd(i);
            if (!embd) {
                throw std::runtime_error("failed to get embeddings");
            }
            scores.push_back(embd[0]);
        }

        const int32_t n_embd_out = llama_model_n_embd_out(model);
        const int32_t n_pointer  = n_embd_out / 2;
        const float * embd_q = d.pointer >= 0 ? get_embd(d.pointer) : nullptr;
        if (d.column < 0 || d.column >= n_embd_out) {
            throw std::runtime_error("embedding column out of range");
        }
        for (const int32_t marker : d.markers) {
            const float * embd = get_embd(marker);
            if (!embd || (d.pointer >= 0 && !embd_q)) {
                throw std::runtime_error("failed to get embeddings at the option markers");
            }
            if (d.pointer < 0) {
                scores.push_back(embd[d.column]);
                continue;
            }
            float dot = 0.0f;
            for (int32_t i = 0; i < n_pointer; i++) {
                dot += embd_q[i] * embd[n_pointer + i];
            }
            scores.push_back(dot / sqrtf((float) n_pointer));
        }
        return scores;
    }

    // One request, start to end under the mutex, last_error included: two threads deciding on the same
    // handle must not write last_error at the same time.
    std::string decide(const std::string & request) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string out;
        try {
            out = answer(request);
        } catch (...) {
            out = error_json("internal error", "server_error");
        }
        if (out.rfind("{\"error\"", 0) == 0) {
            last_error = out;
        } else {
            last_error.clear();
        }
        return out;
    }

private:
    // the answer of each layer in the shape of the final answer: a row of probabilities per option, or the
    // noul probability; null for a layer without a readout
    json layers_answer(const server_decision_question & question, size_t n_variants, size_t first_task,
                       const std::vector<std::vector<std::vector<float>>> & layer_results, const std::string & error) const {
        json out = json::object();
        if (!error.empty()) {
            out["error"] = error;
            return out;
        }
        json options = json::array();
        for (const auto & option : question.options) {
            options.push_back(option.key);
        }
        out["options"] = options;
        const size_t n_layer = layer_results[first_task].size();
        json rows = json::array();
        for (size_t il = 0; il < n_layer; il++) {
            std::vector<std::vector<float>> scores;
            bool complete = true;
            for (size_t v = 0; v < n_variants; v++) {
                const auto & per_layer = layer_results[first_task + v];
                if (il >= per_layer.size() || per_layer[il].empty()) {
                    complete = false;
                    break;
                }
                scores.push_back(per_layer[il]);
            }
            if (!complete) {
                rows.push_back(nullptr);
                continue;
            }
            const json a = decision.format_answer(question, scores, false);
            if (a.contains("noul")) {
                rows.push_back(a.at("noul").get<double>());
            } else if (a.contains("probabilities")) {
                json row = json::array();
                for (const auto & option : question.options) {
                    row.push_back(a.at("probabilities").at(option.key).get<double>());
                }
                rows.push_back(row);
            } else {
                rows.push_back(nullptr);
            }
        }
        out["probabilities"] = rows;
        return out;
    }

    // called with the mutex held
    std::string answer(const std::string & request) {
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            return error_json("This model is not a decision model", "not_supported_error");
        }
        json body;
        try {
            body = json::parse(request);
        } catch (const std::exception & e) {
            return error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error");
        }
        try {
            const auto questions = decision.parse_questions(body);
            std::vector<raw_buffer> files;
            const json state = decision.prepare_state(decision.parse_state(body, files));
            if (!files.empty()) {
                return error_json("images and audio are not supported by this library yet, use llama-server", "not_supported_error");
            }

            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task(SERVER_TASK_TYPE_DECISION);
                decision.fill_task_joint(state, questions, files, nullptr, mtmd_helper_init_opt{}, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task(SERVER_TASK_TYPE_DECISION);
                        decision.fill_task(state, questions, question, variant, files, nullptr, mtmd_helper_init_opt{}, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }

            // "layers": true (an extension of this library) adds the answer of every layer, a logit lens
            const bool want_layers = json_value(body, "layers", false);
            std::vector<std::vector<float>> results;
            std::vector<std::vector<std::vector<float>>> layer_results; // [task][layer][option]
            int32_t n_tokens = 0;
            std::string layers_error;
            for (const auto & task : tasks) {
                std::vector<std::vector<float>> layers;
                if (want_layers && !decision.is_joint() && !task.decision.labels.empty() && !task.decision.pooled) {
                    try {
                        results.push_back(run(task, &layers));
                    } catch (const std::exception & e) {
                        layers_error = e.what();
                        results.push_back(run(task));
                    }
                } else {
                    results.push_back(run(task));
                    if (want_layers && layers_error.empty()) {
                        layers_error = "layer readout is only available for models read from label tokens (d1, lev, pplx-decider, ...)";
                    }
                }
                layer_results.push_back(std::move(layers));
                n_tokens += task.n_tokens();
            }

            json answers = json::object();
            size_t i_result = 0;
            size_t i_score  = 0;
            for (const auto & question : questions) {
                std::vector<std::vector<float>> scores;
                if (decision.is_joint()) {
                    const auto & all = results[0];
                    if (i_score + question.options.size() > all.size()) {
                        throw std::runtime_error("the joint head returned fewer scores than options");
                    }
                    scores.emplace_back(all.begin() + i_score, all.begin() + i_score + question.options.size());
                    i_score += question.options.size();
                } else {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        scores.push_back(results[i_result++]);
                    }
                }
                json answer = decision.format_answer(question, scores, false);
                if (want_layers) {
                    answer["layers"] = layers_answer(question, scores.size(), i_result - scores.size(), layer_results, layers_error);
                }
                answers[question.id] = answer;
            }

            json usage = json::object();
            usage["input_tokens"]  = n_tokens;
            usage["output_tokens"] = 0;
            json out = json::object();
            out["model"]   = model_name;
            out["answers"] = answers;
            out["usage"]   = usage;
            return out.dump();
        } catch (const std::invalid_argument & e) {
            return error_json(e.what(), "invalid_request_error");
        } catch (const std::exception & e) {
            return error_json(e.what(), "server_error");
        }
    }
};

extern "C" {

systemone_params systemone_default_params(void) {
    systemone_params p;
    p.n_gpu_layers = -1;
    p.n_ctx        = 4096;
    p.n_threads    = 0;
    p.verbose      = false;
    p.repack       = true;
    return p;
}

systemone * systemone_load(const char * model_path, const systemone_params * params_in) {
    systemone_params params = params_in ? *params_in : systemone_default_params();
    if (!params.verbose) {
        llama_log_set(quiet_log, nullptr);
    }
    static std::once_flag backend_once;
    std::call_once(backend_once, [] { llama_backend_init(); });

    auto so = std::make_unique<systemone>();
    so->n_ctx = params.n_ctx ? params.n_ctx : 4096;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers < 0 ? 999 : params.n_gpu_layers;
    mparams.use_extra_bufts = params.repack; // repacked copies of the weights: ~1 GB more RAM for d1-2B, faster on ARM
    so->model = llama_model_load_from_file(model_path, mparams);
    if (!so->model) {
        fprintf(stderr, "systemone: failed to load %s\n", model_path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = so->n_ctx;
    cparams.n_batch   = so->n_ctx; // the whole prompt goes through in one batch
    cparams.n_ubatch  = so->n_ctx;
    cparams.n_seq_max = 1;
    cparams.cb_eval           = systemone::cb_eval; // only reads when a request asks for layers
    cparams.cb_eval_user_data = so.get();
    cparams.no_perf   = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (params.n_threads > 0) {
        cparams.n_threads       = params.n_threads;
        cparams.n_threads_batch = params.n_threads;
    }
    so->ctx = llama_init_from_model(so->model, cparams);
    if (!so->ctx) {
        fprintf(stderr, "systemone: failed to create the context\n");
        return nullptr;
    }

    try {
        so->decision.init(so->model);
    } catch (const std::exception & e) {
        fprintf(stderr, "systemone: failed to init the decision model: %s\n", e.what());
        return nullptr;
    }
    so->type_name = decision_type_name(so->decision.type);

    const char * slash = strrchr(model_path, '/');
    so->model_name = slash ? slash + 1 : model_path;
    return so.release();
}

void systemone_free(systemone * so) {
    delete so;
}

const char * systemone_model_type(const systemone * so) {
    return so ? so->type_name.c_str() : "none";
}

char * systemone_decide(systemone * so, const char * request_json) {
    // nothing may cross the C ABI: an exception escaping here would abort the app
    static const char fallback[] = "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}";
    std::string out;
    try {
        if (!so || !request_json) {
            out = error_json("no model or no request", "invalid_request_error");
        } else {
            out = so->decide(request_json);
        }
    } catch (...) {
        out.clear();
    }
    const char * src = out.empty() ? fallback : out.c_str();
    const size_t n   = out.empty() ? sizeof(fallback) : out.size() + 1;
    char * s = (char *) malloc(n);
    if (s) {
        memcpy(s, src, n);
    }
    return s;
}

char * systemone_model_info(const systemone * so) {
    std::string out;
    try {
        out = so ? so->model_info().dump_safe() : error_json("no model", "invalid_request_error");
    } catch (...) {
        out = "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}";
    }
    char * s = (char *) malloc(out.size() + 1);
    if (s) {
        memcpy(s, out.c_str(), out.size() + 1);
    }
    return s;
}

void systemone_free_string(char * s) {
    free(s);
}

const char * systemone_last_error(const systemone * so) {
    return so ? so->last_error.c_str() : "";
}

} // extern "C"
