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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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

    // chat mode
    systemone_model mode = SYSTEMONE_MODEL_DECISION;
    common_chat_templates_ptr chat_templates;
    std::vector<std::string> control_texts; // texts of the control tokens, longest first
    std::string control_first;              // the first byte of each

    ~systemone() {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
    }

    // The scores of one task: the prompt through the model in one batch, read as the server reads it.
    std::vector<float> run(const server_task & task) {
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

        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (rc != 0) {
            throw std::runtime_error(string_format("llama_process failed (%d)", rc));
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

    std::string decide(const std::string & request) {
        return locked([&] { return answer(request); });
    }

    std::string chat(const std::string & request) {
        return locked([&] { return generate(request); });
    }

    // chat mode: the chat template of the model, and the control token texts to break in the messages
    void init_chat() {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        chat_templates = common_chat_templates_init(model, "");
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); id++) {
            if (llama_vocab_get_attr(vocab, id) & LLAMA_TOKEN_ATTR_CONTROL) {
                const std::string text = llama_vocab_get_text(vocab, id);
                if (text.size() >= 2) {
                    control_texts.push_back(text);
                }
            }
        }
        std::stable_sort(control_texts.begin(), control_texts.end(), [](const std::string & a, const std::string & b) {
            return a.size() > b.size();
        });
        for (const auto & text : control_texts) {
            if (control_first.find(text[0]) == std::string::npos) {
                control_first += text[0];
            }
        }
    }

private:
    // One request, start to end under the mutex, last_error included: two threads on the same handle
    // must not write last_error at the same time.
    template <typename F>
    std::string locked(F && fn) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string out;
        try {
            out = fn();
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

    // a zero-width space after the first character of each control token text, so it stays text
    // same as d1_break() of server-decision.cpp, with a cap on the passes for a hostile vocabulary
    std::string break_control_tokens(const std::string & text) const {
        static const std::string zwsp = "\xE2\x80\x8B";
        std::string cur = text;
        bool changed = true;
        for (int pass = 0; changed && pass < 8; pass++) {
            changed = false;
            std::string out;
            for (size_t i = 0; i < cur.size();) {
                const std::string * hit = nullptr;
                if (control_first.find(cur[i]) != std::string::npos) {
                    for (const auto & ctrl : control_texts) {
                        if (cur.compare(i, ctrl.size(), ctrl) == 0) {
                            hit = &ctrl;
                            break;
                        }
                    }
                }
                if (hit == nullptr) {
                    out += cur[i++];
                    continue;
                }
                const unsigned char c = (*hit)[0];
                const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
                out += hit->substr(0, n) + zwsp + hit->substr(n);
                i += hit->size();
                changed = true;
            }
            cur = std::move(out);
        }
        return cur;
    }

    // all the strings of a JSON value, keys too
    json break_control_tokens(const json & val, int depth = 0) const {
        if (depth >= 128) {
            throw std::invalid_argument("JSON is nested more than 128 levels deep");
        }
        if (val.is_string()) {
            return break_control_tokens(val.get<std::string>());
        }
        if (val.is_array()) {
            json out = json::array();
            for (const auto & item : val) {
                out.push_back(break_control_tokens(item, depth + 1));
            }
            return out;
        }
        if (val.is_object()) {
            json out = json::object();
            for (const auto & [key, item] : val.items()) {
                out[break_control_tokens(key)] = break_control_tokens(item, depth + 1);
            }
            return out;
        }
        return val;
    }

    // called with the mutex held
    std::string generate(const std::string & request) {
        if (mode != SYSTEMONE_MODEL_CHAT) {
            return error_json("the model is loaded for decisions, load it with model = SYSTEMONE_MODEL_CHAT to chat", "invalid_request_error");
        }
        json body;
        try {
            body = json::parse(request);
        } catch (const std::exception & e) {
            return error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error");
        }
        try {
            if (!body.is_object() || !body.contains("messages") || !body.at("messages").is_array() || body.at("messages").empty()) {
                throw std::invalid_argument("\"messages\" must be a non-empty array");
            }
            common_chat_templates_inputs inputs;
            try {
                inputs.messages = common_chat_msgs_parse_oaicompat(break_control_tokens(body.at("messages")));
            } catch (const std::exception & e) {
                throw std::invalid_argument(std::string("messages: ") + e.what());
            }
            for (const auto & msg : inputs.messages) {
                if (msg.contains_media()) {
                    throw std::invalid_argument("images and audio are not supported by this library yet, use llama-server");
                }
            }

            const int32_t  max_tokens  = json_value(body, "max_tokens", 256);
            const float    temperature = json_value(body, "temperature", 0.8f);
            const int32_t  top_k       = json_value(body, "top_k", 40);
            const float    top_p       = json_value(body, "top_p", 0.95f);
            const float    min_p       = json_value(body, "min_p", 0.05f);
            const uint32_t seed        = json_value(body, "seed", (uint32_t) LLAMA_DEFAULT_SEED);
            if (max_tokens < 1) {
                throw std::invalid_argument("\"max_tokens\" must be at least 1");
            }

            const llama_vocab * vocab = llama_model_get_vocab(model);
            inputs.add_generation_prompt = true;
            inputs.add_bos = llama_vocab_get_add_bos(vocab); // the BOS text of the template is dropped, the token is added below
            const std::string text = common_chat_templates_apply(chat_templates.get(), inputs).prompt.str();
            llama_tokens prompt = common_tokenize(vocab, text, false, true);
            if (inputs.add_bos) {
                prompt.insert(prompt.begin(), llama_vocab_bos(vocab));
            }
            if (prompt.size() >= n_ctx) {
                throw std::invalid_argument(string_format(
                    "the prompt (%zu tokens) leaves no room for a reply in the context (%u tokens), load with a larger n_ctx", prompt.size(), n_ctx));
            }

            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            std::unique_ptr<llama_sampler, void (*)(llama_sampler *)> smpl_guard(smpl, llama_sampler_free);
            if (temperature <= 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            } else {
                llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
            }

            llama_memory_t mem = llama_get_memory(ctx);
            if (mem) {
                llama_memory_clear(mem, true);
            }
            llama_set_embeddings(ctx, false);

            llama_batch_ext * batch = llama_batch_ext_init(ctx);
            std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch_guard(batch, llama_batch_ext_free);
            auto eval = [&](const llama_token * tokens, int32_t n_tokens, llama_pos pos_0) {
                llama_batch_ext_clear(batch);
                for (int32_t i = 0; i < n_tokens; i++) {
                    const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                    if (idx < 0) {
                        throw std::runtime_error(string_format("the prompt (%d tokens) does not fit in one batch", n_tokens));
                    }
                    const llama_pos pos = pos_0 + i;
                    llama_batch_ext_set_pos(batch, idx, &pos);
                    if (i == n_tokens - 1) {
                        llama_batch_ext_set_output_logits(batch, idx, true);
                    }
                }
                const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
                if (rc != 0) {
                    throw std::runtime_error(string_format("llama_process failed (%d)", rc));
                }
            };

            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();
            eval(prompt.data(), (int32_t) prompt.size(), 0);
            const auto t1 = clock::now();

            std::string reply;
            std::string finish_reason = "length";
            int32_t   n_gen  = 0;
            llama_pos n_past = (llama_pos) prompt.size();
            while (n_gen < max_tokens) {
                const llama_token tok = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, tok)) {
                    finish_reason = "stop";
                    break;
                }
                reply += common_token_to_piece(vocab, tok, false);
                n_gen++;
                if (n_gen == max_tokens || (uint32_t) n_past >= n_ctx) {
                    break;
                }
                eval(&tok, 1, n_past++);
            }
            const auto t2 = clock::now();
            const auto ms = [](clock::time_point a, clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };

            json message = json::object();
            message["role"]    = "assistant";
            message["content"] = reply;
            json usage = json::object();
            usage["input_tokens"]  = (int32_t) prompt.size();
            usage["output_tokens"] = n_gen;
            json timings = json::object();
            timings["prompt_ms"]     = ms(t0, t1);
            timings["generation_ms"] = ms(t1, t2);
            json out = json::object();
            out["model"]         = model_name;
            out["message"]       = message;
            out["finish_reason"] = finish_reason;
            out["usage"]         = usage;
            out["timings"]       = timings;
            return out.dump_safe(); // a reply cut by max_tokens can end inside a UTF-8 character
        } catch (const std::invalid_argument & e) {
            return error_json(e.what(), "invalid_request_error");
        } catch (const std::exception & e) {
            return error_json(e.what(), "server_error");
        }
    }

    // called with the mutex held
    std::string answer(const std::string & request) {
        if (mode != SYSTEMONE_MODEL_DECISION) {
            return error_json("the model is loaded for chat, use systemone_chat()", "invalid_request_error");
        }
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

            std::vector<std::vector<float>> results;
            int32_t n_tokens = 0;
            for (const auto & task : tasks) {
                results.push_back(run(task));
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
                answers[question.id] = decision.format_answer(question, scores, false);
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

// nothing may cross the C ABI: an exception escaping here would abort the app
static char * systemone_call(systemone * so, const char * request_json, std::string (systemone::*fn)(const std::string &)) {
    static const char fallback[] = "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}";
    std::string out;
    try {
        if (!so || !request_json) {
            out = error_json("no model or no request", "invalid_request_error");
        } else {
            out = (so->*fn)(request_json);
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

extern "C" {

systemone_params systemone_default_params(void) {
    systemone_params p;
    p.n_gpu_layers = -1;
    p.n_ctx        = 4096;
    p.n_threads    = 0;
    p.verbose      = false;
    p.repack       = true;
    p.model        = SYSTEMONE_MODEL_DECISION;
    return p;
}

systemone * systemone_load(const char * model_path, const systemone_params * params_in) {
    systemone_params params = params_in ? *params_in : systemone_default_params();
    if (params.model != SYSTEMONE_MODEL_DECISION && params.model != SYSTEMONE_MODEL_CHAT) {
        fprintf(stderr, "systemone: unknown model mode %d\n", (int) params.model);
        return nullptr;
    }
    const bool is_chat = params.model == SYSTEMONE_MODEL_CHAT;
    if (!params.verbose) {
        llama_log_set(quiet_log, nullptr);
    }
    static std::once_flag backend_once;
    std::call_once(backend_once, [] { llama_backend_init(); });

    auto so = std::make_unique<systemone>();
    so->n_ctx = params.n_ctx ? params.n_ctx : 4096;
    so->mode  = params.model;

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
    cparams.n_ubatch  = is_chat ? std::min<uint32_t>(so->n_ctx, 512) : so->n_ctx; // chat: smaller compute buffers, less RAM
    cparams.n_seq_max = 1;
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
        if (is_chat) {
            so->init_chat();
            so->type_name = "chat";
        } else {
            so->decision.init(so->model);
            so->type_name = decision_type_name(so->decision.type);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "systemone: failed to init the %s model: %s\n", is_chat ? "chat" : "decision", e.what());
        return nullptr;
    }

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
    return systemone_call(so, request_json, &systemone::decide);
}

char * systemone_chat(systemone * so, const char * request_json) {
    return systemone_call(so, request_json, &systemone::chat);
}

void systemone_free_string(char * s) {
    free(s);
}

const char * systemone_last_error(const systemone * so) {
    return so ? so->last_error.c_str() : "";
}

} // extern "C"
