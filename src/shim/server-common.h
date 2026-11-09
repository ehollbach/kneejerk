// What server-decision.cpp needs from llama.cpp's tools/server/server-common.h, and nothing else.
// The upstream file is kept as is (src/upstream/), so it can be diffed against new llama.cpp releases;
// this header stands in for the server it was written in.
#pragma once

#include "common.h"
#include "chat.h"
#include "llama.h"
#include "json.h"
#include "base64.hpp"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;
using raw_buffer = std::vector<uint8_t>;

#define SRV_INF(fmt, ...) fprintf(stderr, "systemone: " fmt, __VA_ARGS__)

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

// The prompt of a task: tokens only, media is out of scope here.
struct server_tokens {
    server_tokens() = default;
    server_tokens(const llama_tokens & tokens, bool /*has_mtmd*/) : tokens(tokens) {}

    llama_token operator[](size_t index) const { return tokens[index]; }
    void push_back(llama_token tok) { tokens.push_back(tok); }
    size_t size() const { return tokens.size(); }
    llama_tokens get_text_tokens() const { return tokens; }

    size_t get_common_prefix(const server_tokens & b) const {
        const size_t n = std::min(tokens.size(), b.tokens.size());
        size_t i = 0;
        while (i < n && tokens[i] == b.tokens[i]) {
            i++;
        }
        return i;
    }

    const mtmd::input_chunk_ptr & find_chunk(size_t /*idx*/) const {
        throw std::runtime_error("media chunks are not supported by this library");
    }

    llama_tokens tokens;
};

// Media (images, audio, video) stays with llama-server for now: these are the server's helpers,
// reached only when a request carries files, and they say so.
inline const char * get_media_marker() {
    return "<__media__>";
}

inline void handle_media(std::vector<raw_buffer> & /*out_files*/, const std::string & /*url*/, const std::string & /*media_path*/) {
    throw std::invalid_argument("images and audio are not supported by this library yet, use llama-server");
}

inline server_tokens process_mtmd_prompt(mtmd_context * /*mctx*/, const std::string & /*prompt*/,
                                         const std::vector<raw_buffer> & /*files*/, const mtmd_helper_init_opt & /*opt*/) {
    throw std::invalid_argument("images and audio are not supported by this library yet, use llama-server");
}

// What a model of this decision type can output; copied from server-common.cpp.
inline std::vector<std::string> server_model_output_modalities(common_decision_type decision_type, bool has_cap_chat, bool has_cap_tts) {
    switch (decision_type) {
        case COMMON_DECISION_TYPE_OPENJEV:
        case COMMON_DECISION_TYPE_LEV:
        case COMMON_DECISION_TYPE_KEV:
        case COMMON_DECISION_TYPE_NIMBLE:
        case COMMON_DECISION_TYPE_LAYA:
        case COMMON_DECISION_TYPE_CLEF:
        case COMMON_DECISION_TYPE_PPLX_DECIDER:
        case COMMON_DECISION_TYPE_VLLM_DECISION2:
        case COMMON_DECISION_TYPE_VLLM_D3:
        case COMMON_DECISION_TYPE_LFM2_D1:
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
        case COMMON_DECISION_TYPE_LFM3_D1:
        case COMMON_DECISION_TYPE_LFM3_D1_OMNI:
            return {"decisions"};
        case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            return {"text", "decisions"};
        default:
            if (has_cap_chat || !has_cap_tts) {
                return {"text"};
            }
            return {"audio"};
    }
}
