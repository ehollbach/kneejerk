// kneejerk: the C++ API of libsystemone. One model per Assistant, loaded for decisions (default) or for chat.
//
//   kneejerk::Assistant judge("d1-2B-Q4_K_M.gguf");                         // Model::Decision
//   kneejerk::Assistant talker("qwen1_5-0_5b-chat-q2_k.gguf", kneejerk::Model::Chat);
//
//   std::string answer = judge.decide(R"({"state": "...", "questions": {...}})");
//   std::string reply  = talker.chat("Summarise my last message");
#pragma once

#include "systemone.h"

#include <memory>
#include <string>

namespace kneejerk {

enum class Model {
    Decision, // typed questions in, calibrated probabilities out (systemone_decide)
    Chat,     // text generation with the chat template of the model (systemone_chat)
};

class Assistant {
public:
    // throws std::runtime_error if the model cannot be loaded; params.model is set from model_type
    explicit Assistant(const std::string & model_path, Model model_type = Model::Decision,
                       systemone_params params = systemone_default_params());

    Model model_type() const { return type; }

    // "lfm3_d1", "openjev", ... for a decision model ("none" if it is not one), "chat" for a chat model
    std::string family() const;

    // JSON in, JSON out, errors as {"error": ...}: the bodies of systemone_decide() and systemone_chat()
    std::string decide(const std::string & request_json);
    std::string chat_json(const std::string & request_json);

    // one user message in, the reply text out; throws std::runtime_error on error
    std::string chat(const std::string & user_message, int max_tokens = 256);

private:
    std::unique_ptr<systemone, void (*)(systemone *)> so;
    Model type;
};

} // namespace kneejerk
