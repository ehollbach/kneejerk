#include "kneejerk.hpp"

#include "json.h"

#include <new>
#include <stdexcept>

namespace kneejerk {

namespace {

// the malloc'd answer of the C API as a string
std::string take(char * s) {
    if (!s) {
        throw std::bad_alloc();
    }
    std::string out = s;
    systemone_free_string(s);
    return out;
}

} // namespace

Assistant::Assistant(const std::string & model_path, Model model_type, systemone_params params)
    : so(nullptr, systemone_free), type(model_type) {
    params.model = model_type == Model::Chat ? SYSTEMONE_MODEL_CHAT : SYSTEMONE_MODEL_DECISION;
    so.reset(systemone_load(model_path.c_str(), &params));
    if (!so) {
        throw std::runtime_error("failed to load " + model_path);
    }
}

std::string Assistant::family() const {
    return systemone_model_type(so.get());
}

std::string Assistant::decide(const std::string & request_json) {
    return take(systemone_decide(so.get(), request_json.c_str()));
}

std::string Assistant::chat_json(const std::string & request_json) {
    return take(systemone_chat(so.get(), request_json.c_str()));
}

std::string Assistant::chat(const std::string & user_message, int max_tokens) {
    common_json message = common_json::object();
    message["role"]    = "user";
    message["content"] = user_message;
    common_json messages = common_json::array();
    messages.push_back(message);
    common_json request = common_json::object();
    request["messages"]   = messages;
    request["max_tokens"] = max_tokens;

    const common_json response = common_json::parse(chat_json(request.dump_safe()));
    if (response.contains("error")) {
        throw std::runtime_error(response.at("error").at("message").get<std::string>());
    }
    return response.at("message").at("content").get<std::string>();
}

} // namespace kneejerk
