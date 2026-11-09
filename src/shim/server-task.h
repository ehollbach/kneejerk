// What server-decision.cpp needs from llama.cpp's tools/server/server-task.h: a task is a prompt and
// where in the model's output the answer is read. The decision struct is copied verbatim.
#pragma once

#include "server-common.h"

#include <algorithm>
#include <vector>

enum server_task_type {
    SERVER_TASK_TYPE_DECISION,
};

struct server_task {
    int id    = -1;
    int index = -1;
    server_task_type type = SERVER_TASK_TYPE_DECISION;

    server_tokens tokens;

    // a parent shares the first n_tokens_shared tokens of its prompt with its children (server slots)
    int id_parent = -1;
    std::vector<server_task> child_tasks;
    int32_t n_tokens_shared = 0;

    // where to read the model output of each option, exactly one of the two lists is used
    struct decision {
        std::vector<llama_token> labels;       // logits of these tokens, at the last prompt token
        std::vector<int32_t>     label_groups; // if set, number of labels per output, the output is their max
        std::vector<int32_t>     markers;      // embeddings[column] at these prompt positions
        int32_t                  column = 0;
        // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
        int32_t                  pointer = -1;
        // if set, the prompt must be the only sequence of its batch
        bool                     no_batching = false;

        // if set, the outputs are the pooled embeddings of the sequence, one score per class
        bool                     pooled = false;

        // first prompt position that is read, -1 if none
        int32_t pos_first() const {
            int32_t pos = pointer;
            for (const int32_t marker : markers) {
                pos = pos < 0 ? marker : std::min(pos, marker);
            }
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i] != 0) {
                    return pos < 0 ? (int32_t) i : std::min(pos, (int32_t) i);
                }
            }
            return pos;
        }

        // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
        // the scores are the first n_scores rows of the embeddings
        std::vector<int32_t> order;
        int32_t              n_scores = 0;
    };
    decision decision;

    server_task() = default;
    explicit server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const { return (int32_t) tokens.size(); }
};
