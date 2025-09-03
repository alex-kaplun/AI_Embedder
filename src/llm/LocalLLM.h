#pragma once
#include "llm.h"
#include <string>
#include <vector>

// Stub for LLM using llama.cpp or other local/cloud models
class LocalLLM : public ILLM {
public:
    std::string infer(const std::string& question, const std::vector<std::string>& context) const override {
        // TODO: Integrate llama.cpp or other LLM
        // For now, return dummy answer
        return "[LLM answer would go here]";
    }
};
