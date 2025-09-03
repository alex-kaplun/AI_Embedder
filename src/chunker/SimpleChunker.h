#pragma once
#include "chunker.h"
#include <regex>
#include <vector>
#include <string>

class SimpleChunker : public IChunker {
public:
    // Splits text into paragraphs (double newline) or sentences (period)
    std::vector<std::string> chunk(const std::string& text) const override {
        std::vector<std::string> chunks;
    // Use raw string to avoid C++ escape issues; \s is a regex whitespace class
    std::regex re(R"((.*?\n\n|.*?[.!?]\s))");
        auto begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto end = std::sregex_iterator();
        for (auto i = begin; i != end; ++i) {
            std::string chunk = (*i).str();
            if (!chunk.empty()) chunks.push_back(chunk);
        }
        return chunks;
    }
};
