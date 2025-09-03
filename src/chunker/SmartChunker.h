#pragma once
#include "chunker.h"
#include "../embedder/Tokenizer.h"
#include <vector>
#include <string>
#include <memory>

class SmartChunker : public IChunker {
public:
    SmartChunker(std::shared_ptr<Tokenizer> tokenizer,
                 size_t max_tokens = 400,
                 size_t overlap_tokens = 80);
    std::vector<std::string> chunk(const std::string& text) const override;
private:
    std::shared_ptr<Tokenizer> tokenizer_;
    size_t max_tokens_;
    size_t overlap_tokens_;
};
