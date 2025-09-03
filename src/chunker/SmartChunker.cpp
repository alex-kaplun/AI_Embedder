#include "SmartChunker.h"
#include <regex>
#include <sstream>
#include <iostream>

SmartChunker::SmartChunker(std::shared_ptr<Tokenizer> tokenizer, size_t max_tokens, size_t overlap_tokens)
    : tokenizer_(std::move(tokenizer)), max_tokens_(max_tokens), overlap_tokens_(overlap_tokens) {}

std::vector<std::string> SmartChunker::chunk(const std::string& text) const {
    // Split into sentences (simple regex, can be improved)
    std::regex sentence_re(R"(([^.!?\n]+[.!?]\s*)|([^.!?\n]+$))");
    std::sregex_iterator it(text.begin(), text.end(), sentence_re);
    std::sregex_iterator end;
    std::cout<<"[Chunking]: splitting text into sentences..."<<std::endl;
    std::vector<std::string> sentences;
    for (; it != end; ++it) {
        std::string s = it->str();
        if (!s.empty()) sentences.push_back(s);
    }
    std::cout << "  - Found " << sentences.size() << " sentence(s)." << std::endl;
    // Greedily pack sentences into chunks up to max_tokens_
    std::vector<std::string> chunks;
    size_t i = 0;
    const size_t total = sentences.size();
    size_t last_pct = 0;
    std::cout<<"[Chunking]: packing sentences into chunks..."<<std::endl;
    size_t chunk_count = 0;
    while (i < sentences.size()) {
        std::vector<std::string> chunk_sents;
        size_t tokens = 0;
        size_t start = i;
        while (i < sentences.size()) {
            auto enc = tokenizer_->encode(sentences[i]);
            // Count real tokens using attention_mask (number of 1s), not padded length
            size_t sent_tokens = 0;
            for (auto v : enc.attention_mask) if (v) ++sent_tokens;
            if (sent_tokens == 0) { ++i; continue; }
            if (tokens + sent_tokens > max_tokens_ && !chunk_sents.empty()) break;
            // If a single sentence exceeds max_tokens_, still take it alone to guarantee progress
            if (chunk_sents.empty() && sent_tokens > max_tokens_) {
                chunk_sents.push_back(sentences[i]);
                tokens += sent_tokens;
                ++i;
                break;
            }
            chunk_sents.push_back(sentences[i]);
            tokens += sent_tokens;
            ++i;
        }
        // Merge sentences into chunk
        std::ostringstream oss;
        for (const auto& s : chunk_sents) oss << s << " ";
        chunks.push_back(oss.str());
        ++chunk_count;
        std::cout << "  - Packed chunk " << chunk_count << " (" << tokens << " tokens)" << std::endl;
        // Overlap: step back by overlap_tokens worth of sentences
        if (i < sentences.size() && overlap_tokens_ > 0) {
            size_t overlap = 0;
            size_t j = i - 1;
            while (j >= start && overlap < overlap_tokens_) {
                auto enc = tokenizer_->encode(sentences[j]);
                size_t sent_tokens = 0; for (auto v : enc.attention_mask) if (v) ++sent_tokens;
                overlap += sent_tokens;
                if (j == 0) break;
                --j;
            }
            // Ensure forward progress even if computed j+1 equals current i
            size_t next_i = j + 1;
            if (next_i <= start) next_i = start + 1; // move at least one sentence forward
            if (next_i <= i) next_i = i; // keep non-decreasing
            i = next_i;
        }
    }
    std::cout << "  - Chunking complete: " << chunk_count << " chunk(s) packed." << std::endl;
    return chunks;
}
