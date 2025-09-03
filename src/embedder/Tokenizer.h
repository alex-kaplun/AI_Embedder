#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

// Minimal BERT WordPiece tokenizer (English-focused) sufficient for bge-small-en
// - lowercases
// - splits on whitespace and basic punctuation
// - greedy wordpiece with '##' continuation
// Expects a vocab.txt file with one token per line.
class Tokenizer {
public:
    struct Encoded { std::vector<int64_t> input_ids; std::vector<int64_t> attention_mask; };

    Tokenizer(const std::string& vocab_path,
              int cls_id = 101, int sep_id = 102, int unk_id = 100,
              size_t max_len = 256)
        : cls_id_(cls_id), sep_id_(sep_id), unk_id_(unk_id), max_len_(max_len) {
        load_vocab(vocab_path);
    }

    bool ok() const { return !id_to_token_.empty(); }

    Encoded encode(const std::string& text) const {
        std::vector<std::string> words = basic_tokenize(text);
        std::vector<int64_t> ids;
        ids.reserve(max_len_);
        ids.push_back(cls_id_);
        for (const auto& w : words) {
            auto pieces = wordpiece_tokenize(w);
            for (const auto& p : pieces) {
                ids.push_back(lookup_id(p));
                if (ids.size() + 1 >= max_len_) break; // reserve space for SEP
            }
            if (ids.size() + 1 >= max_len_) break;
        }
        ids.push_back(sep_id_);
        // pad
        std::vector<int64_t> mask(ids.size(), 1);
        ids.resize(max_len_, 0);
        mask.resize(max_len_, 0);
        return { ids, mask };
    }

private:
    void load_vocab(const std::string& path) {
        std::ifstream in(path);
        if (!in) return;
        std::string line;
        int index = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            token_to_id_[line] = index;
            id_to_token_.push_back(line);
            ++index;
        }
    }

    static std::string to_lower(const std::string& s) {
        std::string out(s);
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c){ return (char)std::tolower(c); });
        return out;
    }

    static bool is_delim(char c) {
        return std::isspace((unsigned char)c) || std::ispunct((unsigned char)c);
    }

    static std::vector<std::string> basic_tokenize(const std::string& text) {
        std::string s = to_lower(text);
        std::vector<std::string> out;
        std::string cur;
        for (char c : s) {
            if (is_delim(c)) {
                if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    std::vector<std::string> wordpiece_tokenize(const std::string& word) const {
        std::vector<std::string> pieces;
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            int found_id = -1;
            std::string found_piece;
            while (end > start) {
                std::string substr = word.substr(start, end - start);
                if (start > 0) substr = "##" + substr;
                auto it = token_to_id_.find(substr);
                if (it != token_to_id_.end()) { found_id = it->second; found_piece = substr; break; }
                --end;
            }
            if (found_id == -1) { pieces.push_back("[UNK]"); break; }
            pieces.push_back(found_piece);
            start = end;
        }
        return pieces;
    }

    int64_t lookup_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) return it->second;
        return unk_id_;
    }

    std::unordered_map<std::string,int> token_to_id_;
    std::vector<std::string> id_to_token_;
    int cls_id_;
    int sep_id_;
    int unk_id_;
    size_t max_len_;
};
