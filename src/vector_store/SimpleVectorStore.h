#pragma once
#include "vector_store.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <atomic>

// Simple brute-force vector store for demonstration
class SimpleVectorStore : public IVectorStore {
public:
    void add(const std::vector<float>& embedding, const std::string& chunk) override {
        data_[current_index++] = std::make_pair(embedding, chunk);
    }

    void resize(size_t new_size) override {
        data_.resize(new_size);
    }

    std::vector<std::string> query(const std::vector<float>& embedding, size_t top_k) const override {
        // Compute cosine similarity for all stored embeddings
        std::vector<std::pair<float, std::string>> scored;
        for (const auto& [vec, chunk] : data_) {
            float score = cosine_similarity(embedding, vec);
            scored.emplace_back(score, chunk);
        }
        // Sort by score descending
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        // Return top_k chunks
        std::vector<std::string> result;
        for (size_t i = 0; i < std::min(top_k, scored.size()); ++i) {
            result.push_back(scored[i].second);
        }
        return result;
    }

private:
    std::vector<std::pair<std::vector<float>, std::string>> data_;
    std::atomic<size_t> current_index{0};

    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6f);
    }
};
