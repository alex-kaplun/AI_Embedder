#pragma once
#include <vector>

class IVectorStore {
public:
    virtual ~IVectorStore() = default;
    virtual void resize(size_t new_size) = 0;
    virtual void add(const std::vector<float>& embedding, const std::string& chunk) = 0;
    virtual std::vector<std::string> query(const std::vector<float>& embedding, size_t top_k) const = 0;
};
