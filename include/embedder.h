#pragma once
#include <vector>
#include <string>

class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    virtual std::vector<float> embed(const std::string& text) const = 0;
};
