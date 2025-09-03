#pragma once
#include <string>
#include <vector>

class IChunker {
public:
    virtual ~IChunker() = default;
    virtual std::vector<std::string> chunk(const std::string& text) const = 0;
};
